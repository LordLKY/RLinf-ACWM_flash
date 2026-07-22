# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import copy
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from rlinf.models import get_model
from rlinf.models.embodiment.slice_model.common import (
    ensure_output_dir,
    export_vla_input,
    export_vla_output,
    load_hydra_config,
    load_slice_sample,
    reset_export_dir,
    sampling_kwargs,
)
from rlinf.utils.utils import nvtx_range


PROFILE_ITERATIONS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay one profiled VLA rollout slice outside Ray/RLinf runner."
    )
    parser.add_argument("--sample-path", required=True, type=Path)
    parser.add_argument("--output-dir", default=None, type=Path)
    parser.add_argument(
        "--config-name",
        default="wan_libero_spatial_grpo_openvlaoft_ngpu",
        help="Hydra config name under examples/embodiment/config.",
    )
    parser.add_argument("--config-dir", default=None, type=Path)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override. Can be passed multiple times.",
    )
    parser.add_argument(
        "--model-source",
        choices=["actor", "rollout"],
        default="actor",
        help=(
            "Base model config to use. Training rollout uses actor.model, then "
            "overrides model_path/precision from rollout.model."
        ),
    )
    parser.add_argument("--ckpt-path", default=None, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare predicted outputs with reference outputs in the sample.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run the slice inference 10 times without exporting outputs.",
    )
    return parser.parse_args()


def _build_model_cfg(cfg, model_source: str):
    if model_source == "actor" and cfg.get("actor", None) is not None:
        model_cfg = copy.deepcopy(cfg.actor.model)
    else:
        model_cfg = copy.deepcopy(cfg.rollout.model)

    with open_dict(model_cfg):
        model_cfg.model_path = cfg.rollout.model.model_path
        model_cfg.precision = cfg.rollout.model.precision
        model_cfg.load_to_device = False
    return model_cfg


def _load_model(cfg, args: argparse.Namespace, device: torch.device):
    model_cfg = _build_model_cfg(cfg, args.model_source)
    model = get_model(model_cfg)
    if model is None:
        raise ValueError(f"Could not build model for model_type={model_cfg.model_type}")

    if args.ckpt_path is not None:
        state_dict = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model, model_cfg


def _predict_once(model, env_obs, kwargs: dict[str, Any], device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad(), nvtx_range("slice/vla_predict"):
        actions, result = model.predict_action_batch(env_obs=env_obs, **kwargs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return actions, result, elapsed


def _predict_forward_inputs_once(
    model, forward_inputs: dict[str, Any], kwargs: dict[str, Any], device: torch.device
):
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad(), nvtx_range("slice/vla_model_inference"):
        actions, result = model.predict_action_batch(
            input_ids=forward_inputs["input_ids"],
            attention_mask=forward_inputs["attention_mask"],
            pixel_values=forward_inputs["pixel_values"],
            env_obs=None,
            **kwargs,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return actions, result, elapsed


def run_slice(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_hydra_config(
        config_name=args.config_name,
        config_dir=args.config_dir,
        overrides=args.override,
    )
    sample = load_slice_sample(args.sample_path, expected_kind="vla")
    if args.profile:
        os.environ.setdefault("RLINF_USE_NVTX", "1")
    elif args.output_dir is None:
        raise ValueError("--output-dir is required unless --profile is enabled")
    output_dir = ensure_output_dir(args.output_dir) if args.output_dir is not None else None

    device = torch.device(args.device)
    model, model_cfg = _load_model(cfg, args, device)

    env_obs = sample["payload"]["input"]["env_obs"]
    kwargs = sampling_kwargs(cfg, args.mode)

    if args.profile:
        _, warmup_result, _ = _predict_once(model, env_obs, kwargs, device)
        forward_inputs = warmup_result["forward_inputs"]
        timings = []
        actions_for_shape = None
        for _ in range(PROFILE_ITERATIONS):
            actions, _, elapsed = _predict_forward_inputs_once(
                model, forward_inputs, kwargs, device
            )
            timings.append(elapsed)
            actions_for_shape = actions
        if isinstance(actions_for_shape, np.ndarray):
            actions_for_shape = torch.from_numpy(actions_for_shape)
        timing = {
            "profile": True,
            "profile_target": "vla_model_inference_only",
            "iterations": PROFILE_ITERATIONS,
            "elapsed_seconds_total": float(sum(timings)),
            "elapsed_seconds_mean": float(sum(timings) / len(timings)),
            "elapsed_seconds_min": float(min(timings)),
            "elapsed_seconds_max": float(max(timings)),
            "batch_size": int(actions_for_shape.shape[0]),
            "num_action_chunks": int(actions_for_shape.shape[1])
            if actions_for_shape.ndim >= 2
            else None,
            "nvtx_enabled": os.environ.get("RLINF_USE_NVTX", "0"),
        }
        return {"metadata": {}, "timing": timing, "diff": {}}

    actions, result, elapsed = _predict_once(model, env_obs, kwargs, device)

    if isinstance(actions, np.ndarray):
        actions_for_save = torch.from_numpy(actions)
    else:
        actions_for_save = actions
    predicted = {
        "actions": actions_for_save,
        "forward_inputs": result.get("forward_inputs"),
        "prev_logprobs": result.get("prev_logprobs"),
        "prev_values": result.get("prev_values"),
        "raw_result": result,
    }
    reference = sample["payload"].get("output", {})

    metadata = {
        "sample_path": str(args.sample_path),
        "sample_metadata": sample["metadata"],
        "config_name": args.config_name,
        "config_dir": str(args.config_dir) if args.config_dir else None,
        "overrides": args.override,
        "mode": args.mode,
        "device": str(device),
        "model_source": args.model_source,
        "model_type": str(model_cfg.model_type),
        "model_path": str(model_cfg.model_path),
        "ckpt_path": str(args.ckpt_path) if args.ckpt_path else None,
        "sampling_kwargs": kwargs,
        "elapsed_seconds": elapsed,
    }

    export_vla_input(sample["payload"].get("input", {}), reset_export_dir(output_dir / "input"))
    export_vla_output(predicted, reset_export_dir(output_dir / "predicted_output"))
    export_vla_output(reference, reset_export_dir(output_dir / "reference_output"))

    timing = {
        "profile": False,
        "elapsed_seconds": elapsed,
        "batch_size": int(actions_for_save.shape[0]),
        "num_action_chunks": int(actions_for_save.shape[1])
        if actions_for_save.ndim >= 2
        else None,
    }
    return {"metadata": metadata, "timing": timing, "diff": {}}


def main() -> None:
    result = run_slice(parse_args())
    print(OmegaConf.to_yaml(OmegaConf.create(result["timing"])))
    if result["diff"]:
        print(f"Saved diff summary with {len(result['diff'])} entries.")


if __name__ == "__main__":
    main()
