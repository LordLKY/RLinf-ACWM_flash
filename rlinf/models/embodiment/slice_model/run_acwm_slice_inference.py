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
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from rlinf.models.embodiment.slice_model.common import (
    ensure_output_dir,
    export_acwm_input,
    export_acwm_output,
    load_hydra_config,
    load_slice_sample,
    reset_export_dir,
)
from rlinf.scheduler import Worker
from rlinf.utils.utils import nvtx_range


PROFILE_ITERATIONS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay one profiled Wan ACWM chunk slice outside Ray/RLinf runner."
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
    parser.add_argument("--device", default="cuda")
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


def _set_worker_device(device: torch.device) -> None:
    Worker.torch_device_type = device.type


def _build_env_cfg(cfg, *, num_envs: int, sample_input: dict[str, Any]):
    env_cfg = OmegaConf.create(OmegaConf.to_container(cfg.env.train, resolve=True))
    with open_dict(env_cfg):
        env_cfg.total_num_envs = num_envs
        env_cfg.group_size = int(sample_input.get("group_size", cfg.algorithm.group_size))
        env_cfg.reward_coef = float(
            sample_input.get("reward_coef", cfg.algorithm.get("reward_coef", env_cfg.reward_coef))
        )
        env_cfg.profile = OmegaConf.create(
            {
                "profile_rollout": False,
                "profile_early_stop": False,
                "profile_vla_data": False,
                "profile_acwm_data": False,
            }
        )
        env_cfg.continuous_batching = OmegaConf.create({"enabled": False})
        env_cfg.video_cfg.save_video = False
        env_cfg.auto_reset = False
        env_cfg.chunk = int(sample_input.get("chunk", env_cfg.chunk))
    return env_cfg


def _reconstruct_image_queue(current_obs: torch.Tensor) -> list[list[torch.Tensor]]:
    if current_obs.ndim != 6:
        raise ValueError(f"Expected current_obs [B,C,V,T,H,W], got {current_obs.shape}")
    num_envs, _, views, time_steps, _, _ = current_obs.shape
    if views != 1:
        raise ValueError(f"Only single-view WanEnv slices are supported, got V={views}")
    image_queue: list[list[torch.Tensor]] = []
    for env_idx in range(num_envs):
        frames = []
        for t in range(time_steps):
            frames.append(current_obs[env_idx, :, 0, t : t + 1].detach().cpu())
        image_queue.append(frames)
    return image_queue


def _restore_env_state(env: Any, sample_input: dict[str, Any], device: torch.device) -> None:
    current_obs = sample_input["current_obs"].to(device)
    condition_action = sample_input["condition_action"].to(device)

    env.current_obs = current_obs
    env.condition_action = condition_action
    env.image_queue = _reconstruct_image_queue(current_obs)
    env.reset_state_ids = sample_input["reset_state_ids"].to(device)
    env.task_descriptions = list(sample_input["task_descriptions"])
    env.init_ee_poses = list(sample_input["init_ee_poses"])
    env.elapsed_steps = int(sample_input["elapsed_steps"])
    env.chunk = int(sample_input["chunk"])
    env._is_start = False


def _normalize_actions(policy_output_action: Any) -> Any:
    if isinstance(policy_output_action, torch.Tensor):
        return policy_output_action.detach().cpu().numpy()
    if isinstance(policy_output_action, np.ndarray):
        return policy_output_action
    raise TypeError(f"Unsupported policy_output_action type: {type(policy_output_action)}")


def _chunk_step_once(env: Any, policy_output_action: Any, device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad(), nvtx_range("slice/acwm_chunk_step"):
        extracted_obs_list, rewards, terminations, truncations, infos_list = env.chunk_step(
            policy_output_action
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return extracted_obs_list, rewards, terminations, truncations, infos_list, elapsed


def run_slice(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_hydra_config(
        config_name=args.config_name,
        config_dir=args.config_dir,
        overrides=args.override,
    )
    sample = load_slice_sample(args.sample_path, expected_kind="acwm")
    if args.profile:
        os.environ.setdefault("RLINF_USE_NVTX", "1")
    elif args.output_dir is None:
        raise ValueError("--output-dir is required unless --profile is enabled")
    output_dir = ensure_output_dir(args.output_dir) if args.output_dir is not None else None
    sample_input = sample["payload"]["input"]
    reference = sample["payload"].get("output", {})

    device = torch.device(args.device)
    _set_worker_device(device)

    from rlinf.envs.world_model.world_model_wan_env import WanEnv

    num_envs = int(sample_input["current_obs"].shape[0])
    env_cfg = _build_env_cfg(cfg, num_envs=num_envs, sample_input=sample_input)
    env = WanEnv(
        env_cfg,
        num_envs=num_envs,
        seed_offset=0,
        total_num_processes=1,
        record_metrics=True,
    )
    _restore_env_state(env, sample_input, device)
    policy_output_action = _normalize_actions(sample_input["policy_output_action"])

    warning_messages = []
    if bool(env.use_rel_reward):
        warning_messages.append(
            "env.use_rel_reward=True but profile_acwm_data did not save "
            "prev_step_reward; replayed reward diffs may not match exactly."
        )

    if args.profile:
        timings = []
        for _ in range(PROFILE_ITERATIONS):
            _restore_env_state(env, sample_input, device)
            *_, elapsed = _chunk_step_once(env, policy_output_action, device)
            timings.append(elapsed)
        timing = {
            "profile": True,
            "iterations": PROFILE_ITERATIONS,
            "elapsed_seconds_total": float(sum(timings)),
            "elapsed_seconds_mean": float(sum(timings) / len(timings)),
            "elapsed_seconds_min": float(min(timings)),
            "elapsed_seconds_max": float(max(timings)),
            "batch_size": num_envs,
            "chunk": int(env.chunk),
            "num_inference_steps": int(env.num_inference_steps),
            "nvtx_enabled": os.environ.get("RLINF_USE_NVTX", "0"),
        }
        return {"metadata": {"warnings": warning_messages}, "timing": timing, "diff": {}}

    (
        extracted_obs_list,
        rewards,
        terminations,
        truncations,
        infos_list,
        elapsed,
    ) = _chunk_step_once(env, policy_output_action, device)

    predicted = {
        "current_obs": env.current_obs,
        "extracted_obs": extracted_obs_list[0],
        "chunk_rewards_tensors": rewards,
        "chunk_terminations": terminations,
        "chunk_truncations": truncations,
        "past_dones": torch.logical_or(
            terminations.detach().bool().any(dim=1),
            truncations.detach().bool().any(dim=1),
        ),
        "infos": infos_list[0],
    }

    metadata = {
        "sample_path": str(args.sample_path),
        "sample_metadata": sample["metadata"],
        "config_name": args.config_name,
        "config_dir": str(args.config_dir) if args.config_dir else None,
        "overrides": args.override,
        "device": str(device),
        "num_envs": num_envs,
        "num_inference_steps": int(env.num_inference_steps),
        "elapsed_seconds": elapsed,
        "warnings": warning_messages,
    }

    export_acwm_input(sample_input, reset_export_dir(output_dir / "input"))
    export_acwm_output(predicted, reset_export_dir(output_dir / "predicted_output"))
    export_acwm_output(reference, reset_export_dir(output_dir / "reference_output"))

    timing = {
        "profile": False,
        "elapsed_seconds": elapsed,
        "batch_size": num_envs,
        "chunk": int(env.chunk),
        "num_inference_steps": int(env.num_inference_steps),
    }
    return {"metadata": metadata, "timing": timing, "diff": {}}


def main() -> None:
    result = run_slice(parse_args())
    print(OmegaConf.to_yaml(OmegaConf.create(result["timing"])))
    for message in result["metadata"].get("warnings", []):
        print(f"WARNING: {message}")
    if result["diff"]:
        print(f"Saved diff summary with {len(result['diff'])} entries.")


if __name__ == "__main__":
    main()
