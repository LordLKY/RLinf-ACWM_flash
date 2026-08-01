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
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict
from transformers.generation import TopKLogitsWarper

from rlinf.models import get_model
from rlinf.models.embodiment.slice_model.common import (
    bytes_to_gb,
    cuda_memory_snapshot,
    default_local_src_dir,
    export_vla_input,
    export_vla_output,
    is_cuda_oom,
    load_hydra_config,
    load_slice_sample,
    parse_batch_sizes,
    prepend_local_src,
    reset_export_dir,
    reset_cuda_peak_memory,
    save_pt,
    sampling_kwargs,
    scale_nested_batch,
)
from rlinf.utils.utils import compute_logprobs_from_logits, nvtx_range


PROFILE_ITERATIONS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay one profiled OpenVLA-OFT rollout slice outside Ray/RLinf runner."
        )
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
    parser.add_argument(
        "--profile-scale",
        action="store_true",
        help="Sweep batch sizes and report latency per sample and CUDA peak memory.",
    )
    parser.add_argument(
        "--profile-scale-modules",
        action="store_true",
        help=(
            "With --profile-scale, report OpenVLA-OFT module-level scale results "
            "for image_embedding and after_image_embedding instead of whole-model results."
        ),
    )
    parser.add_argument(
        "--scale-batch-sizes",
        default="1,2,4,8,16,32",
        help="Comma-separated batch sizes for --profile-scale.",
    )
    parser.add_argument("--profile-scale-iters", type=int, default=PROFILE_ITERATIONS)
    parser.add_argument("--profile-scale-warmup", type=int, default=1)
    parser.add_argument(
        "--profile-scale-stop-on-oom",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop the batch-size sweep after the first CUDA OOM.",
    )
    parser.add_argument(
        "--profile-scale-empty-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Call torch.cuda.empty_cache() between profile-scale batch sizes.",
    )
    parser.add_argument(
        "--save-pt",
        action="store_true",
        help="Also save predicted.pt and reference.pt under output-dir.",
    )
    parser.add_argument(
        "--local-prismatic-src",
        nargs="?",
        const=default_local_src_dir("openvla_oft"),
        default=None,
        type=Path,
        help=(
            "Prepend a local source directory containing the prismatic package. "
            "If passed without a value, uses slice_model/local_src/openvla_oft."
        ),
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


def _source_batch_size_from_forward_inputs(forward_inputs: dict[str, Any]) -> int:
    for key in ("input_ids", "attention_mask", "pixel_values"):
        value = forward_inputs.get(key)
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return int(value.shape[0])
    raise ValueError("Could not infer source batch size from forward_inputs.")


def _time_cuda_module(
    name: str,
    func,
    *,
    device: torch.device,
) -> tuple[Any, dict[str, Any]]:
    if device.type == "cuda":
        torch.cuda.synchronize()
        reset_cuda_peak_memory(device)
    start = time.perf_counter()
    with torch.no_grad(), nvtx_range(f"slice/openvla_module/{name}"):
        output = func()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_memory = cuda_memory_snapshot(device)
    metric = {"elapsed_seconds": float(elapsed)}
    if peak_memory is not None:
        metric.update(
            {
                "peak_memory_allocated_gb": bytes_to_gb(
                    peak_memory["max_allocated_bytes"]
                ),
                "peak_memory_reserved_gb": bytes_to_gb(
                    peak_memory["max_reserved_bytes"]
                ),
            }
        )
    return output, metric


def _summarize_module_records(
    records: list[dict[str, Any]],
    *,
    batch_size: int,
) -> dict[str, Any]:
    timings = [float(record["elapsed_seconds"]) for record in records]
    summary = {
        "calls": len(records),
        "elapsed_seconds_total": float(sum(timings)),
        "elapsed_seconds_mean": float(sum(timings) / len(timings)),
        "elapsed_seconds_min": float(min(timings)),
        "elapsed_seconds_max": float(max(timings)),
        "latency_per_sample_seconds": float(sum(timings) / len(timings) / batch_size),
    }
    allocated = [
        float(record["peak_memory_allocated_gb"])
        for record in records
        if "peak_memory_allocated_gb" in record
    ]
    reserved = [
        float(record["peak_memory_reserved_gb"])
        for record in records
        if "peak_memory_reserved_gb" in record
    ]
    if allocated:
        summary["peak_memory_allocated_gb"] = round(max(allocated), 3)
    if reserved:
        summary["peak_memory_reserved_gb"] = round(max(reserved), 3)
    return summary


def _predict_openvla_modules_once(
    model,
    forward_inputs: dict[str, Any],
    kwargs: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, dict[str, Any]]]:
    input_ids = forward_inputs["input_ids"]
    attention_mask = forward_inputs["attention_mask"]
    pixel_values = forward_inputs["pixel_values"]
    do_sample = bool(kwargs["do_sample"])

    assert torch.all(input_ids[:, 0] == 1)
    assert torch.all(attention_mask[:, 0] == 1)
    assert torch.all(input_ids[:, -1] == 29871)
    assert torch.all(attention_mask[:, -1] == 1)

    n_prompt_tokens = input_ids.shape[-1] - 1
    n_patches = (
        model.vision_backbone.get_num_patches()
        * model.vision_backbone.get_num_images_in_input()
    )

    input_ids, attention_mask = model._prepare_input_for_action_prediction(
        input_ids, attention_mask
    )
    assert torch.all(
        attention_mask[:, -1 - model.action_dim * model.num_action_chunks :] == 1
    )

    (mm_embeddings, mm_attention_mask), embedding_metric = _time_cuda_module(
        "image_embedding",
        lambda: model._build_embedding(input_ids, attention_mask, pixel_values),
        device=device,
    )

    def run_after_image_embedding():
        multimodal_position_ids = mm_attention_mask.cumsum(dim=1) - 1
        outputs = model.language_model(
            input_ids=None,
            attention_mask=mm_attention_mask,
            position_ids=multimodal_position_ids,
            past_key_values=None,
            inputs_embeds=mm_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        last_hidden_states = outputs.hidden_states[-1]
        assert last_hidden_states.shape[1] == mm_embeddings.shape[1]

        logits_tensor = outputs.logits[
            :,
            n_patches + n_prompt_tokens : n_patches
            + n_prompt_tokens
            + model.action_dim * model.num_action_chunks,
            :,
        ]
        last_hidden_states = last_hidden_states[
            :, -model.action_dim * model.num_action_chunks - 1 : -1
        ]

        logits_tensor[..., : model.vocab_size - model.config.n_action_bins] = -torch.inf
        logits_tensor[..., model.vocab_size :] = -torch.inf

        if do_sample:
            processed_logits_tensor = logits_tensor / kwargs["temperature"]
            top_k = min(kwargs["top_k"], processed_logits_tensor.size(-1))
            if top_k > 0:
                logits_warper = TopKLogitsWarper(top_k)
                processed_logits_tensor = logits_warper(None, processed_logits_tensor)
            processed_logprob_tensor = F.log_softmax(processed_logits_tensor, dim=-1)
            probs_tensor = torch.exp(processed_logprob_tensor)
            probs_flat = probs_tensor.view(-1, processed_logprob_tensor.shape[-1])
            sample_flat = torch.multinomial(probs_flat, num_samples=1, replacement=True)
            idxs = sample_flat.view(
                processed_logprob_tensor.shape[0],
                processed_logprob_tensor.shape[1],
            )
        else:
            processed_logits_tensor = logits_tensor
            idxs = processed_logits_tensor.argmax(dim=-1)

        assert torch.all(idxs >= model.vocab_size - model.config.n_action_bins)
        assert torch.all(idxs < model.vocab_size)

        chunk_action_tokens = idxs.reshape(-1, model.action_dim)
        predicted_action_token_ids = chunk_action_tokens.cpu().numpy()
        discretized_actions = model.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(
            discretized_actions - 1,
            a_min=0,
            a_max=model.bin_centers.shape[0] - 1,
        )
        normalized_actions = np.asarray(
            [model.bin_centers[da] for da in discretized_actions]
        )
        normalized_actions = normalized_actions.reshape(-1, model.action_dim)
        actions = model._unnormalize_actions(normalized_actions, model.unnorm_key)
        actions = actions.reshape(idxs.shape)

        action_logits = processed_logits_tensor
        action_logits[..., : model.vocab_size - model.config.n_action_bins] = -torch.inf
        action_logits[..., model.vocab_size :] = -torch.inf
        chunk_logprobs = compute_logprobs_from_logits(logits=action_logits, target=idxs)

        if hasattr(model, "value_head"):
            hidden_features = last_hidden_states[
                :, -model.action_dim * model.num_action_chunks
            ]
            chunk_values = model.value_head(hidden_features)
        else:
            chunk_values = torch.zeros_like(chunk_logprobs[..., :1])
        del chunk_values

        return torch.as_tensor(
            actions.reshape(-1, model.num_action_chunks, model.action_dim)
        )

    actions, after_metric = _time_cuda_module(
        "after_image_embedding",
        run_after_image_embedding,
        device=device,
    )
    return actions, {
        "image_embedding": embedding_metric,
        "after_image_embedding": after_metric,
    }


def _profile_scale_forward_inputs(
    *,
    model,
    forward_inputs: dict[str, Any],
    kwargs: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
    local_prismatic_src: Path | None,
) -> dict[str, Any]:
    if args.profile_scale_iters <= 0:
        raise ValueError("--profile-scale-iters must be positive.")
    if args.profile_scale_warmup < 0:
        raise ValueError("--profile-scale-warmup must be non-negative.")

    source_batch_size = _source_batch_size_from_forward_inputs(forward_inputs)
    batch_sizes = parse_batch_sizes(args.scale_batch_sizes)
    results = []

    for batch_size in batch_sizes:
        if device.type == "cuda" and args.profile_scale_empty_cache:
            torch.cuda.empty_cache()
        scaled_forward_inputs = scale_nested_batch(
            forward_inputs,
            source_batch_size=source_batch_size,
            target_batch_size=batch_size,
        )
        result: dict[str, Any] = {
            "batch_size": batch_size,
            "source_batch_size": source_batch_size,
            "status": "ok",
        }
        try:
            for _ in range(args.profile_scale_warmup):
                _predict_forward_inputs_once(model, scaled_forward_inputs, kwargs, device)

            reset_cuda_peak_memory(device)
            timings = []
            actions_for_shape = None
            for _ in range(args.profile_scale_iters):
                actions, _, elapsed = _predict_forward_inputs_once(
                    model, scaled_forward_inputs, kwargs, device
                )
                timings.append(elapsed)
                actions_for_shape = actions
            peak_memory = cuda_memory_snapshot(device)

            if isinstance(actions_for_shape, np.ndarray):
                actions_for_shape = torch.from_numpy(actions_for_shape)
            result.update(
                {
                    "iterations": int(args.profile_scale_iters),
                    "warmup_iterations": int(args.profile_scale_warmup),
                    "elapsed_seconds_total": float(sum(timings)),
                    "elapsed_seconds_mean": float(sum(timings) / len(timings)),
                    "elapsed_seconds_min": float(min(timings)),
                    "elapsed_seconds_max": float(max(timings)),
                    "latency_per_sample_seconds": float(
                        sum(timings) / len(timings) / batch_size
                    ),
                    "num_action_chunks": int(actions_for_shape.shape[1])
                    if actions_for_shape is not None and actions_for_shape.ndim >= 2
                    else None,
                }
            )
            if peak_memory is not None:
                result.update(
                    {
                        "peak_memory_allocated_gb": bytes_to_gb(
                            peak_memory["max_allocated_bytes"]
                        ),
                        "peak_memory_reserved_gb": bytes_to_gb(
                            peak_memory["max_reserved_bytes"]
                        ),
                    }
                )
        except Exception as exc:
            if not is_cuda_oom(exc):
                raise
            result.update({"status": "oom", "error": str(exc)})
            results.append(result)
            del scaled_forward_inputs
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if args.profile_scale_stop_on_oom:
                break
            continue

        results.append(result)
        del scaled_forward_inputs

    return {
        "profile": True,
        "profile_scale": True,
        "profile_target": "vla_model_inference_only",
        "batch_sizes": batch_sizes,
        "source_batch_size": source_batch_size,
        "iterations": int(args.profile_scale_iters),
        "warmup_iterations": int(args.profile_scale_warmup),
        "stop_on_oom": bool(args.profile_scale_stop_on_oom),
        "empty_cache": bool(args.profile_scale_empty_cache),
        "nvtx_enabled": os.environ.get("RLINF_USE_NVTX", "0"),
        "local_prismatic_src": str(local_prismatic_src) if local_prismatic_src else None,
        "results": results,
    }


def _profile_scale_openvla_modules(
    *,
    model,
    forward_inputs: dict[str, Any],
    kwargs: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
    local_prismatic_src: Path | None,
) -> dict[str, Any]:
    if args.profile_scale_iters <= 0:
        raise ValueError("--profile-scale-iters must be positive.")
    if args.profile_scale_warmup < 0:
        raise ValueError("--profile-scale-warmup must be non-negative.")

    source_batch_size = _source_batch_size_from_forward_inputs(forward_inputs)
    batch_sizes = parse_batch_sizes(args.scale_batch_sizes)
    results = []

    for batch_size in batch_sizes:
        if device.type == "cuda" and args.profile_scale_empty_cache:
            torch.cuda.empty_cache()
        scaled_forward_inputs = scale_nested_batch(
            forward_inputs,
            source_batch_size=source_batch_size,
            target_batch_size=batch_size,
        )
        result: dict[str, Any] = {
            "batch_size": batch_size,
            "source_batch_size": source_batch_size,
            "status": "ok",
        }
        try:
            for _ in range(args.profile_scale_warmup):
                _predict_openvla_modules_once(
                    model, scaled_forward_inputs, kwargs, device
                )

            module_records = {
                "image_embedding": [],
                "after_image_embedding": [],
            }
            actions_for_shape = None
            for _ in range(args.profile_scale_iters):
                actions, records = _predict_openvla_modules_once(
                    model, scaled_forward_inputs, kwargs, device
                )
                actions_for_shape = actions
                for module_name, record in records.items():
                    module_records[module_name].append(record)

            if isinstance(actions_for_shape, np.ndarray):
                actions_for_shape = torch.from_numpy(actions_for_shape)
            result.update(
                {
                    "iterations": int(args.profile_scale_iters),
                    "warmup_iterations": int(args.profile_scale_warmup),
                    "num_action_chunks": int(actions_for_shape.shape[1])
                    if actions_for_shape is not None and actions_for_shape.ndim >= 2
                    else None,
                    "modules": {
                        module_name: _summarize_module_records(
                            records, batch_size=batch_size
                        )
                        for module_name, records in module_records.items()
                    },
                }
            )
        except Exception as exc:
            if not is_cuda_oom(exc):
                raise
            result.update({"status": "oom", "error": str(exc)})
            results.append(result)
            del scaled_forward_inputs
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if args.profile_scale_stop_on_oom:
                break
            continue

        results.append(result)
        del scaled_forward_inputs

    return {
        "profile": True,
        "profile_scale": True,
        "profile_scale_modules": True,
        "profile_target": "openvlaoft_modules",
        "modules": ["image_embedding", "after_image_embedding"],
        "batch_sizes": batch_sizes,
        "source_batch_size": source_batch_size,
        "iterations": int(args.profile_scale_iters),
        "warmup_iterations": int(args.profile_scale_warmup),
        "stop_on_oom": bool(args.profile_scale_stop_on_oom),
        "empty_cache": bool(args.profile_scale_empty_cache),
        "nvtx_enabled": os.environ.get("RLINF_USE_NVTX", "0"),
        "local_prismatic_src": str(local_prismatic_src) if local_prismatic_src else None,
        "results": results,
    }


def run_slice(args: argparse.Namespace) -> dict[str, Any]:
    if args.profile and args.profile_scale:
        raise ValueError("--profile and --profile-scale are mutually exclusive.")
    if args.profile_scale_modules and not args.profile_scale:
        raise ValueError("--profile-scale-modules requires --profile-scale.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_hydra_config(
        config_name=args.config_name,
        config_dir=args.config_dir,
        overrides=args.override,
    )
    sample = load_slice_sample(args.sample_path, expected_kind="vla")
    if args.profile or args.profile_scale:
        os.environ.setdefault("RLINF_USE_NVTX", "1")
    elif args.output_dir is None:
        raise ValueError("--output-dir is required unless --profile/--profile-scale is enabled")
    output_dir = reset_export_dir(args.output_dir) if args.output_dir is not None else None

    device = torch.device(args.device)
    local_prismatic_src = prepend_local_src(
        args.local_prismatic_src, package_name="prismatic"
    )
    model, model_cfg = _load_model(cfg, args, device)

    env_obs = sample["payload"]["input"]["env_obs"]
    kwargs = sampling_kwargs(cfg, args.mode)

    if args.profile_scale:
        _, warmup_result, _ = _predict_once(model, env_obs, kwargs, device)
        forward_inputs = warmup_result["forward_inputs"]
        if args.profile_scale_modules:
            timing = _profile_scale_openvla_modules(
                model=model,
                forward_inputs=forward_inputs,
                kwargs=kwargs,
                device=device,
                args=args,
                local_prismatic_src=local_prismatic_src,
            )
            return {"metadata": {}, "timing": timing, "diff": {}}
        timing = _profile_scale_forward_inputs(
            model=model,
            forward_inputs=forward_inputs,
            kwargs=kwargs,
            device=device,
            args=args,
            local_prismatic_src=local_prismatic_src,
        )
        return {"metadata": {}, "timing": timing, "diff": {}}

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
            "local_prismatic_src": str(local_prismatic_src)
            if local_prismatic_src
            else None,
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
        "local_prismatic_src": str(local_prismatic_src)
        if local_prismatic_src
        else None,
        "model_source": args.model_source,
        "model_type": str(model_cfg.model_type),
        "model_path": str(model_cfg.model_path),
        "ckpt_path": str(args.ckpt_path) if args.ckpt_path else None,
        "sampling_kwargs": kwargs,
        "elapsed_seconds": elapsed,
    }

    if args.save_pt:
        save_pt(predicted, output_dir / "predicted.pt")
        save_pt(reference, output_dir / "reference.pt")
    else:
        for stale_pt in ("predicted.pt", "reference.pt"):
            (output_dir / stale_pt).unlink(missing_ok=True)

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
