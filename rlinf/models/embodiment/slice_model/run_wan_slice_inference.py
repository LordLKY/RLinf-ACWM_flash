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
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from PIL import Image, ImageDraw

from rlinf.models.embodiment.slice_model.common import (
    bytes_to_gb,
    cuda_memory_snapshot,
    default_local_src_dir,
    export_acwm_input,
    export_acwm_output,
    is_cuda_oom,
    load_hydra_config,
    load_slice_sample,
    nested_diff_summary,
    parse_batch_sizes,
    prepend_local_src,
    repo_root,
    reset_export_dir,
    reset_cuda_peak_memory,
    save_json,
    save_image,
    save_pt,
    scale_nested_batch,
    tensor_diff_summary,
)
from rlinf.scheduler import Worker
from rlinf.utils.utils import nvtx_range


PROFILE_ITERATIONS = 10
DEFAULT_DIT_RESIDUAL_DIR = repo_root() / "profile" / "wan_slice" / "dit_residual"
DEFAULT_MIDDLE_RESULT_DIR = repo_root() / "profile" / "wan_slice" / "middle_result"


class DitResidualRecorder:
    def __init__(self, chunk_dir: Path):
        self.chunk_dir = reset_export_dir(chunk_dir)
        self.step_index: int | None = None
        self.saved_steps: set[int] = set()

    def begin_step(self, step_index: int, timestep: torch.Tensor) -> None:
        del timestep
        self.step_index = int(step_index)

    def save_residual(self, residual: torch.Tensor) -> None:
        if self.step_index is None:
            raise RuntimeError("DiT residual recorder received residual before begin_step().")
        if self.step_index in self.saved_steps:
            return
        output_path = self.chunk_dir / f"step_{self.step_index:06d}.pt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(residual.detach().to(device="cpu", dtype=torch.float32), output_path)
        self.saved_steps.add(self.step_index)


class WanMiddleResultRecorder:
    def __init__(self):
        self.records: list[dict[str, Any]] = []

    def save_latents(self, step_index: int, timestep: torch.Tensor, latents: torch.Tensor) -> None:
        timestep_value = (
            timestep.detach().cpu().flatten().tolist()
            if isinstance(timestep, torch.Tensor)
            else timestep
        )
        self.records.append(
            {
                "step_index": int(step_index),
                "timestep": timestep_value,
                "latents": latents.detach().to(device="cpu").clone(),
            }
        )


class WanModuleCallProfiler:
    def __init__(self, device: torch.device):
        self.device = device
        self.records: dict[str, list[dict[str, Any]]] = {
            "dit": [],
            "vae_decode": [],
        }

    def clear(self) -> None:
        for records in self.records.values():
            records.clear()

    def measure(self, module_name: str, func, *args, **kwargs):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            reset_cuda_peak_memory(self.device)
        start = time.perf_counter()
        with torch.no_grad(), nvtx_range(f"slice/wan_module/{module_name}"):
            output = func(*args, **kwargs)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        peak_memory = cuda_memory_snapshot(self.device)
        record = {"elapsed_seconds": float(elapsed)}
        if peak_memory is not None:
            record.update(
                {
                    "peak_memory_allocated_gb": bytes_to_gb(
                        peak_memory["max_allocated_bytes"]
                    ),
                    "peak_memory_reserved_gb": bytes_to_gb(
                        peak_memory["max_reserved_bytes"]
                    ),
                }
            )
        self.records[module_name].append(record)
        return output


def _set_dit_step_recorder(recorder: DitResidualRecorder | None) -> None:
    from diffsynth.pipelines.wan_video_new import set_dit_step_recorder

    set_dit_step_recorder(recorder)


def _set_middle_result_recorder(recorder: WanMiddleResultRecorder | None) -> None:
    from diffsynth.pipelines.wan_video_new import set_middle_result_recorder

    set_middle_result_recorder(recorder)


def _reset_dit_residual_output_dir(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for chunk_path in output_dir.glob("chunk_*"):
        if chunk_path.is_dir():
            shutil.rmtree(chunk_path)
        else:
            chunk_path.unlink()
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay one profiled Wan world-model chunk slice outside Ray/RLinf runner."
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
    parser.add_argument(
        "--profile-scale",
        action="store_true",
        help="Sweep batch sizes and report latency per sample and CUDA peak memory.",
    )
    parser.add_argument(
        "--profile-scale-modules",
        action="store_true",
        help=(
            "With --profile-scale, report Wan module-level scale results for "
            "DiT and VAE decode instead of whole chunk_step results."
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
        "--sequence",
        action="store_true",
        help=(
            "Replay consecutive acwm_*.pt chunks starting from sample-path. "
            "Outputs are written under output-dir/sequence_from_<sample-stem>."
        ),
    )
    parser.add_argument(
        "--sequence-mode",
        choices=["rollout", "teacher_forced"],
        default="rollout",
        help=(
            "rollout keeps WanEnv state rolling forward and only reads later chunk "
            "actions; teacher_forced restores each sample input before the chunk."
        ),
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Maximum number of consecutive chunks to replay in --sequence mode.",
    )
    parser.add_argument(
        "--save-pt",
        action="store_true",
        help="Also save predicted.pt and reference.pt under output-dir.",
    )
    parser.add_argument(
        "--save-output-current-obs-frames",
        action="store_true",
        help=(
            "Also export full current_obs frame windows for predicted/reference outputs. "
            "By default only input current_obs and output extracted_obs images are exported."
        ),
    )
    parser.add_argument(
        "--dump-dit-residuals",
        action="store_true",
        help="Save fp32 DiT block residual tensors for every Wan denoising step.",
    )
    parser.add_argument(
        "--share-initial-noise",
        action="store_true",
        help=(
            "Use one shared Wan initial diffusion noise sample for all batch lanes "
            "and reuse it across sequence chunks with the same shape/seed. "
            "This is intended for slice analysis only."
        ),
    )
    parser.add_argument(
        "--profile-prefix-step",
        action="store_true",
        help=(
            "Run one Wan prefix-step quality experiment: baseline full-batch denoise "
            "and one prefix run whose first --prefix-steps denoise steps share the "
            "reference batch lane's generated latents."
        ),
    )
    parser.add_argument(
        "--prefix-steps",
        type=int,
        default=1,
        help="Number of initial Wan denoise steps to run with a shared prefix.",
    )
    parser.add_argument(
        "--prefix-reference-batch-id",
        type=int,
        default=0,
        help="Batch lane used as the shared denoise prefix representative.",
    )
    parser.add_argument(
        "--profile-middle-result",
        action="store_true",
        help=(
            "Record Wan latents after each denoise step and decode them after "
            "the chunk to visualize progressive denoising."
        ),
    )
    parser.add_argument(
        "--middle-result-dir",
        default=DEFAULT_MIDDLE_RESULT_DIR,
        type=Path,
        help="Directory for --profile-middle-result decoded step frames.",
    )
    parser.add_argument(
        "--middle-result-save-latents",
        action="store_true",
        help="Also save each denoise-step latent as .pt under --middle-result-dir.",
    )
    parser.add_argument(
        "--middle-result-generated-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only export generated frames after the clean context window.",
    )
    parser.add_argument(
        "--middle-result-max-envs",
        type=int,
        default=8,
        help="Maximum batch lanes to decode/export for --profile-middle-result.",
    )
    parser.add_argument(
        "--dit-residual-dir",
        default=DEFAULT_DIT_RESIDUAL_DIR,
        type=Path,
        help="Directory for --dump-dit-residuals output.",
    )
    parser.add_argument(
        "--local-wan-src",
        nargs="?",
        const=default_local_src_dir("wan"),
        default=None,
        type=Path,
        help=(
            "Prepend a local source directory containing the diffsynth package. "
            "If passed without a value, uses slice_model/local_src/wan."
        ),
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


def _enable_shared_initial_noise(env: Any) -> None:
    pipe = env.pipe
    if getattr(pipe, "_rlinf_shared_initial_noise_enabled", False):
        return

    original_generate_noise = pipe.generate_noise
    noise_cache: dict[tuple[Any, ...], torch.Tensor] = {}

    def generate_shared_noise(
        shape,
        seed=None,
        rand_device="cpu",
        rand_torch_dtype=torch.float32,
        device=None,
        torch_dtype=None,
    ):
        shape_tuple = tuple(int(dim) for dim in shape)
        if not shape_tuple or shape_tuple[0] <= 1:
            return original_generate_noise(
                shape_tuple,
                seed=seed,
                rand_device=rand_device,
                rand_torch_dtype=rand_torch_dtype,
                device=device,
                torch_dtype=torch_dtype,
            )

        batch_size = shape_tuple[0]
        base_shape = (1, *shape_tuple[1:])
        key = (
            base_shape,
            seed,
            str(rand_device),
            str(rand_torch_dtype),
            str(device or pipe.device),
            str(torch_dtype or pipe.torch_dtype),
        )
        base_noise = noise_cache.get(key)
        if base_noise is None:
            base_noise = original_generate_noise(
                base_shape,
                seed=seed,
                rand_device=rand_device,
                rand_torch_dtype=rand_torch_dtype,
                device=device,
                torch_dtype=torch_dtype,
            ).detach()
            noise_cache[key] = base_noise.clone()
        return noise_cache[key].expand(batch_size, *base_shape[1:]).clone()

    pipe.generate_noise = generate_shared_noise
    pipe._rlinf_shared_initial_noise_enabled = True


def _set_wan_prefix_steps(env: Any, *, prefix_steps: int, reference_batch_id: int) -> None:
    pipe = env.pipe
    pipe._rlinf_prefix_steps = int(prefix_steps)
    pipe._rlinf_prefix_reference_batch_id = int(reference_batch_id)


def _clear_wan_prefix_steps(env: Any) -> None:
    pipe = env.pipe
    pipe._rlinf_prefix_steps = 0
    pipe._rlinf_prefix_reference_batch_id = 0


def _patch_wan_module_profiler(
    env: Any, profiler: WanModuleCallProfiler
):
    pipe = env.pipe
    original_model_fn = pipe.model_fn
    original_vae_decode = pipe.vae.decode

    def profiled_model_fn(*args, **kwargs):
        return profiler.measure("dit", original_model_fn, *args, **kwargs)

    def profiled_vae_decode(*args, **kwargs):
        return profiler.measure("vae_decode", original_vae_decode, *args, **kwargs)

    pipe.model_fn = profiled_model_fn
    pipe.vae.decode = profiled_vae_decode

    def restore() -> None:
        pipe.model_fn = original_model_fn
        pipe.vae.decode = original_vae_decode

    return restore


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


def _create_wan_env(cfg, sample_input: dict[str, Any], device: torch.device):
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
    return env, num_envs


def _source_batch_size_from_wan_input(sample_input: dict[str, Any]) -> int:
    current_obs = sample_input.get("current_obs")
    if not isinstance(current_obs, torch.Tensor) or current_obs.ndim == 0:
        raise ValueError("Wan sample input must contain current_obs with a batch dimension.")
    return int(current_obs.shape[0])


def _scaled_wan_sample_input(
    sample_input: dict[str, Any],
    *,
    source_batch_size: int,
    target_batch_size: int,
) -> dict[str, Any]:
    scaled_input = scale_nested_batch(
        sample_input,
        source_batch_size=source_batch_size,
        target_batch_size=target_batch_size,
    )
    scaled_input["group_size"] = int(target_batch_size)
    return scaled_input


def _summarize_wan_module_records(
    records: list[dict[str, Any]],
    *,
    batch_size: int,
    iterations: int,
) -> dict[str, Any]:
    if not records:
        return {
            "calls": 0,
            "calls_per_iteration": 0.0,
            "elapsed_seconds_total": 0.0,
            "elapsed_seconds_mean_per_call": None,
            "elapsed_seconds_min_per_call": None,
            "elapsed_seconds_max_per_call": None,
            "elapsed_seconds_mean_per_iteration": 0.0,
            "latency_per_sample_seconds": 0.0,
        }
    timings = [float(record["elapsed_seconds"]) for record in records]
    summary = {
        "calls": len(records),
        "calls_per_iteration": float(len(records) / iterations),
        "elapsed_seconds_total": float(sum(timings)),
        "elapsed_seconds_mean_per_call": float(sum(timings) / len(timings)),
        "elapsed_seconds_min_per_call": float(min(timings)),
        "elapsed_seconds_max_per_call": float(max(timings)),
        "elapsed_seconds_mean_per_iteration": float(sum(timings) / iterations),
        "latency_per_sample_seconds": float(sum(timings) / iterations / batch_size),
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


def _warning_messages_for_env(env: Any) -> list[str]:
    warning_messages = []
    if bool(env.use_rel_reward):
        warning_messages.append(
            "env.use_rel_reward=True but profile_acwm_data did not save "
            "prev_step_reward; replayed reward diffs may not match exactly."
        )
    return warning_messages


def _run_one_chunk(
    env: Any,
    sample_input: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], float]:
    policy_output_action = _normalize_actions(sample_input["policy_output_action"])
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
    return predicted, elapsed


def _export_acwm_chunk(
    *,
    sample_input: dict[str, Any],
    predicted: dict[str, Any],
    reference: dict[str, Any],
    output_dir: Path,
    save_pt_outputs: bool,
    save_output_current_obs_frames: bool,
    save_input_current_obs_frames: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_pt_outputs:
        save_pt(predicted, output_dir / "predicted.pt")
        save_pt(reference, output_dir / "reference.pt")
    else:
        for stale_pt in ("predicted.pt", "reference.pt"):
            (output_dir / stale_pt).unlink(missing_ok=True)

    export_acwm_input(
        sample_input,
        reset_export_dir(output_dir / "input"),
        save_current_obs_frames=save_input_current_obs_frames,
    )
    export_acwm_output(
        predicted,
        reset_export_dir(output_dir / "predicted_output"),
        save_current_obs_frames=save_output_current_obs_frames,
    )
    export_acwm_output(
        reference,
        reset_export_dir(output_dir / "reference_output"),
        save_current_obs_frames=save_output_current_obs_frames,
    )


def _quality_compare_payload(payload: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "current_obs",
        "extracted_obs",
        "chunk_rewards_tensors",
        "chunk_terminations",
        "chunk_truncations",
        "past_dones",
    )
    return {key: payload[key] for key in keys if key in payload}


def _image_diff_by_env(pred: Any, ref: Any) -> dict[str, Any]:
    pred_tensor = torch.as_tensor(pred).detach().cpu().to(torch.float32)
    ref_tensor = torch.as_tensor(ref).detach().cpu().to(torch.float32)
    if tuple(pred_tensor.shape) != tuple(ref_tensor.shape):
        return {
            "comparable": False,
            "reason": "shape_mismatch",
            "pred_shape": list(pred_tensor.shape),
            "ref_shape": list(ref_tensor.shape),
        }
    if pred_tensor.ndim == 0:
        return {"comparable": False, "reason": "scalar_input"}
    diff = pred_tensor - ref_tensor
    flat = diff.reshape(diff.shape[0], -1)
    abs_flat = flat.abs()
    rmse = torch.sqrt((flat * flat).mean(dim=1))
    per_env = [
        {
            "env_id": int(env_id),
            "mean_abs": float(abs_flat[env_id].mean().item()),
            "rmse": float(rmse[env_id].item()),
            "max_abs": float(abs_flat[env_id].max().item()),
        }
        for env_id in range(flat.shape[0])
    ]
    return {
        "comparable": True,
        "shape": list(pred_tensor.shape),
        "overall": {
            "mean_abs": float(abs_flat.mean().item()),
            "rmse": float(torch.sqrt((diff * diff).mean()).item()),
            "max_abs": float(abs_flat.max().item()),
        },
        "per_env": per_env,
    }


def _main_images_diff_by_env(predicted: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    pred_images = predicted.get("extracted_obs", {}).get("main_images")
    ref_images = reference.get("extracted_obs", {}).get("main_images")
    if pred_images is None or ref_images is None:
        return {
            "comparable": False,
            "reason": "missing_extracted_obs_main_images",
        }
    return _image_diff_by_env(pred_images, ref_images)


def _prefix_quality_summary(
    *,
    timing: dict[str, Any],
    prefix: dict[str, Any],
    baseline: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    return {
        "timing": {
            "prefix_steps": timing["prefix_steps"],
            "prefix_reference_batch_id": timing["prefix_reference_batch_id"],
            "batch_size": timing["batch_size"],
            "num_inference_steps": timing["num_inference_steps"],
            "baseline_elapsed_seconds": timing["baseline_elapsed_seconds"],
            "prefix_elapsed_seconds": timing["prefix_elapsed_seconds"],
            "prefix_speedup_vs_baseline": timing["prefix_speedup_vs_baseline"],
        },
        "final_image_diff": {
            "prefix_vs_baseline": _main_images_diff_by_env(prefix, baseline),
            "prefix_vs_reference": _main_images_diff_by_env(prefix, reference),
            "baseline_vs_reference": _main_images_diff_by_env(baseline, reference),
        },
    }


def _format_image_diff_summary(title: str, diff: dict[str, Any]) -> list[str]:
    if not diff.get("comparable"):
        return [f"{title}: not comparable ({diff.get('reason', 'unknown')})"]
    overall = diff["overall"]
    lines = [
        f"{title}:",
        f"  overall MAE  = {overall['mean_abs']:.6g} px",
        f"  overall RMSE = {overall['rmse']:.6g} px",
        f"  overall max  = {overall['max_abs']:.6g} px",
        "  per env MAE/RMSE/max:",
    ]
    for item in diff["per_env"]:
        lines.append(
            "    env{env_id}: {mean_abs:.6g} / {rmse:.6g} / {max_abs:.6g}".format(
                **item
            )
        )
    return lines


def _write_prefix_quality_summary(summary: dict[str, Any], output_dir: Path) -> None:
    save_json(summary, output_dir / "summary.json")
    timing = summary["timing"]
    lines = [
        "Wan prefix-step quality summary",
        "",
        f"prefix_steps: {timing['prefix_steps']}",
        f"prefix_reference_batch_id: {timing['prefix_reference_batch_id']}",
        f"batch_size: {timing['batch_size']}",
        f"num_inference_steps: {timing['num_inference_steps']}",
        f"baseline_elapsed_seconds: {timing['baseline_elapsed_seconds']:.6g}",
        f"prefix_elapsed_seconds: {timing['prefix_elapsed_seconds']:.6g}",
        f"prefix_speedup_vs_baseline: {timing['prefix_speedup_vs_baseline']:.6g}",
        "",
    ]
    image_diff = summary["final_image_diff"]
    lines.extend(
        _format_image_diff_summary(
            "prefix_vs_baseline final image", image_diff["prefix_vs_baseline"]
        )
    )
    lines.append("")
    lines.extend(
        _format_image_diff_summary(
            "prefix_vs_reference final image", image_diff["prefix_vs_reference"]
        )
    )
    lines.append("")
    lines.extend(
        _format_image_diff_summary(
            "baseline_vs_reference final image", image_diff["baseline_vs_reference"]
        )
    )
    summary_path = output_dir / "summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _decode_middle_latents(env: Any, latents: torch.Tensor) -> torch.Tensor:
    pipe = env.pipe
    with torch.no_grad(), nvtx_range("slice/wan_middle_result/vae_decode"):
        videos = pipe.vae.decode(
            latents,
            device=pipe.device,
            tiled=False,
            tile_size=(30, 52),
            tile_stride=(15, 26),
        )
    videos = videos.detach().cpu()
    if videos.ndim == 4:
        videos = videos.unsqueeze(0)
    if videos.ndim != 5:
        raise ValueError(f"Unexpected decoded Wan video tensor shape: {tuple(videos.shape)}")
    return videos


def _make_middle_result_montage(
    *,
    chunk_dir: Path,
    env_id: int,
    frame_indices: list[int],
    step_indices: list[int],
) -> None:
    if not frame_indices or not step_indices:
        return
    thumb_size = 96
    label_h = 22
    label_w = 64
    width = label_w + len(frame_indices) * thumb_size
    height = label_h + len(step_indices) * (thumb_size + label_h)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for col, frame_id in enumerate(frame_indices):
        draw.text((label_w + col * thumb_size + 4, 4), f"f{frame_id}", fill="black")
    for row, step_id in enumerate(step_indices):
        y = label_h + row * (thumb_size + label_h)
        draw.text((4, y + 4), f"step{step_id}", fill="black")
        step_dir = chunk_dir / f"step_{step_id:06d}"
        for col, frame_id in enumerate(frame_indices):
            image_path = step_dir / f"env{env_id:03d}_frame{frame_id:03d}.png"
            if not image_path.exists():
                continue
            image = Image.open(image_path).convert("RGB").resize(
                (thumb_size, thumb_size)
            )
            canvas.paste(image, (label_w + col * thumb_size, y))
    montage_dir = chunk_dir / "montage"
    montage_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(montage_dir / f"env{env_id:03d}.jpg", quality=95)


def _export_middle_results(
    *,
    env: Any,
    recorder: WanMiddleResultRecorder,
    sample: dict[str, Any],
    sample_path: Path,
    output_root: Path,
    save_latents: bool,
    generated_only: bool,
    max_envs: int,
) -> dict[str, Any]:
    if not recorder.records:
        raise RuntimeError("No Wan middle-result latents were recorded.")
    if max_envs <= 0:
        raise ValueError("--middle-result-max-envs must be positive.")

    sample_index = int(sample["metadata"].get("sample_index", 0))
    chunk_dir = reset_export_dir(
        Path(output_root).expanduser().resolve() / f"chunk_{sample_index:06d}"
    )
    latents_dir = chunk_dir / "latents"
    max_envs_seen = 0
    frame_indices: list[int] = []
    step_indices: list[int] = []
    record_summaries = []
    context_frames = int(getattr(env, "condition_frame_length", 0))

    env.pipe.load_models_to_device(["vae"])
    try:
        for record in recorder.records:
            step_id = int(record["step_index"])
            step_indices.append(step_id)
            latents = record["latents"]
            env_count = min(int(latents.shape[0]), int(max_envs))
            max_envs_seen = max(max_envs_seen, env_count)
            latents = latents[:env_count]
            if save_latents:
                latents_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    latents.detach().cpu().to(torch.float32),
                    latents_dir / f"step_{step_id:06d}.pt",
                )

            videos = _decode_middle_latents(env, latents)
            total_frames = int(videos.shape[2])
            frame_start = min(context_frames, total_frames) if generated_only else 0
            current_frame_indices = list(range(frame_start, total_frames))
            if not frame_indices:
                frame_indices = current_frame_indices

            step_dir = chunk_dir / f"step_{step_id:06d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            for env_id in range(env_count):
                for frame_id in current_frame_indices:
                    save_image(
                        videos[env_id, :, frame_id],
                        step_dir / f"env{env_id:03d}_frame{frame_id:03d}.png",
                    )
            record_summaries.append(
                {
                    "step_index": step_id,
                    "timestep": record["timestep"],
                    "latent_shape": list(record["latents"].shape),
                    "decoded_shape": list(videos.shape),
                    "saved_envs": env_count,
                    "saved_frame_indices": current_frame_indices,
                }
            )
    finally:
        env.pipe.load_models_to_device([])

    unique_step_indices = sorted(set(step_indices))
    for env_id in range(max_envs_seen):
        _make_middle_result_montage(
            chunk_dir=chunk_dir,
            env_id=env_id,
            frame_indices=frame_indices,
            step_indices=unique_step_indices,
        )

    metadata = {
        "sample_path": str(sample_path),
        "sample_metadata": sample["metadata"],
        "chunk_dir": str(chunk_dir),
        "save_latents": bool(save_latents),
        "generated_only": bool(generated_only),
        "context_frames": context_frames,
        "records": record_summaries,
    }
    save_json(metadata, chunk_dir / "metadata.json")
    return metadata


def _profile_prefix_step_quality(
    *,
    args: argparse.Namespace,
    cfg,
    sample: dict[str, Any],
    sample_input: dict[str, Any],
    reference: dict[str, Any],
    device: torch.device,
    output_dir: Path,
    local_wan_src: Path | None,
) -> dict[str, Any]:
    if args.prefix_steps < 0:
        raise ValueError("--prefix-steps must be non-negative.")

    env, num_envs = _create_wan_env(cfg, sample_input, device)
    if args.prefix_reference_batch_id < 0 or args.prefix_reference_batch_id >= num_envs:
        raise ValueError(
            f"--prefix-reference-batch-id={args.prefix_reference_batch_id} is outside batch size {num_envs}."
        )

    _enable_shared_initial_noise(env)
    warning_messages = _warning_messages_for_env(env)
    if not args.share_initial_noise:
        warning_messages.append(
            "--profile-prefix-step enabled shared initial noise for a fair baseline/prefix comparison."
        )

    _clear_wan_prefix_steps(env)
    _restore_env_state(env, sample_input, device)
    baseline, baseline_elapsed = _run_one_chunk(env, sample_input, device)

    _restore_env_state(env, sample_input, device)
    _set_wan_prefix_steps(
        env,
        prefix_steps=args.prefix_steps,
        reference_batch_id=args.prefix_reference_batch_id,
    )
    try:
        prefix, prefix_elapsed = _run_one_chunk(env, sample_input, device)
    finally:
        _clear_wan_prefix_steps(env)

    baseline_dir = output_dir / "baseline"
    prefix_dir = output_dir / f"prefix_steps_{args.prefix_steps:02d}"
    _export_acwm_chunk(
        sample_input=sample_input,
        predicted=baseline,
        reference=reference,
        output_dir=baseline_dir,
        save_pt_outputs=args.save_pt,
        save_output_current_obs_frames=args.save_output_current_obs_frames,
        save_input_current_obs_frames=True,
    )
    _export_acwm_chunk(
        sample_input=sample_input,
        predicted=prefix,
        reference=reference,
        output_dir=prefix_dir,
        save_pt_outputs=args.save_pt,
        save_output_current_obs_frames=args.save_output_current_obs_frames,
        save_input_current_obs_frames=True,
    )

    baseline_payload = _quality_compare_payload(baseline)
    prefix_payload = _quality_compare_payload(prefix)
    reference_payload = _quality_compare_payload(reference)
    diff = {
        "prefix_vs_baseline": nested_diff_summary(prefix_payload, baseline_payload),
        "prefix_vs_reference": nested_diff_summary(prefix_payload, reference_payload)
        if reference_payload
        else {},
        "baseline_vs_reference": nested_diff_summary(baseline_payload, reference_payload)
        if reference_payload
        else {},
    }

    timing = {
        "profile_prefix_step": True,
        "prefix_steps": int(args.prefix_steps),
        "prefix_reference_batch_id": int(args.prefix_reference_batch_id),
        "batch_size": num_envs,
        "chunk": int(env.chunk),
        "num_inference_steps": int(env.num_inference_steps),
        "shared_initial_noise": True,
        "baseline_elapsed_seconds": float(baseline_elapsed),
        "prefix_elapsed_seconds": float(prefix_elapsed),
        "prefix_speedup_vs_baseline": float(baseline_elapsed / prefix_elapsed)
        if prefix_elapsed > 0
        else None,
        "baseline_dir": str(baseline_dir),
        "prefix_dir": str(prefix_dir),
        "local_wan_src": str(local_wan_src) if local_wan_src else None,
    }
    metadata = {
        "sample_path": str(args.sample_path),
        "sample_metadata": sample["metadata"],
        "config_name": args.config_name,
        "config_dir": str(args.config_dir) if args.config_dir else None,
        "overrides": args.override,
        "device": str(device),
        "local_wan_src": str(local_wan_src) if local_wan_src else None,
        "num_envs": num_envs,
        "warnings": warning_messages,
    }
    summary = _prefix_quality_summary(
        timing=timing,
        prefix=prefix,
        baseline=baseline,
        reference=reference,
    )
    save_json(timing, output_dir / "timing.json")
    save_json(metadata, output_dir / "metadata.json")
    save_json(diff, output_dir / "diff.json")
    _write_prefix_quality_summary(summary, output_dir)
    return {"metadata": metadata, "timing": timing, "diff": diff}


def _profile_middle_result(
    *,
    args: argparse.Namespace,
    cfg,
    sample: dict[str, Any],
    sample_input: dict[str, Any],
    reference: dict[str, Any],
    device: torch.device,
    output_dir: Path,
    local_wan_src: Path | None,
) -> dict[str, Any]:
    env, num_envs = _create_wan_env(cfg, sample_input, device)
    if args.share_initial_noise:
        _enable_shared_initial_noise(env)
    warning_messages = _warning_messages_for_env(env)

    recorder = WanMiddleResultRecorder()
    _set_middle_result_recorder(recorder)
    try:
        predicted, elapsed = _run_one_chunk(env, sample_input, device)
    finally:
        _set_middle_result_recorder(None)

    _export_acwm_chunk(
        sample_input=sample_input,
        predicted=predicted,
        reference=reference,
        output_dir=output_dir,
        save_pt_outputs=args.save_pt,
        save_output_current_obs_frames=args.save_output_current_obs_frames,
        save_input_current_obs_frames=True,
    )
    middle_metadata = _export_middle_results(
        env=env,
        recorder=recorder,
        sample=sample,
        sample_path=args.sample_path,
        output_root=args.middle_result_dir,
        save_latents=args.middle_result_save_latents,
        generated_only=args.middle_result_generated_only,
        max_envs=args.middle_result_max_envs,
    )

    timing = {
        "profile_middle_result": True,
        "elapsed_seconds": float(elapsed),
        "batch_size": num_envs,
        "chunk": int(env.chunk),
        "num_inference_steps": int(env.num_inference_steps),
        "recorded_steps": len(recorder.records),
        "middle_result_dir": middle_metadata["chunk_dir"],
        "middle_result_generated_only": bool(args.middle_result_generated_only),
        "middle_result_save_latents": bool(args.middle_result_save_latents),
        "middle_result_max_envs": int(args.middle_result_max_envs),
        "share_initial_noise": bool(args.share_initial_noise),
        "local_wan_src": str(local_wan_src) if local_wan_src else None,
    }
    metadata = {
        "sample_path": str(args.sample_path),
        "sample_metadata": sample["metadata"],
        "config_name": args.config_name,
        "config_dir": str(args.config_dir) if args.config_dir else None,
        "overrides": args.override,
        "device": str(device),
        "local_wan_src": str(local_wan_src) if local_wan_src else None,
        "num_envs": num_envs,
        "middle_result": middle_metadata,
        "warnings": warning_messages,
    }
    save_json(timing, output_dir / "timing.json")
    save_json(metadata, output_dir / "metadata.json")
    return {"metadata": metadata, "timing": timing, "diff": {}}


def _sequence_sort_key(record: dict[str, Any]) -> tuple[int, int, str]:
    sample_index = int(record.get("sample_index", 0))
    elapsed_steps_before = int(record.get("elapsed_steps_before", sample_index))
    return (elapsed_steps_before, sample_index, str(record["path"]))


def _same_sequence_record(record: dict[str, Any], start_metadata: dict[str, Any]) -> bool:
    keys = (
        "kind",
        "worker_label",
        "source",
        "group_index_in_batch",
        "batch_start",
        "batch_end",
        "batch_size",
        "group_size",
        "worker_rank",
    )
    return all(record.get(key) == start_metadata.get(key) for key in keys)


def _read_sequence_manifest_records(
    sample_path: Path, start_sample: dict[str, Any]
) -> list[dict[str, Any]]:
    sample_path = sample_path.expanduser().resolve()
    sample_dir = sample_path.parent
    sample_root = sample_dir.parent
    manifest_path = sample_root / "manifest.jsonl"
    records: list[dict[str, Any]] = []

    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record_path = (sample_root / record["sample_path"]).resolve()
                records.append({**record, "path": record_path})
    else:
        for record_path in sorted(sample_dir.glob("acwm_*.pt")):
            sample = load_slice_sample(record_path, expected_kind="acwm")
            records.append({**sample["metadata"], "path": record_path.resolve()})

    start_metadata = start_sample["metadata"]
    records = [
        record
        for record in records
        if _same_sequence_record(record, start_metadata)
        and Path(record["path"]).is_file()
    ]
    records.sort(key=_sequence_sort_key)

    start_index = None
    for idx, record in enumerate(records):
        if Path(record["path"]).resolve() == sample_path:
            start_index = idx
            break
    if start_index is None:
        raise ValueError(f"Could not find start sample in sequence records: {sample_path}")
    return records[start_index:]


def _select_sequence_records(
    sample_path: Path,
    start_sample: dict[str, Any],
    *,
    max_chunks: int | None,
) -> list[dict[str, Any]]:
    if max_chunks is not None and max_chunks <= 0:
        raise ValueError("--max-chunks must be positive when provided.")

    records = _read_sequence_manifest_records(sample_path, start_sample)
    if max_chunks is not None:
        records = records[:max_chunks]
    if not records:
        raise ValueError(f"No sequence records found from sample: {sample_path}")

    for prev_record, record in zip(records, records[1:]):
        prev_after = prev_record.get("elapsed_steps_after")
        current_before = record.get("elapsed_steps_before")
        if prev_after is not None and current_before is not None:
            if int(prev_after) != int(current_before):
                raise ValueError(
                    "Non-consecutive ACWM sequence records: "
                    f"{prev_record['path']} elapsed_steps_after={prev_after}, "
                    f"{record['path']} elapsed_steps_before={current_before}."
                )
    return records


def _run_sequence(
    *,
    args: argparse.Namespace,
    cfg,
    start_sample: dict[str, Any],
    device: torch.device,
    output_dir: Path,
    local_wan_src: Path | None,
    dit_residual_root: Path | None,
) -> dict[str, Any]:
    records = _select_sequence_records(
        args.sample_path,
        start_sample,
        max_chunks=args.max_chunks,
    )
    sequence_dir = reset_export_dir(output_dir / f"sequence_from_{Path(records[0]['path']).stem}")

    first_input = start_sample["payload"]["input"]
    env, num_envs = _create_wan_env(cfg, first_input, device)
    if args.share_initial_noise:
        _enable_shared_initial_noise(env)
    warning_messages = _warning_messages_for_env(env)
    export_acwm_input(
        first_input,
        reset_export_dir(sequence_dir / "initial_input"),
        save_current_obs_frames=True,
    )

    chunk_records = []
    timings = []
    for chunk_idx, record in enumerate(records):
        sample = start_sample if chunk_idx == 0 else load_slice_sample(
            record["path"], expected_kind="acwm"
        )
        sample_input = sample["payload"]["input"]
        reference = sample["payload"].get("output", {})
        sample_index = int(sample["metadata"].get("sample_index", chunk_idx))

        state_drift = None
        if args.sequence_mode == "teacher_forced":
            _restore_env_state(env, sample_input, device)
        else:
            state_drift = tensor_diff_summary(env.current_obs, sample_input["current_obs"])

        recorder = None
        if dit_residual_root is not None:
            recorder = DitResidualRecorder(
                dit_residual_root / f"chunk_{sample_index:06d}"
            )
            _set_dit_step_recorder(recorder)
        try:
            predicted, elapsed = _run_one_chunk(env, sample_input, device)
        finally:
            if recorder is not None:
                _set_dit_step_recorder(None)
        timings.append(elapsed)

        chunk_dir = sequence_dir / f"chunk_{sample_index:06d}"
        _export_acwm_chunk(
            sample_input=sample_input,
            predicted=predicted,
            reference=reference,
            output_dir=chunk_dir,
            save_pt_outputs=args.save_pt,
            save_output_current_obs_frames=args.save_output_current_obs_frames,
            save_input_current_obs_frames=False,
        )

        chunk_timing = {
            "elapsed_seconds": elapsed,
            "elapsed_steps_before": sample["metadata"].get("elapsed_steps_before"),
            "elapsed_steps_after": sample["metadata"].get("elapsed_steps_after"),
            "sample_index": sample_index,
        }
        chunk_metadata = {
            "sample_path": str(record["path"]),
            "sample_metadata": sample["metadata"],
            "chunk_dir": str(chunk_dir),
            "sequence_mode": args.sequence_mode,
            "rollout_state_vs_reference_input_current_obs": state_drift,
        }
        save_json(chunk_timing, chunk_dir / "timing.json")
        save_json(chunk_metadata, chunk_dir / "metadata.json")
        chunk_records.append({**chunk_timing, **chunk_metadata})

    timing = {
        "profile": False,
        "sequence": True,
        "sequence_mode": args.sequence_mode,
        "chunks": len(records),
        "elapsed_seconds_total": float(sum(timings)),
        "elapsed_seconds_mean": float(sum(timings) / len(timings)),
        "elapsed_seconds_min": float(min(timings)),
        "elapsed_seconds_max": float(max(timings)),
        "batch_size": num_envs,
        "chunk": int(env.chunk),
        "num_inference_steps": int(env.num_inference_steps),
        "sequence_dir": str(sequence_dir),
        "dit_residual_dir": str(dit_residual_root) if dit_residual_root else None,
        "share_initial_noise": bool(args.share_initial_noise),
    }
    metadata = {
        "sample_path": str(args.sample_path),
        "config_name": args.config_name,
        "config_dir": str(args.config_dir) if args.config_dir else None,
        "overrides": args.override,
        "device": str(device),
        "local_wan_src": str(local_wan_src) if local_wan_src else None,
        "share_initial_noise": bool(args.share_initial_noise),
        "warnings": warning_messages,
        "chunks": chunk_records,
    }
    save_json(timing, sequence_dir / "timing.json")
    save_json(metadata, sequence_dir / "manifest.json")
    return {"metadata": metadata, "timing": timing, "diff": {}}


def _profile_scale_wan_chunk_step(
    *,
    args: argparse.Namespace,
    cfg,
    sample_input: dict[str, Any],
    device: torch.device,
    local_wan_src: Path | None,
) -> dict[str, Any]:
    if args.profile_scale_iters <= 0:
        raise ValueError("--profile-scale-iters must be positive.")
    if args.profile_scale_warmup < 0:
        raise ValueError("--profile-scale-warmup must be non-negative.")

    source_batch_size = _source_batch_size_from_wan_input(sample_input)
    batch_sizes = parse_batch_sizes(args.scale_batch_sizes)
    results = []

    for batch_size in batch_sizes:
        if device.type == "cuda" and args.profile_scale_empty_cache:
            torch.cuda.empty_cache()
        scaled_input = _scaled_wan_sample_input(
            sample_input,
            source_batch_size=source_batch_size,
            target_batch_size=batch_size,
        )
        env = None
        result: dict[str, Any] = {
            "batch_size": batch_size,
            "source_batch_size": source_batch_size,
            "status": "ok",
        }
        try:
            env, num_envs = _create_wan_env(cfg, scaled_input, device)
            if args.share_initial_noise:
                _enable_shared_initial_noise(env)
            policy_output_action = _normalize_actions(scaled_input["policy_output_action"])

            for _ in range(args.profile_scale_warmup):
                _restore_env_state(env, scaled_input, device)
                _chunk_step_once(env, policy_output_action, device)

            reset_cuda_peak_memory(device)
            timings = []
            for _ in range(args.profile_scale_iters):
                _restore_env_state(env, scaled_input, device)
                *_, elapsed = _chunk_step_once(env, policy_output_action, device)
                timings.append(elapsed)
            peak_memory = cuda_memory_snapshot(device)

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
                    "num_envs": int(num_envs),
                    "chunk": int(env.chunk),
                    "num_inference_steps": int(env.num_inference_steps),
                    "group_size": int(getattr(env, "group_size", batch_size)),
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
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if args.profile_scale_stop_on_oom:
                break
            continue
        finally:
            del env
            del scaled_input
            if device.type == "cuda" and args.profile_scale_empty_cache:
                torch.cuda.empty_cache()

        results.append(result)

    return {
        "profile": True,
        "profile_scale": True,
        "profile_target": "wan_chunk_step",
        "batch_sizes": batch_sizes,
        "source_batch_size": source_batch_size,
        "iterations": int(args.profile_scale_iters),
        "warmup_iterations": int(args.profile_scale_warmup),
        "stop_on_oom": bool(args.profile_scale_stop_on_oom),
        "empty_cache": bool(args.profile_scale_empty_cache),
        "share_initial_noise": bool(args.share_initial_noise),
        "nvtx_enabled": os.environ.get("RLINF_USE_NVTX", "0"),
        "local_wan_src": str(local_wan_src) if local_wan_src else None,
        "results": results,
    }


def _profile_scale_wan_modules(
    *,
    args: argparse.Namespace,
    cfg,
    sample_input: dict[str, Any],
    device: torch.device,
    local_wan_src: Path | None,
) -> dict[str, Any]:
    if args.profile_scale_iters <= 0:
        raise ValueError("--profile-scale-iters must be positive.")
    if args.profile_scale_warmup < 0:
        raise ValueError("--profile-scale-warmup must be non-negative.")

    source_batch_size = _source_batch_size_from_wan_input(sample_input)
    batch_sizes = parse_batch_sizes(args.scale_batch_sizes)
    results = []

    for batch_size in batch_sizes:
        if device.type == "cuda" and args.profile_scale_empty_cache:
            torch.cuda.empty_cache()
        scaled_input = _scaled_wan_sample_input(
            sample_input,
            source_batch_size=source_batch_size,
            target_batch_size=batch_size,
        )
        env = None
        restore_profiler = None
        result: dict[str, Any] = {
            "batch_size": batch_size,
            "source_batch_size": source_batch_size,
            "status": "ok",
        }
        try:
            env, num_envs = _create_wan_env(cfg, scaled_input, device)
            if args.share_initial_noise:
                _enable_shared_initial_noise(env)
            profiler = WanModuleCallProfiler(device)
            restore_profiler = _patch_wan_module_profiler(env, profiler)
            policy_output_action = _normalize_actions(scaled_input["policy_output_action"])

            for _ in range(args.profile_scale_warmup):
                profiler.clear()
                _restore_env_state(env, scaled_input, device)
                _chunk_step_once(env, policy_output_action, device)

            module_records = {
                "dit": [],
                "vae_decode": [],
            }
            chunk_timings = []
            for _ in range(args.profile_scale_iters):
                profiler.clear()
                _restore_env_state(env, scaled_input, device)
                *_, elapsed = _chunk_step_once(env, policy_output_action, device)
                chunk_timings.append(elapsed)
                for module_name in module_records:
                    module_records[module_name].extend(profiler.records[module_name])

            result.update(
                {
                    "iterations": int(args.profile_scale_iters),
                    "warmup_iterations": int(args.profile_scale_warmup),
                    "chunk_step_elapsed_seconds_mean": float(
                        sum(chunk_timings) / len(chunk_timings)
                    ),
                    "chunk_step_latency_per_sample_seconds": float(
                        sum(chunk_timings) / len(chunk_timings) / batch_size
                    ),
                    "num_envs": int(num_envs),
                    "chunk": int(env.chunk),
                    "num_inference_steps": int(env.num_inference_steps),
                    "group_size": int(getattr(env, "group_size", batch_size)),
                    "modules": {
                        module_name: _summarize_wan_module_records(
                            records,
                            batch_size=batch_size,
                            iterations=int(args.profile_scale_iters),
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
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if args.profile_scale_stop_on_oom:
                break
            continue
        finally:
            if restore_profiler is not None:
                restore_profiler()
            del env
            del scaled_input
            if device.type == "cuda" and args.profile_scale_empty_cache:
                torch.cuda.empty_cache()

        results.append(result)

    return {
        "profile": True,
        "profile_scale": True,
        "profile_scale_modules": True,
        "profile_target": "wan_modules",
        "modules": ["dit", "vae_decode"],
        "batch_sizes": batch_sizes,
        "source_batch_size": source_batch_size,
        "iterations": int(args.profile_scale_iters),
        "warmup_iterations": int(args.profile_scale_warmup),
        "stop_on_oom": bool(args.profile_scale_stop_on_oom),
        "empty_cache": bool(args.profile_scale_empty_cache),
        "share_initial_noise": bool(args.share_initial_noise),
        "nvtx_enabled": os.environ.get("RLINF_USE_NVTX", "0"),
        "local_wan_src": str(local_wan_src) if local_wan_src else None,
        "results": results,
    }


def run_slice(args: argparse.Namespace) -> dict[str, Any]:
    if args.profile and args.sequence:
        raise ValueError("--sequence cannot be combined with --profile.")
    if args.profile and args.dump_dit_residuals:
        raise ValueError("--dump-dit-residuals cannot be combined with --profile.")
    if args.profile and args.profile_scale:
        raise ValueError("--profile and --profile-scale are mutually exclusive.")
    if args.profile_scale_modules and not args.profile_scale:
        raise ValueError("--profile-scale-modules requires --profile-scale.")
    if args.profile_scale and args.sequence:
        raise ValueError("--sequence cannot be combined with --profile-scale.")
    if args.profile_scale and args.dump_dit_residuals:
        raise ValueError("--dump-dit-residuals cannot be combined with --profile-scale.")
    if args.profile_prefix_step and (
        args.profile or args.profile_scale or args.sequence or args.dump_dit_residuals
    ):
        raise ValueError(
            "--profile-prefix-step cannot be combined with --profile, --profile-scale, "
            "--sequence, or --dump-dit-residuals."
        )
    if args.profile_middle_result and (
        args.profile
        or args.profile_scale
        or args.sequence
        or args.dump_dit_residuals
        or args.profile_prefix_step
    ):
        raise ValueError(
            "--profile-middle-result cannot be combined with --profile, --profile-scale, "
            "--sequence, --dump-dit-residuals, or --profile-prefix-step."
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_hydra_config(
        config_name=args.config_name,
        config_dir=args.config_dir,
        overrides=args.override,
    )
    sample = load_slice_sample(args.sample_path, expected_kind="acwm")
    if args.profile or args.profile_scale:
        os.environ.setdefault("RLINF_USE_NVTX", "1")
    elif args.output_dir is None:
        raise ValueError(
            "--output-dir is required unless --profile/--profile-scale is enabled"
        )
    output_dir = reset_export_dir(args.output_dir) if args.output_dir is not None else None
    sample_input = sample["payload"]["input"]
    reference = sample["payload"].get("output", {})

    device = torch.device(args.device)
    _set_worker_device(device)
    local_wan_src = prepend_local_src(args.local_wan_src, package_name="diffsynth")
    dit_residual_root = (
        _reset_dit_residual_output_dir(args.dit_residual_dir)
        if args.dump_dit_residuals
        else None
    )

    if args.profile_scale:
        if args.profile_scale_modules:
            timing = _profile_scale_wan_modules(
                args=args,
                cfg=cfg,
                sample_input=sample_input,
                device=device,
                local_wan_src=local_wan_src,
            )
            return {"metadata": {}, "timing": timing, "diff": {}}
        timing = _profile_scale_wan_chunk_step(
            args=args,
            cfg=cfg,
            sample_input=sample_input,
            device=device,
            local_wan_src=local_wan_src,
        )
        return {"metadata": {}, "timing": timing, "diff": {}}

    if args.profile_prefix_step:
        return _profile_prefix_step_quality(
            args=args,
            cfg=cfg,
            sample=sample,
            sample_input=sample_input,
            reference=reference,
            device=device,
            output_dir=output_dir,
            local_wan_src=local_wan_src,
        )

    if args.profile_middle_result:
        return _profile_middle_result(
            args=args,
            cfg=cfg,
            sample=sample,
            sample_input=sample_input,
            reference=reference,
            device=device,
            output_dir=output_dir,
            local_wan_src=local_wan_src,
        )

    if args.sequence:
        return _run_sequence(
            args=args,
            cfg=cfg,
            start_sample=sample,
            device=device,
            output_dir=output_dir,
            local_wan_src=local_wan_src,
            dit_residual_root=dit_residual_root,
        )

    env, num_envs = _create_wan_env(cfg, sample_input, device)
    if args.share_initial_noise:
        _enable_shared_initial_noise(env)
    policy_output_action = _normalize_actions(sample_input["policy_output_action"])

    warning_messages = _warning_messages_for_env(env)

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
            "local_wan_src": str(local_wan_src) if local_wan_src else None,
            "share_initial_noise": bool(args.share_initial_noise),
        }
        return {"metadata": {"warnings": warning_messages}, "timing": timing, "diff": {}}

    sample_index = int(sample["metadata"].get("sample_index", 0))
    recorder = None
    if dit_residual_root is not None:
        recorder = DitResidualRecorder(dit_residual_root / f"chunk_{sample_index:06d}")
        _set_dit_step_recorder(recorder)
    try:
        predicted, elapsed = _run_one_chunk(env, sample_input, device)
    finally:
        if recorder is not None:
            _set_dit_step_recorder(None)

    metadata = {
        "sample_path": str(args.sample_path),
        "sample_metadata": sample["metadata"],
        "config_name": args.config_name,
        "config_dir": str(args.config_dir) if args.config_dir else None,
        "overrides": args.override,
        "device": str(device),
        "local_wan_src": str(local_wan_src) if local_wan_src else None,
        "num_envs": num_envs,
        "num_inference_steps": int(env.num_inference_steps),
        "elapsed_seconds": elapsed,
        "dit_residual_dir": str(dit_residual_root) if dit_residual_root else None,
        "share_initial_noise": bool(args.share_initial_noise),
        "warnings": warning_messages,
    }

    _export_acwm_chunk(
        sample_input=sample_input,
        predicted=predicted,
        reference=reference,
        output_dir=output_dir,
        save_pt_outputs=args.save_pt,
        save_output_current_obs_frames=args.save_output_current_obs_frames,
        save_input_current_obs_frames=True,
    )

    timing = {
        "profile": False,
        "elapsed_seconds": elapsed,
        "batch_size": num_envs,
        "chunk": int(env.chunk),
        "num_inference_steps": int(env.num_inference_steps),
        "dit_residual_dir": str(dit_residual_root) if dit_residual_root else None,
        "share_initial_noise": bool(args.share_initial_noise),
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
