from __future__ import annotations

import argparse
import os
import time
from collections import deque
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
    reset_cuda_peak_memory,
    reset_export_dir,
    save_image,
    save_json,
    save_pt,
    scale_nested_batch,
    tensor_diff_summary,
)
from rlinf.scheduler import Worker
from rlinf.utils.utils import nvtx_range

PROFILE_ITERATIONS = 10
DEFAULT_MIDDLE_RESULT_DIR = repo_root() / "profile" / "opensora_slice" / "middle_result"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay and profile one OpenSora world-model slice sample."
    )
    parser.add_argument("--sample-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--config-name",
        default="opensora_libero_spatial_grpo_openvlaoft_ngpu",
        help="Hydra config name under examples/embodiment/config.",
    )
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional Hydra override. Can be repeated.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare predicted slice outputs with reference outputs saved in the sample.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run replay repeatedly without exporting images.",
    )
    parser.add_argument(
        "--profile-scale",
        action="store_true",
        help="Benchmark end-to-end one-chunk inference at multiple batch sizes.",
    )
    parser.add_argument(
        "--profile-scale-modules",
        action="store_true",
        help="Benchmark DiT and VAE decode modules separately at multiple batch sizes.",
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
    )
    parser.add_argument(
        "--profile-scale-empty-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save-pt",
        action="store_true",
        help="Also save input/predicted/reference tensors as .pt files.",
    )
    parser.add_argument(
        "--save-output-current-obs-frames",
        action="store_true",
        help="Export predicted/reference current_obs frames in addition to extracted observations.",
    )
    parser.add_argument(
        "--share-initial-noise",
        action="store_true",
        help=(
            "Use one shared OpenSora initial diffusion noise sample for all batch lanes. "
            "This only affects slice runs using --local-opensora-src."
        ),
    )
    parser.add_argument(
        "--profile-prefix-step",
        action="store_true",
        help=(
            "Run one OpenSora prefix-step quality experiment: baseline full-batch "
            "denoise and one prefix run whose first --prefix-steps denoise steps "
            "share the reference batch lane's generated latents."
        ),
    )
    parser.add_argument(
        "--prefix-steps",
        type=int,
        default=1,
        help="Number of initial OpenSora denoise steps to run with a shared prefix.",
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
            "Record OpenSora latents after each denoise step and decode them after "
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
        "--local-opensora-src",
        nargs="?",
        const=default_local_src_dir("opensora"),
        default=None,
        type=Path,
        help=(
            "Prepend a local source directory containing the opensora package. "
            "If passed without a value, uses slice_model/local_src/opensora."
        ),
    )
    return parser.parse_args()


def _set_worker_device(device: torch.device) -> None:
    Worker.torch_device_type = device.type


def _set_opensora_step_recorder(recorder: Any | None) -> None:
    from opensora.schedulers.rf import set_opensora_step_recorder

    set_opensora_step_recorder(recorder)


def _set_opensora_shared_initial_noise(
    enabled: bool,
    reference_batch_id: int = 0,
) -> None:
    from opensora.schedulers.rf import set_opensora_shared_initial_noise

    set_opensora_shared_initial_noise(
        enabled=enabled,
        reference_batch_id=reference_batch_id,
    )


def _set_opensora_prefix_steps(
    prefix_steps: int,
    reference_batch_id: int = 0,
) -> None:
    from opensora.schedulers.rf import set_opensora_prefix_steps

    set_opensora_prefix_steps(
        prefix_steps=prefix_steps,
        reference_batch_id=reference_batch_id,
    )


def _clear_opensora_prefix_steps() -> None:
    from opensora.schedulers.rf import clear_opensora_prefix_steps

    clear_opensora_prefix_steps()


def _sample_batch_size(sample_input: dict[str, Any]) -> int:
    if "current_obs" not in sample_input:
        raise KeyError("OpenSora sample input must contain current_obs.")
    return int(sample_input["current_obs"].shape[0])


def _normalize_actions(policy_output_action: Any) -> np.ndarray:
    if isinstance(policy_output_action, torch.Tensor):
        return policy_output_action.detach().cpu().numpy()
    return np.asarray(policy_output_action)


def _build_env_cfg(
    cfg: Any,
    sample_input: dict[str, Any],
    num_envs: int,
) -> Any:
    env_cfg = OmegaConf.create(OmegaConf.to_container(cfg.env.train, resolve=True))
    algorithm_cfg = cfg.get("algorithm", {})

    with open_dict(env_cfg):
        env_cfg.total_num_envs = num_envs
        env_cfg.group_size = int(
            sample_input.get("group_size", algorithm_cfg.get("group_size", num_envs))
        )
        if "reward_coef" in sample_input:
            env_cfg.reward_coef = float(sample_input["reward_coef"])
        elif "reward_coef" in algorithm_cfg:
            env_cfg.reward_coef = float(algorithm_cfg.get("reward_coef"))

        if env_cfg.get("video_cfg", None) is not None:
            env_cfg.video_cfg.save_video = False
        env_cfg.auto_reset = False

        if env_cfg.get("world_model_cfg", None) is not None:
            wm_cfg = env_cfg.world_model_cfg
            if "chunk" in sample_input:
                wm_cfg.chunk = int(sample_input["chunk"])
            if "condition_frame_length" in sample_input:
                wm_cfg.condition_frame_length = int(
                    sample_input["condition_frame_length"]
                )

        env_cfg.profile = OmegaConf.create(
            {
                "profile_rollout": False,
                "profile_early_stop": False,
                "profile_vla_data": False,
                "profile_acwm_data": False,
            }
        )

    return env_cfg


def _restore_env_state(
    env: Any,
    sample_input: dict[str, Any],
    device: torch.device,
) -> None:
    env.current_obs = sample_input["current_obs"].to(device)
    env.image_queue = [
        deque(
            [frame.to(device) for frame in per_env_frames],
            maxlen=env.z_condition_frame_length,
        )
        for per_env_frames in sample_input["image_queue"]
    ]
    env.reset_state_ids = sample_input["reset_state_ids"].to(device)
    env.task_descriptions = list(sample_input["task_descriptions"])
    env.init_ee_poses = list(sample_input["init_ee_poses"])
    env.elapsed_steps = int(sample_input.get("elapsed_steps", 0))
    env.chunk = int(sample_input.get("chunk", env.chunk))
    if "prev_step_reward" in sample_input:
        env.prev_step_reward = sample_input["prev_step_reward"].to(device)
    env._is_start = False


def _image_queue_state(env: Any) -> list[list[torch.Tensor]]:
    return [
        [frame.detach().cpu().clone() for frame in per_env_frames]
        for per_env_frames in env.image_queue
    ]


def _create_env(
    cfg: Any,
    sample_input: dict[str, Any],
    device: torch.device,
) -> tuple[Any, int]:
    from rlinf.envs.world_model.world_model_opensora_env import OpenSoraEnv

    num_envs = _sample_batch_size(sample_input)
    env_cfg = _build_env_cfg(cfg, sample_input, num_envs)
    env = OpenSoraEnv(
        env_cfg,
        num_envs=num_envs,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
        record_metrics=True,
    )
    _restore_env_state(env, sample_input, device)
    return env, num_envs


@torch.no_grad()
def _execute_chunk_step(
    env: Any,
    sample_input: dict[str, Any],
    device: torch.device,
) -> tuple[Any, Any, Any, Any, Any]:
    _restore_env_state(env, sample_input, device)
    policy_output_action = _normalize_actions(sample_input["policy_output_action"])
    with nvtx_range("slice/opensora_chunk_step"):
        return env.chunk_step(policy_output_action)


def _prediction_from_chunk_output(
    env: Any,
    output: tuple[Any, Any, Any, Any, Any],
) -> dict[str, Any]:
    extracted_obs_list, rewards, terminations, truncations, infos_list = output
    past_dones = torch.logical_or(
        terminations.detach().bool().any(dim=1),
        truncations.detach().bool().any(dim=1),
    )
    return {
        "current_obs": env.current_obs.detach().cpu().clone(),
        "image_queue": _image_queue_state(env),
        "extracted_obs": extracted_obs_list[0],
        "chunk_rewards_tensors": rewards.detach().cpu().clone(),
        "chunk_terminations": terminations.detach().cpu().clone(),
        "chunk_truncations": truncations.detach().cpu().clone(),
        "raw_chunk_terminations": terminations.detach().cpu().clone(),
        "raw_chunk_truncations": truncations.detach().cpu().clone(),
        "past_dones": past_dones.detach().cpu().clone(),
        "infos": infos_list[0],
    }


@torch.no_grad()
def _chunk_step_once(
    env: Any,
    sample_input: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    output = _execute_chunk_step(env, sample_input, device)
    return _prediction_from_chunk_output(env, output)


def _quality_compare_payload(payload: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "current_obs",
        "extracted_obs",
        "chunk_rewards_tensors",
        "chunk_terminations",
        "chunk_truncations",
        "raw_chunk_terminations",
        "raw_chunk_truncations",
        "past_dones",
    ]
    return {key: payload[key] for key in keys if key in payload}


def _warning_messages_for_env(
    env: Any,
    sample_input: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    if bool(getattr(env, "use_rel_reward", False)) and "prev_step_reward" not in sample_input:
        warnings.append(
            "OpenSora sample does not contain prev_step_reward; reward comparison "
            "may differ when use_rel_reward=True."
        )
    return warnings


def _export_replay_outputs(
    output_dir: Path,
    sample_input: dict[str, Any],
    predicted: dict[str, Any],
    reference: dict[str, Any],
    *,
    save_pt_outputs: bool,
    save_output_current_obs_frames: bool,
) -> None:
    reset_export_dir(output_dir)
    export_acwm_input(sample_input, output_dir / "input")
    export_acwm_output(
        predicted,
        output_dir / "predicted_output",
        save_current_obs_frames=save_output_current_obs_frames,
    )
    if reference:
        export_acwm_output(
            reference,
            output_dir / "reference_output",
            save_current_obs_frames=save_output_current_obs_frames,
        )

    if save_pt_outputs:
        save_pt(sample_input, output_dir / "input.pt")
        save_pt(predicted, output_dir / "predicted_output.pt")
        if reference:
            save_pt(reference, output_dir / "reference_output.pt")


def _scaled_sample_input(
    sample_input: dict[str, Any],
    batch_size: int,
) -> dict[str, Any]:
    source_batch_size = _sample_batch_size(sample_input)
    scaled = scale_nested_batch(
        sample_input,
        source_batch_size=source_batch_size,
        target_batch_size=batch_size,
    )
    scaled["group_size"] = batch_size
    return scaled


def _empty_cuda_cache_if_needed(device: torch.device, enabled: bool) -> None:
    if enabled and device.type == "cuda":
        torch.cuda.empty_cache()


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_one_chunk(
    env: Any,
    sample_input: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], float]:
    _sync_if_cuda(device)
    start = time.perf_counter()
    predicted = _chunk_step_once(env, sample_input, device)
    _sync_if_cuda(device)
    return predicted, time.perf_counter() - start


def _peak_memory_gb(memory: dict[str, int] | None) -> dict[str, float]:
    if memory is None:
        return {}
    return {
        "peak_memory_allocated_gb": bytes_to_gb(memory["max_allocated_bytes"]),
        "peak_memory_reserved_gb": bytes_to_gb(memory["max_reserved_bytes"]),
    }


def _profile_replay(
    cfg: Any,
    sample_input: dict[str, Any],
    device: torch.device,
    iterations: int,
) -> dict[str, Any]:
    env, num_envs = _create_env(cfg, sample_input, device)
    timings: list[float] = []

    for _ in range(iterations):
        _sync_if_cuda(device)
        start = time.perf_counter()
        _execute_chunk_step(env, sample_input, device)
        _sync_if_cuda(device)
        timings.append(time.perf_counter() - start)

    return {
        "profile": True,
        "profile_target": "opensora_chunk_step",
        "iterations": iterations,
        "batch_size": num_envs,
        "elapsed_seconds_total": float(sum(timings)),
        "elapsed_seconds_mean": float(sum(timings) / max(len(timings), 1)),
        "elapsed_seconds_min": float(min(timings)) if timings else 0.0,
        "elapsed_seconds_max": float(max(timings)) if timings else 0.0,
        "latency_per_sample_seconds": float(
            sum(timings) / max(len(timings), 1) / max(num_envs, 1)
        ),
        "num_sampling_steps": int(getattr(env.scheduler, "num_sampling_steps", 0)),
        "peak_memory": cuda_memory_snapshot(device),
    }


def _profile_scale_one(
    cfg: Any,
    sample_input: dict[str, Any],
    device: torch.device,
    *,
    iterations: int,
    warmup_iterations: int,
) -> dict[str, Any]:
    env, num_envs = _create_env(cfg, sample_input, device)

    for _ in range(warmup_iterations):
        _execute_chunk_step(env, sample_input, device)

    reset_cuda_peak_memory(device)
    timings: list[float] = []
    for _ in range(iterations):
        _sync_if_cuda(device)
        start = time.perf_counter()
        _execute_chunk_step(env, sample_input, device)
        _sync_if_cuda(device)
        timings.append(time.perf_counter() - start)

    memory = cuda_memory_snapshot(device)
    result = {
        "batch_size": num_envs,
        "status": "ok",
        "iterations": iterations,
        "warmup_iterations": warmup_iterations,
        "elapsed_seconds_total": float(sum(timings)),
        "elapsed_seconds_mean": float(sum(timings) / max(len(timings), 1)),
        "elapsed_seconds_min": float(min(timings)) if timings else 0.0,
        "elapsed_seconds_max": float(max(timings)) if timings else 0.0,
        "latency_per_sample_seconds": float(
            sum(timings) / max(len(timings), 1) / max(num_envs, 1)
        ),
        "num_sampling_steps": int(getattr(env.scheduler, "num_sampling_steps", 0)),
    }
    result.update(_peak_memory_gb(memory))
    return result


class OpenSoraModuleProfiler:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.records: dict[str, list[dict[str, Any]]] = {
            "dit": [],
            "vae_decode": [],
        }

    def clear(self) -> None:
        for records in self.records.values():
            records.clear()

    def measure(self, name: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            reset_cuda_peak_memory(self.device)
        start = time.perf_counter()
        with torch.no_grad(), nvtx_range(f"slice/opensora_module/{name}"):
            output = fn(*args, **kwargs)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - start
        record: dict[str, Any] = {"elapsed_seconds": float(elapsed)}
        record.update(_peak_memory_gb(cuda_memory_snapshot(self.device)))
        self.records.setdefault(name, []).append(record)
        return output


def _patch_module_profiler(env: Any, profiler: OpenSoraModuleProfiler) -> Any:
    original_model_forward = env.model.forward
    original_vae_decode = env.vae.decode

    def profiled_model_forward(*args: Any, **kwargs: Any) -> Any:
        return profiler.measure("dit", original_model_forward, *args, **kwargs)

    def profiled_vae_decode(*args: Any, **kwargs: Any) -> Any:
        return profiler.measure("vae_decode", original_vae_decode, *args, **kwargs)

    env.model.forward = profiled_model_forward
    env.vae.decode = profiled_vae_decode

    def restore() -> None:
        env.model.forward = original_model_forward
        env.vae.decode = original_vae_decode

    return restore


def _summarize_module_records(
    records: dict[str, list[dict[str, Any]]],
    batch_size: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, module_records in records.items():
        timings = [float(record["elapsed_seconds"]) for record in module_records]
        total = float(sum(timings))
        calls = len(timings)
        module_summary = {
            "calls": calls,
            "elapsed_seconds_total": total,
            "elapsed_seconds_mean": total / max(calls, 1),
            "elapsed_seconds_min": float(min(timings)) if timings else 0.0,
            "elapsed_seconds_max": float(max(timings)) if timings else 0.0,
            "latency_per_sample_seconds": total / max(calls, 1) / max(batch_size, 1),
        }
        allocated = [
            float(record["peak_memory_allocated_gb"])
            for record in module_records
            if "peak_memory_allocated_gb" in record
        ]
        reserved = [
            float(record["peak_memory_reserved_gb"])
            for record in module_records
            if "peak_memory_reserved_gb" in record
        ]
        if allocated:
            module_summary["peak_memory_allocated_gb"] = round(max(allocated), 3)
        if reserved:
            module_summary["peak_memory_reserved_gb"] = round(max(reserved), 3)
        summary[name] = module_summary
    return summary


def _profile_scale_modules_one(
    cfg: Any,
    sample_input: dict[str, Any],
    device: torch.device,
    *,
    iterations: int,
    warmup_iterations: int,
) -> dict[str, Any]:
    env, num_envs = _create_env(cfg, sample_input, device)
    profiler = OpenSoraModuleProfiler(device)
    restore_profiler = _patch_module_profiler(env, profiler)

    try:
        for _ in range(warmup_iterations):
            _execute_chunk_step(env, sample_input, device)

        profiler.clear()
        for _ in range(iterations):
            _execute_chunk_step(env, sample_input, device)
    finally:
        restore_profiler()

    return {
        "batch_size": num_envs,
        "status": "ok",
        "iterations": iterations,
        "warmup_iterations": warmup_iterations,
        "num_sampling_steps": int(getattr(env.scheduler, "num_sampling_steps", 0)),
        "modules": _summarize_module_records(profiler.records, num_envs),
    }


def _profile_scale(
    cfg: Any,
    sample_input: dict[str, Any],
    device: torch.device,
    *,
    batch_sizes: list[int],
    iterations: int,
    warmup_iterations: int,
    stop_on_oom: bool,
    empty_cache: bool,
    modules: bool,
) -> dict[str, Any]:
    source_batch_size = _sample_batch_size(sample_input)
    results: list[dict[str, Any]] = []

    for batch_size in batch_sizes:
        _empty_cuda_cache_if_needed(device, empty_cache)
        scaled_input = _scaled_sample_input(sample_input, batch_size)
        try:
            if modules:
                result = _profile_scale_modules_one(
                    cfg,
                    scaled_input,
                    device,
                    iterations=iterations,
                    warmup_iterations=warmup_iterations,
                )
            else:
                result = _profile_scale_one(
                    cfg,
                    scaled_input,
                    device,
                    iterations=iterations,
                    warmup_iterations=warmup_iterations,
                )
            result["source_batch_size"] = source_batch_size
            results.append(result)
        except RuntimeError as exc:
            if not is_cuda_oom(exc):
                raise
            memory = cuda_memory_snapshot(device)
            result = {
                "batch_size": batch_size,
                "source_batch_size": source_batch_size,
                "status": "oom",
                "error": str(exc),
            }
            result.update(_peak_memory_gb(memory))
            results.append(result)
            _empty_cuda_cache_if_needed(device, True)
            if stop_on_oom:
                break

    return {
        "profile": True,
        "profile_scale": True,
        "profile_scale_modules": modules,
        "profile_target": (
            "opensora_modules" if modules else "opensora_chunk_step_end_to_end"
        ),
        "batch_sizes": batch_sizes,
        "source_batch_size": source_batch_size,
        "iterations": iterations,
        "warmup_iterations": warmup_iterations,
        "stop_on_oom": stop_on_oom,
        "empty_cache": empty_cache,
        "nvtx_enabled": os.environ.get("RLINF_NVTX", "1"),
        "results": results,
    }


def _timestep_value(timestep: Any) -> float | list[float]:
    if isinstance(timestep, torch.Tensor):
        values = timestep.detach().flatten().float().cpu().tolist()
        return values[0] if len(values) == 1 else values
    if isinstance(timestep, np.ndarray):
        values = np.asarray(timestep).reshape(-1).astype(float).tolist()
        return values[0] if len(values) == 1 else values
    return float(timestep)


class OpenSoraMiddleResultRecorder:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def end_step(
        self,
        *,
        step_index: int,
        timestep: Any,
        latents: torch.Tensor,
        **_: Any,
    ) -> None:
        self.records.append(
            {
                "step_index": int(step_index),
                "timestep": _timestep_value(timestep),
                "latents": latents.detach().to(device="cpu", dtype=torch.float32).clone(),
            }
        )


def _decode_middle_latents(env: Any, latents: torch.Tensor) -> torch.Tensor:
    latents = latents.to(device=env.device, dtype=env.inference_dtype)
    with torch.no_grad(), nvtx_range("slice/opensora_middle_result/vae_decode"):
        if bool(getattr(env, "is_vae_v1_2", False)):
            videos = env.vae.decode(latents, num_frames=int(env.num_frames))
        else:
            try:
                videos = env.vae.decode(latents, num_frames=int(env.num_frames))
            except TypeError:
                videos = env.vae.decode(latents)
    videos = videos.detach().cpu()
    if videos.ndim == 4:
        videos = videos.unsqueeze(0)
    if videos.ndim != 5:
        raise ValueError(
            f"Unexpected decoded OpenSora video tensor shape: {tuple(videos.shape)}"
        )
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
    recorder: OpenSoraMiddleResultRecorder,
    sample: dict[str, Any],
    sample_path: Path,
    output_root: Path,
    save_latents: bool,
    generated_only: bool,
    max_envs: int,
) -> dict[str, Any]:
    if not recorder.records:
        raise RuntimeError("No OpenSora middle-result latents were recorded.")
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


def _as_float_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(torch.float32)
    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
        return torch.from_numpy(value).detach().cpu().to(torch.float32)
    return None


def _image_diff_by_env(pred: Any, ref: Any) -> dict[str, Any]:
    pred_tensor = _as_float_tensor(pred)
    ref_tensor = _as_float_tensor(ref)
    if pred_tensor is None or ref_tensor is None:
        return {"comparable": False, "reason": "non_numeric_images"}
    if tuple(pred_tensor.shape) != tuple(ref_tensor.shape):
        return {
            "comparable": False,
            "reason": "shape_mismatch",
            "pred_shape": list(pred_tensor.shape),
            "ref_shape": list(ref_tensor.shape),
        }
    if pred_tensor.ndim < 1:
        return {"comparable": False, "reason": "missing_batch_dimension"}

    diff = pred_tensor - ref_tensor
    abs_diff = diff.abs().reshape(diff.shape[0], -1)
    rmse = torch.sqrt((diff.reshape(diff.shape[0], -1) ** 2).mean(dim=1))
    per_env = [
        {
            "env_id": int(env_id),
            "mean_abs": float(abs_diff[env_id].mean().item()),
            "rmse": float(rmse[env_id].item()),
            "max_abs": float(abs_diff[env_id].max().item()),
        }
        for env_id in range(int(abs_diff.shape[0]))
    ]
    return {
        "comparable": True,
        "shape": list(pred_tensor.shape),
        "overall": {
            "mean_abs": float(abs_diff.mean().item()),
            "rmse": float(torch.sqrt((diff * diff).mean()).item()),
            "max_abs": float(abs_diff.max().item()),
        },
        "per_env": per_env,
    }


def _main_images_diff_by_env(
    predicted: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    pred_images = predicted.get("extracted_obs", {}).get("main_images")
    ref_images = reference.get("extracted_obs", {}).get("main_images")
    if pred_images is None or ref_images is None:
        return {
            "comparable": False,
            "reason": "missing_extracted_obs_main_images",
        }
    return _image_diff_by_env(pred_images, ref_images)


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
        "OpenSora prefix-step quality summary",
        "",
        f"prefix_steps: {timing['prefix_steps']}",
        f"prefix_reference_batch_id: {timing['prefix_reference_batch_id']}",
        f"batch_size: {timing['batch_size']}",
        f"num_sampling_steps: {timing['num_sampling_steps']}",
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
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _profile_prefix_step_quality(
    *,
    args: argparse.Namespace,
    cfg: Any,
    sample: dict[str, Any],
    sample_input: dict[str, Any],
    reference: dict[str, Any],
    device: torch.device,
    output_dir: Path,
    local_opensora_src: Path | None,
) -> dict[str, Any]:
    if args.prefix_steps < 0:
        raise ValueError("--prefix-steps must be non-negative.")

    output_dir = reset_export_dir(output_dir)
    env, num_envs = _create_env(cfg, sample_input, device)
    if args.prefix_reference_batch_id < 0 or args.prefix_reference_batch_id >= num_envs:
        raise ValueError(
            f"--prefix-reference-batch-id={args.prefix_reference_batch_id} is outside batch size {num_envs}."
        )

    _set_opensora_shared_initial_noise(True)
    warning_messages = _warning_messages_for_env(env, sample_input)
    if not args.share_initial_noise:
        warning_messages.append(
            "--profile-prefix-step enabled shared initial noise for a fair baseline/prefix comparison."
        )

    try:
        _clear_opensora_prefix_steps()
        baseline, baseline_elapsed = _run_one_chunk(env, sample_input, device)

        _set_opensora_prefix_steps(
            prefix_steps=args.prefix_steps,
            reference_batch_id=args.prefix_reference_batch_id,
        )
        try:
            prefix, prefix_elapsed = _run_one_chunk(env, sample_input, device)
        finally:
            _clear_opensora_prefix_steps()
    finally:
        _clear_opensora_prefix_steps()
        _set_opensora_shared_initial_noise(False)

    baseline_dir = output_dir / "baseline"
    prefix_dir = output_dir / f"prefix_steps_{args.prefix_steps:02d}"
    _export_replay_outputs(
        baseline_dir,
        sample_input,
        baseline,
        reference,
        save_pt_outputs=args.save_pt,
        save_output_current_obs_frames=args.save_output_current_obs_frames,
    )
    _export_replay_outputs(
        prefix_dir,
        sample_input,
        prefix,
        reference,
        save_pt_outputs=args.save_pt,
        save_output_current_obs_frames=args.save_output_current_obs_frames,
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
        "num_sampling_steps": int(getattr(env.scheduler, "num_sampling_steps", 0)),
        "shared_initial_noise": True,
        "baseline_elapsed_seconds": float(baseline_elapsed),
        "prefix_elapsed_seconds": float(prefix_elapsed),
        "prefix_speedup_vs_baseline": float(baseline_elapsed / prefix_elapsed)
        if prefix_elapsed > 0
        else None,
        "baseline_dir": str(baseline_dir),
        "prefix_dir": str(prefix_dir),
        "local_opensora_src": str(local_opensora_src) if local_opensora_src else None,
    }
    metadata = {
        "sample_path": str(args.sample_path),
        "sample_metadata": sample["metadata"],
        "config_name": args.config_name,
        "config_dir": str(args.config_dir) if args.config_dir else None,
        "overrides": args.override,
        "device": str(device),
        "local_opensora_src": str(local_opensora_src) if local_opensora_src else None,
        "num_envs": num_envs,
        "warnings": warning_messages,
    }
    summary = {
        "timing": timing,
        "final_image_diff": {
            "prefix_vs_baseline": _main_images_diff_by_env(prefix, baseline),
            "prefix_vs_reference": _main_images_diff_by_env(prefix, reference),
            "baseline_vs_reference": _main_images_diff_by_env(baseline, reference),
        },
    }
    save_json(timing, output_dir / "timing.json")
    save_json(metadata, output_dir / "metadata.json")
    save_json(diff, output_dir / "diff.json")
    _write_prefix_quality_summary(summary, output_dir)
    return {"metadata": metadata, "timing": timing, "diff": diff}


def _profile_middle_result(
    *,
    args: argparse.Namespace,
    cfg: Any,
    sample: dict[str, Any],
    sample_input: dict[str, Any],
    reference: dict[str, Any],
    device: torch.device,
    output_dir: Path,
    local_opensora_src: Path | None,
) -> dict[str, Any]:
    env, num_envs = _create_env(cfg, sample_input, device)
    if args.share_initial_noise:
        _set_opensora_shared_initial_noise(True)

    recorder = OpenSoraMiddleResultRecorder()
    _set_opensora_step_recorder(recorder)
    try:
        predicted, elapsed = _run_one_chunk(env, sample_input, device)
    finally:
        _set_opensora_step_recorder(None)
        _set_opensora_shared_initial_noise(False)

    output_dir = reset_export_dir(output_dir)
    _export_replay_outputs(
        output_dir,
        sample_input,
        predicted,
        reference,
        save_pt_outputs=args.save_pt,
        save_output_current_obs_frames=args.save_output_current_obs_frames,
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
        "num_sampling_steps": int(getattr(env.scheduler, "num_sampling_steps", 0)),
        "recorded_steps": len(recorder.records),
        "middle_result_dir": middle_metadata["chunk_dir"],
        "middle_result_generated_only": bool(args.middle_result_generated_only),
        "middle_result_save_latents": bool(args.middle_result_save_latents),
        "middle_result_max_envs": int(args.middle_result_max_envs),
        "share_initial_noise": bool(args.share_initial_noise),
        "local_opensora_src": str(local_opensora_src) if local_opensora_src else None,
    }
    metadata = {
        "sample_path": str(args.sample_path),
        "sample_metadata": sample["metadata"],
        "config_name": args.config_name,
        "config_dir": str(args.config_dir) if args.config_dir else None,
        "overrides": args.override,
        "device": str(device),
        "local_opensora_src": str(local_opensora_src) if local_opensora_src else None,
        "num_envs": num_envs,
        "middle_result": middle_metadata,
        "warnings": _warning_messages_for_env(env, sample_input),
    }
    save_json(timing, output_dir / "timing.json")
    save_json(metadata, output_dir / "metadata.json")
    return {"metadata": metadata, "timing": timing, "diff": {}}


def run_slice(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    _set_worker_device(device)
    local_opensora_src = prepend_local_src(
        args.local_opensora_src, package_name="opensora"
    )

    cfg = load_hydra_config(
        config_name=args.config_name,
        config_dir=args.config_dir,
        overrides=args.override,
    )
    sample = load_slice_sample(args.sample_path, expected_kind="opensora_wm")
    sample_input = sample["payload"]["input"]
    reference_output = sample["payload"].get("output", {})

    if args.profile_prefix_step and (
        args.profile
        or args.profile_scale
        or args.profile_scale_modules
        or args.profile_middle_result
    ):
        raise ValueError(
            "--profile-prefix-step cannot be combined with --profile, "
            "--profile-scale, --profile-scale-modules, or --profile-middle-result."
        )
    if args.profile_middle_result and (
        args.profile
        or args.profile_scale
        or args.profile_scale_modules
        or args.profile_prefix_step
    ):
        raise ValueError(
            "--profile-middle-result cannot be combined with --profile, "
            "--profile-scale, --profile-scale-modules, or --profile-prefix-step."
        )
    if (args.profile_prefix_step or args.profile_middle_result) and args.output_dir is None:
        raise ValueError(
            "--output-dir is required for --profile-prefix-step and --profile-middle-result."
        )

    if args.profile_prefix_step:
        return _profile_prefix_step_quality(
            args=args,
            cfg=cfg,
            sample=sample,
            sample_input=sample_input,
            reference=reference_output,
            device=device,
            output_dir=args.output_dir,
            local_opensora_src=local_opensora_src,
        )

    if args.profile_middle_result:
        return _profile_middle_result(
            args=args,
            cfg=cfg,
            sample=sample,
            sample_input=sample_input,
            reference=reference_output,
            device=device,
            output_dir=args.output_dir,
            local_opensora_src=local_opensora_src,
        )

    if args.profile_scale or args.profile_scale_modules:
        if args.share_initial_noise:
            _set_opensora_shared_initial_noise(True)
        try:
            result = _profile_scale(
                cfg,
                sample_input,
                device,
                batch_sizes=parse_batch_sizes(args.scale_batch_sizes),
                iterations=args.profile_scale_iters,
                warmup_iterations=args.profile_scale_warmup,
                stop_on_oom=args.profile_scale_stop_on_oom,
                empty_cache=args.profile_scale_empty_cache,
                modules=args.profile_scale_modules,
            )
        finally:
            if args.share_initial_noise:
                _set_opensora_shared_initial_noise(False)
        result["local_opensora_src"] = (
            str(local_opensora_src) if local_opensora_src else None
        )
        result["share_initial_noise"] = bool(args.share_initial_noise)
        return result

    if args.profile:
        if args.share_initial_noise:
            _set_opensora_shared_initial_noise(True)
        try:
            result = _profile_replay(cfg, sample_input, device, PROFILE_ITERATIONS)
        finally:
            if args.share_initial_noise:
                _set_opensora_shared_initial_noise(False)
        result["local_opensora_src"] = (
            str(local_opensora_src) if local_opensora_src else None
        )
        result["share_initial_noise"] = bool(args.share_initial_noise)
        return result

    env, num_envs = _create_env(cfg, sample_input, device)
    if args.share_initial_noise:
        _set_opensora_shared_initial_noise(True)
    try:
        predicted, elapsed = _run_one_chunk(env, sample_input, device)
    finally:
        if args.share_initial_noise:
            _set_opensora_shared_initial_noise(False)

    result: dict[str, Any] = {
        "sample_path": str(args.sample_path),
        "sample_metadata": sample["metadata"],
        "batch_size": num_envs,
        "elapsed_seconds": elapsed,
        "local_opensora_src": str(local_opensora_src) if local_opensora_src else None,
        "share_initial_noise": bool(args.share_initial_noise),
        "warnings": _warning_messages_for_env(env, sample_input),
    }

    if args.output_dir is not None:
        _export_replay_outputs(
            args.output_dir,
            sample_input,
            predicted,
            reference_output,
            save_pt_outputs=args.save_pt,
            save_output_current_obs_frames=args.save_output_current_obs_frames,
        )
        result["output_dir"] = str(args.output_dir)

    if args.compare and reference_output:
        result["diff_summary"] = nested_diff_summary(
            _quality_compare_payload(predicted),
            _quality_compare_payload(reference_output),
        )
        if "current_obs" in reference_output:
            result["current_obs_diff"] = tensor_diff_summary(
                predicted["current_obs"], reference_output["current_obs"]
            )

    if args.output_dir is not None:
        save_json(result, args.output_dir / "summary.json")

    return result


def main() -> None:
    result = run_slice(parse_args())
    print(OmegaConf.to_yaml(OmegaConf.create(result), resolve=True))


if __name__ == "__main__":
    main()
