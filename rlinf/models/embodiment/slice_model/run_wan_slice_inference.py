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

from rlinf.models.embodiment.slice_model.common import (
    default_local_src_dir,
    export_acwm_input,
    export_acwm_output,
    load_hydra_config,
    load_slice_sample,
    prepend_local_src,
    repo_root,
    reset_export_dir,
    save_json,
    save_pt,
    tensor_diff_summary,
)
from rlinf.scheduler import Worker
from rlinf.utils.utils import nvtx_range


PROFILE_ITERATIONS = 10
DEFAULT_DIT_RESIDUAL_DIR = repo_root() / "profile" / "wan_slice" / "dit_residual"


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


def _set_dit_step_recorder(recorder: DitResidualRecorder | None) -> None:
    from diffsynth.pipelines.wan_video_new import set_dit_step_recorder

    set_dit_step_recorder(recorder)


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


def run_slice(args: argparse.Namespace) -> dict[str, Any]:
    if args.profile and args.sequence:
        raise ValueError("--sequence cannot be combined with --profile.")
    if args.profile and args.dump_dit_residuals:
        raise ValueError("--dump-dit-residuals cannot be combined with --profile.")

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
