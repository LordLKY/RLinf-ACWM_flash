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

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from rlinf.utils.nested_dict_process import clone_nested_to_cpu
from rlinf.utils.omega_resolver import omegaconf_register
from rlinf.utils.slice_profile import json_safe


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_config_dir() -> Path:
    return repo_root() / "examples" / "embodiment" / "config"


def load_hydra_config(
    *,
    config_name: str,
    config_dir: str | Path | None = None,
    overrides: list[str] | None = None,
) -> DictConfig:
    omegaconf_register()
    cfg_dir = Path(config_dir).expanduser().resolve() if config_dir else default_config_dir()
    os.environ.setdefault("EMBODIED_PATH", str(repo_root() / "examples" / "embodiment"))
    with initialize_config_dir(version_base="1.1", config_dir=str(cfg_dir)):
        return compose(config_name=config_name, overrides=overrides or [])


def load_slice_sample(sample_path: str | Path, expected_kind: str | None = None) -> dict[str, Any]:
    path = Path(sample_path).expanduser().resolve()
    sample = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(sample, dict) or "metadata" not in sample or "payload" not in sample:
        raise ValueError(f"Invalid slice sample format: {path}")
    if expected_kind is not None:
        kind = sample["metadata"].get("kind")
        if kind != expected_kind:
            raise ValueError(f"Expected {expected_kind!r} sample, got {kind!r}: {path}")
    sample["sample_path"] = str(path)
    return sample


def ensure_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def reset_export_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir).expanduser().resolve()
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(data), f, indent=2, ensure_ascii=False)


def save_pt(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(clone_nested_to_cpu(data), path)


def summarize_nested(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "type": "torch.Tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, np.ndarray):
        return {
            "type": "numpy.ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, dict):
        return {str(key): summarize_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        if value and all(isinstance(item, str) for item in value):
            return {"type": "list[str]", "len": len(value)}
        if len(value) <= 16:
            return [summarize_nested(item) for item in value]
        return {
            "type": "list",
            "len": len(value),
            "first": summarize_nested(value[0]),
        }
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [summarize_nested(item) for item in value]}
    return json_safe(value)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float().numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def save_array_text(value: Any, path: str | Path, *, name: str | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = _to_numpy(value)
    with path.open("w", encoding="utf-8") as f:
        if name:
            f.write(f"# {name}\n")
        f.write(f"# shape={arr.shape} dtype={arr.dtype}\n")
        if arr.ndim == 0:
            f.write(f"{arr.item()}\n")
        elif arr.ndim <= 2:
            np.savetxt(f, arr.reshape(arr.shape[0], -1), fmt="%.10g")
        else:
            for idx in range(arr.shape[0]):
                f.write(f"\n# index {idx}\n")
                item = arr[idx]
                item_2d = item.reshape(-1, item.shape[-1]) if item.ndim > 1 else item
                np.savetxt(f, item_2d, fmt="%.10g")


def save_string_list(values: list[str], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for idx, text in enumerate(values):
            f.write(f"[{idx}]\n{text}\n\n")


def _image_array_uint8(value: Any) -> np.ndarray:
    arr = _to_numpy(value)
    arr = np.squeeze(arr)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim != 3 or arr.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Cannot convert array with shape {arr.shape} to image.")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if np.issubdtype(arr.dtype, np.floating):
        finite_arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
        if finite_arr.min() < -0.01:
            finite_arr = (finite_arr + 1.0) * 0.5 * 255.0
        elif finite_arr.max() <= 1.5:
            finite_arr = finite_arr * 255.0
        arr = finite_arr
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def save_image(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_image_array_uint8(value)).save(path)


def save_obs_images(obs: dict[str, Any], output_dir: str | Path, *, prefix: str = "") -> None:
    output_dir = Path(output_dir)
    main_images = obs.get("main_images")
    if isinstance(main_images, torch.Tensor) or isinstance(main_images, np.ndarray):
        arr = main_images.detach().cpu() if isinstance(main_images, torch.Tensor) else main_images
        if arr.ndim == 5:
            arr = arr[:, -1]
        for env_idx in range(int(arr.shape[0])):
            save_image(arr[env_idx], output_dir / f"{prefix}main_env{env_idx:03d}.png")

    wrist_images = obs.get("wrist_images")
    if isinstance(wrist_images, torch.Tensor) or isinstance(wrist_images, np.ndarray):
        arr = wrist_images.detach().cpu() if isinstance(wrist_images, torch.Tensor) else wrist_images
        if arr.ndim == 5:
            arr = arr[:, -1]
        for env_idx in range(int(arr.shape[0])):
            save_image(arr[env_idx], output_dir / f"{prefix}wrist_env{env_idx:03d}.png")


def save_wan_frames(
    current_obs: Any,
    output_dir: str | Path,
    *,
    prefix: str = "frame",
) -> None:
    output_dir = Path(output_dir)
    obs = _to_numpy(current_obs)
    if obs.ndim != 6:
        raise ValueError(f"Expected Wan current_obs [B,C,V,T,H,W], got {obs.shape}")
    batch_size, _, views, time_steps, _, _ = obs.shape
    for env_idx in range(batch_size):
        for view_idx in range(views):
            for time_idx in range(time_steps):
                save_image(
                    obs[env_idx, :, view_idx, time_idx],
                    output_dir
                    / f"env{env_idx:03d}_view{view_idx:02d}_{prefix}{time_idx:03d}.png",
                )


def save_pixel_values_images(
    pixel_values: Any,
    output_dir: str | Path,
    *,
    prefix: str = "pixel",
) -> None:
    output_dir = Path(output_dir)
    pixels = _to_numpy(pixel_values)
    if pixels.ndim == 4 and pixels.shape[1] % 3 == 0:
        batch_size, channels, _, _ = pixels.shape
        num_images = channels // 3
        for env_idx in range(batch_size):
            for image_idx in range(num_images):
                save_image(
                    pixels[env_idx, image_idx * 3 : (image_idx + 1) * 3],
                    output_dir / f"env{env_idx:03d}_{prefix}{image_idx:02d}.png",
                )


def export_vla_input(input_payload: dict[str, Any], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env_obs = input_payload.get("env_obs", {})
    save_obs_images(env_obs, output_dir / "images")


def export_vla_output(output_payload: dict[str, Any], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    value = output_payload.get("actions")
    if isinstance(value, (torch.Tensor, np.ndarray)):
        save_array_text(value, output_dir / "actions.txt", name="actions")


def export_acwm_input(input_payload: dict[str, Any], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if "current_obs" in input_payload:
        save_wan_frames(input_payload["current_obs"], output_dir / "current_obs_frames")
    value = input_payload.get("policy_output_action")
    if isinstance(value, (torch.Tensor, np.ndarray)):
        save_array_text(value, output_dir / "actions.txt", name="policy_output_action")


def export_acwm_output(output_payload: dict[str, Any], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if "current_obs" in output_payload and output_payload["current_obs"] is not None:
        save_wan_frames(output_payload["current_obs"], output_dir / "current_obs_frames")
    extracted_obs = output_payload.get("extracted_obs")
    if isinstance(extracted_obs, dict):
        save_obs_images(extracted_obs, output_dir / "extracted_obs_images")
    value = output_payload.get("actions")
    if isinstance(value, (torch.Tensor, np.ndarray)):
        save_array_text(value, output_dir / "actions.txt", name="actions")


def to_device_nested(value: Any, device: torch.device | str) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device_nested(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [to_device_nested(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(to_device_nested(item, device) for item in value)
    return value


def _as_float_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        if not torch.is_floating_point(value) and not torch.is_complex(value):
            return value.detach().cpu().to(torch.float32)
        return value.detach().cpu().to(torch.float32)
    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
        return torch.from_numpy(value).detach().cpu().to(torch.float32)
    return None


def tensor_diff_summary(pred: Any, ref: Any) -> dict[str, Any]:
    pred_tensor = _as_float_tensor(pred)
    ref_tensor = _as_float_tensor(ref)
    if pred_tensor is None or ref_tensor is None:
        return {
            "comparable": False,
            "pred_type": type(pred).__name__,
            "ref_type": type(ref).__name__,
        }
    if tuple(pred_tensor.shape) != tuple(ref_tensor.shape):
        return {
            "comparable": False,
            "reason": "shape_mismatch",
            "pred_shape": list(pred_tensor.shape),
            "ref_shape": list(ref_tensor.shape),
        }
    diff = pred_tensor - ref_tensor
    abs_diff = diff.abs()
    finite = torch.isfinite(pred_tensor) & torch.isfinite(ref_tensor)
    return {
        "comparable": True,
        "shape": list(pred_tensor.shape),
        "pred_dtype": str(getattr(pred, "dtype", "")),
        "ref_dtype": str(getattr(ref, "dtype", "")),
        "finite_ratio": float(finite.float().mean().item()) if finite.numel() else 1.0,
        "max_abs": float(abs_diff.max().item()) if abs_diff.numel() else 0.0,
        "mean_abs": float(abs_diff.mean().item()) if abs_diff.numel() else 0.0,
        "rmse": float(torch.sqrt((diff * diff).mean()).item()) if diff.numel() else 0.0,
    }


def nested_diff_summary(
    pred: Any,
    ref: Any,
    *,
    prefix: str = "",
    max_items: int = 256,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if len(out) >= max_items:
        return out

    if isinstance(pred, dict) and isinstance(ref, dict):
        for key in sorted(set(pred.keys()) | set(ref.keys())):
            name = f"{prefix}.{key}" if prefix else str(key)
            if key not in pred:
                out[name] = {"comparable": False, "reason": "missing_in_pred"}
            elif key not in ref:
                out[name] = {"comparable": False, "reason": "missing_in_ref"}
            else:
                out.update(
                    nested_diff_summary(
                        pred[key], ref[key], prefix=name, max_items=max_items - len(out)
                    )
                )
            if len(out) >= max_items:
                break
        return out

    if isinstance(pred, (list, tuple)) and isinstance(ref, (list, tuple)):
        if len(pred) != len(ref):
            out[prefix] = {
                "comparable": False,
                "reason": "length_mismatch",
                "pred_len": len(pred),
                "ref_len": len(ref),
            }
            return out
        for idx, (pred_item, ref_item) in enumerate(zip(pred, ref)):
            name = f"{prefix}[{idx}]"
            out.update(
                nested_diff_summary(
                    pred_item, ref_item, prefix=name, max_items=max_items - len(out)
                )
            )
            if len(out) >= max_items:
                break
        return out

    out[prefix or "value"] = tensor_diff_summary(pred, ref)
    return out


def sampling_kwargs(cfg: DictConfig, mode: str) -> dict[str, Any]:
    params = cfg.rollout.get("sampling_params", None)
    if params is None:
        return {}
    params_dict = OmegaConf.to_container(params, resolve=True)
    do_sample = bool(params_dict.get("do_sample", True))
    if mode == "eval":
        temperature = params_dict.get("temperature_eval", params_dict.get("temperature", 0.0))
        do_sample = bool(temperature is not None and temperature > 0)
    else:
        temperature = (
            params_dict.get("temperature_train", params_dict.get("temperature", 1.0))
            if do_sample
            else 1.0
        )
    return {
        "do_sample": do_sample,
        "temperature": temperature,
        "top_k": params_dict.get("top_k", 0),
        "top_p": params_dict.get("top_p", 1.0),
        "max_new_tokens": params_dict.get("max_new_tokens", cfg.rollout.get("max_new_tokens", None)),
    }
