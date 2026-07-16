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

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any


@dataclass
class GroupRMSActionNormConfig:
    """Scale-free group-level RMS normalization.

    The RMS statistics are computed from the observed group action prefix, so no
    task-specific action scale is required.
    """

    eps: float = 1.0e-6
    clamp: float = 5.0
    include_scale_features: bool = True


@dataclass
class EarlyStopModelConfig:
    """Configuration for the lightweight group all-fail classifier."""

    action_dim: int = 7
    hidden_dim: int = 32
    traj_tcn_layers: int = 2
    group_tcn_layers: int = 2
    traj_kernel_size: int = 5
    group_kernel_size: int = 3
    dropout: float = 0.1
    action_processor_type: str = "group_rms_delta"
    action_norm: GroupRMSActionNormConfig = field(default_factory=GroupRMSActionNormConfig)


@dataclass
class EarlyStopOnlineInferenceConfig:
    """Configuration for online all-fail inference."""

    checkpoint_path: str | None = None
    threshold: float = 0.5
    device: str = "cuda"
    torch_dtype: str = "float32"
    compile_model: bool = False
    strict_load: bool = True
    model: EarlyStopModelConfig | None = None


def _as_plain_dict(cfg: Any) -> dict[str, Any]:
    if cfg is None:
        return {}
    if is_dataclass(cfg):
        return {field.name: getattr(cfg, field.name) for field in fields(cfg)}
    if isinstance(cfg, dict):
        return dict(cfg)
    if hasattr(cfg, "items"):
        return {key: value for key, value in cfg.items()}
    raise TypeError(f"Unsupported config type: {type(cfg)}")


def build_early_stop_config(cfg: Any | None = None) -> EarlyStopModelConfig:
    """Build an EarlyStopModelConfig from a dataclass, dict, or DictConfig."""

    cfg_dict = _as_plain_dict(cfg)
    cfg_dict.pop("model_type", None)
    action_processor_type = cfg_dict.get("action_processor_type", "group_rms_delta")
    if action_processor_type != "group_rms_delta":
        raise ValueError(
            "Unsupported action_processor_type: "
            f"{action_processor_type}. Only group_rms_delta is available."
        )
    if "action_norm" in cfg_dict:
        action_norm_dict = _as_plain_dict(cfg_dict["action_norm"])
        action_norm_cls = GroupRMSActionNormConfig
        action_norm_keys = {field.name for field in fields(action_norm_cls)}
        unknown_action_norm_keys = set(action_norm_dict) - action_norm_keys
        if unknown_action_norm_keys:
            raise ValueError(
                f"Unknown {action_norm_cls.__name__} keys: "
                f"{sorted(unknown_action_norm_keys)}"
            )
        cfg_dict["action_norm"] = action_norm_cls(
            **{key: action_norm_dict[key] for key in action_norm_keys if key in action_norm_dict}
        )
    else:
        cfg_dict["action_norm"] = GroupRMSActionNormConfig()
    model_keys = {field.name for field in fields(EarlyStopModelConfig)}
    unknown_model_keys = set(cfg_dict) - model_keys
    if unknown_model_keys:
        raise ValueError(
            f"Unknown EarlyStopModelConfig keys: {sorted(unknown_model_keys)}"
        )
    return EarlyStopModelConfig(**cfg_dict)


def build_online_inference_config(
    cfg: Any | None = None,
) -> EarlyStopOnlineInferenceConfig:
    """Build online inference config from a dataclass, dict, or DictConfig."""

    cfg_dict = _as_plain_dict(cfg)
    cfg_dict.pop("model_type", None)
    if "model" in cfg_dict and cfg_dict["model"] is not None:
        cfg_dict["model"] = build_early_stop_config(cfg_dict["model"])
    model_keys = {field.name for field in fields(EarlyStopOnlineInferenceConfig)}
    unknown_keys = set(cfg_dict) - model_keys
    if unknown_keys:
        raise ValueError(
            f"Unknown EarlyStopOnlineInferenceConfig keys: {sorted(unknown_keys)}"
        )
    config = EarlyStopOnlineInferenceConfig(**cfg_dict)
    if not 0.0 <= float(config.threshold) <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {config.threshold}.")
    return config
