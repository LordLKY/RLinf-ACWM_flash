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

from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.early_stop_model.config import (
    EarlyStopModelConfig,
    EarlyStopOnlineInferenceConfig,
    build_early_stop_config,
    build_online_inference_config,
)
from rlinf.models.embodiment.early_stop_model.temporal_setnet import (
    LiteGroupAllFailClassifier,
    build_early_stop_model,
)


def _resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _resolve_dtype(dtype: str) -> torch.dtype:
    dtype = str(dtype).lower()
    if dtype in {"float32", "fp32", "torch.float32"}:
        return torch.float32
    if dtype in {"bfloat16", "bf16", "torch.bfloat16"}:
        return torch.bfloat16
    if dtype in {"float16", "fp16", "torch.float16"}:
        return torch.float16
    raise ValueError(f"Unsupported torch_dtype={dtype!r}.")


def _model_cfg_from_checkpoint(
    checkpoint: dict[str, Any],
    fallback: EarlyStopModelConfig | None,
    checkpoint_path: Path,
) -> EarlyStopModelConfig:
    cfg = checkpoint.get("config", None)
    if isinstance(cfg, dict):
        model_cfg = cfg.get("model", None)
        if model_cfg is not None:
            return build_early_stop_config(model_cfg)
    run_dir = checkpoint_path.parent.parent if checkpoint_path.parent.name == "ckpt" else checkpoint_path.parent
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        file_cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        if isinstance(file_cfg, dict):
            model_cfg = file_cfg.get("model", None)
            if model_cfg is not None:
                return build_early_stop_config(model_cfg)
    if fallback is not None:
        return fallback
    raise ValueError(
        "Checkpoint does not contain config.model and no sibling run config.yaml "
        "was found. Provide early_stop_model.model in the online inference config."
    )


class EarlyStopOnlineInferencer:
    """Online inference wrapper for group all-fail prediction.

    Inputs follow the training convention:

    actions: [G, N, M, D]
        G is number of groups, N is group size, M is observed prefix length.

    valid_mask: optional [G, M] or [G, N, M]
    """

    def __init__(
        self,
        cfg: EarlyStopOnlineInferenceConfig | dict[str, Any] | None = None,
        *,
        checkpoint_path: str | Path | None = None,
        threshold: float | None = None,
        device: str | None = None,
    ):
        cfg = build_online_inference_config(cfg)
        if checkpoint_path is not None:
            cfg.checkpoint_path = str(checkpoint_path)
        if threshold is not None:
            cfg.threshold = float(threshold)
        if device is not None:
            cfg.device = str(device)
        if not cfg.checkpoint_path:
            raise ValueError("checkpoint_path is required for online inference.")

        self.cfg = cfg
        self.threshold = float(cfg.threshold)
        self.device = _resolve_device(str(cfg.device))
        self.dtype = _resolve_dtype(str(cfg.torch_dtype))
        self.checkpoint_path = Path(str(cfg.checkpoint_path)).expanduser().resolve()

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise ValueError(
                f"{self.checkpoint_path} must be a checkpoint dict containing key 'model'."
            )
        model_cfg = _model_cfg_from_checkpoint(
            checkpoint, cfg.model, self.checkpoint_path
        )
        self.model_cfg = model_cfg
        model = build_early_stop_model(model_cfg)
        model.load_state_dict(checkpoint["model"], strict=bool(cfg.strict_load))
        model = model.to(device=self.device)
        if self.dtype != torch.float32:
            model = model.to(dtype=self.dtype)
        model.eval()
        if bool(cfg.compile_model):
            model = torch.compile(model)
        self.model: LiteGroupAllFailClassifier = model

    @torch.no_grad()
    def logits(
        self, actions: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if actions.ndim != 4:
            raise ValueError(
                f"actions must have shape [G, N, M, D], got {tuple(actions.shape)}."
            )
        actions = actions.to(device=self.device, dtype=self.dtype, non_blocking=True)
        if valid_mask is not None:
            valid_mask = valid_mask.to(device=self.device, non_blocking=True).bool()
        logits = self.model(actions, valid_mask=valid_mask).squeeze(-1)
        return logits.float()

    @torch.no_grad()
    def predict_proba(
        self, actions: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return torch.sigmoid(self.logits(actions, valid_mask=valid_mask))

    @torch.no_grad()
    def predict(
        self, actions: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probabilities = self.predict_proba(actions, valid_mask=valid_mask)
        decisions = probabilities >= self.threshold
        return decisions.cpu(), probabilities.cpu()


def build_online_inferencer(
    cfg: EarlyStopOnlineInferenceConfig | dict[str, Any] | None = None,
    **kwargs: Any,
) -> EarlyStopOnlineInferencer:
    if cfg is not None and kwargs:
        cfg_dict = dict(cfg) if isinstance(cfg, dict) else cfg
        if isinstance(cfg_dict, dict):
            cfg = {**cfg_dict, **kwargs}
            return EarlyStopOnlineInferencer(cfg)
        raise ValueError("Pass either cfg or keyword arguments, not both.")
    return EarlyStopOnlineInferencer(cfg if cfg is not None else kwargs)
