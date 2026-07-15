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

from typing import Protocol

import torch
import torch.nn as nn

from rlinf.models.embodiment.early_stop_model.config import (
    EarlyStopModelConfig,
    GroupRMSActionNormConfig,
)


class ActionProcessor(Protocol):
    output_dim: int

    def __call__(self, actions: torch.Tensor) -> torch.Tensor: ...


class GroupRMSDeltaActionProcessor(nn.Module):
    """Scale-free RMS action normalization computed within each group.

    This keeps zero-action semantics intact by avoiding mean centering. It also
    appends log-RMS scale features so the classifier can still use magnitude
    information without requiring task-specific scale constants.
    """

    def __init__(self, action_dim: int = 7, cfg: GroupRMSActionNormConfig | None = None):
        super().__init__()
        cfg = cfg or GroupRMSActionNormConfig()
        self.action_dim = int(action_dim)
        self.eps = float(cfg.eps)
        self.clamp = float(cfg.clamp)
        self.include_scale_features = bool(cfg.include_scale_features)
        self.output_dim = (
            4 * self.action_dim
            if self.include_scale_features
            else 2 * self.action_dim
        )
        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}.")
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}.")

    @staticmethod
    def _normalize_mask(
        valid_mask: torch.Tensor | None,
        *,
        batch_size: int,
        group_size: int,
        steps: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if valid_mask is None:
            return None
        if valid_mask.ndim == 2:
            valid_mask = valid_mask[:, None, :].expand(batch_size, group_size, steps)
        if valid_mask.ndim != 3:
            raise ValueError(
                "valid_mask must have shape [B, M] or [B, N, M], "
                f"got {tuple(valid_mask.shape)}."
            )
        if tuple(valid_mask.shape) != (batch_size, group_size, steps):
            raise ValueError(
                f"valid_mask shape {tuple(valid_mask.shape)} does not match "
                f"actions prefix {(batch_size, group_size, steps)}."
            )
        return valid_mask.to(device=device).bool()

    def _group_rms(
        self, value: torch.Tensor, valid_mask: torch.Tensor | None
    ) -> torch.Tensor:
        # value: [B, N, M, D], rms: [B, 1, 1, D]
        if valid_mask is None:
            return value.square().mean(dim=(1, 2), keepdim=True).add(self.eps).sqrt()
        mask = valid_mask.unsqueeze(-1).to(dtype=value.dtype)
        count = mask.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
        mean_square = (value.square() * mask).sum(dim=(1, 2), keepdim=True) / count
        return mean_square.add(self.eps).sqrt()

    def forward(
        self, actions: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if actions.ndim != 4:
            raise ValueError(
                "GroupRMSDeltaActionProcessor expects actions with shape "
                f"[B, N, M, D], got {tuple(actions.shape)}."
            )
        batch_size, group_size, steps, action_dim = actions.shape
        if action_dim != self.action_dim:
            raise ValueError(
                f"GroupRMSDeltaActionProcessor expected D={self.action_dim}, "
                f"got D={action_dim}."
            )
        valid_mask = self._normalize_mask(
            valid_mask,
            batch_size=batch_size,
            group_size=group_size,
            steps=steps,
            device=actions.device,
        )

        delta = torch.zeros_like(actions)
        delta[:, :, 1:] = actions[:, :, 1:] - actions[:, :, :-1]
        if valid_mask is not None:
            delta_valid_mask = valid_mask.clone()
            delta_valid_mask[:, :, 0] = False
            delta_valid_mask[:, :, 1:] &= valid_mask[:, :, :-1]
        else:
            delta_valid_mask = None

        action_rms = self._group_rms(actions, valid_mask)
        delta_rms = self._group_rms(delta, delta_valid_mask)
        actions_norm = actions / action_rms
        delta_norm = delta / delta_rms
        if self.clamp > 0:
            actions_norm = actions_norm.clamp(-self.clamp, self.clamp)
            delta_norm = delta_norm.clamp(-self.clamp, self.clamp)

        features = [actions_norm, delta_norm]
        if self.include_scale_features:
            log_action_rms = action_rms.log().expand_as(actions_norm)
            log_delta_rms = delta_rms.log().expand_as(delta_norm)
            features.extend([log_action_rms, log_delta_rms])
        return torch.cat(features, dim=-1)


def build_action_processor(cfg: EarlyStopModelConfig) -> nn.Module:
    if cfg.action_processor_type == "group_rms_delta":
        if not isinstance(cfg.action_norm, GroupRMSActionNormConfig):
            raise TypeError(
                "group_rms_delta requires GroupRMSActionNormConfig, "
                f"got {type(cfg.action_norm)}."
            )
        return GroupRMSDeltaActionProcessor(cfg.action_dim, cfg.action_norm)
    raise ValueError(f"Unsupported action_processor_type: {cfg.action_processor_type}")
