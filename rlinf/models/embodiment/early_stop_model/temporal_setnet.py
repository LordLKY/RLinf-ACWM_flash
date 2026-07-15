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

from typing import Any

import torch
import torch.nn as nn

from rlinf.models.embodiment.early_stop_model.action_processor import (
    build_action_processor,
)
from rlinf.models.embodiment.early_stop_model.config import (
    EarlyStopModelConfig,
    build_early_stop_config,
)


class ConvBlock1D(nn.Module):
    """Residual Conv1D block that preserves [B, C, T] shape."""

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int = 5,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(
                f"ConvBlock1D requires odd kernel_size to preserve length, got {kernel_size}."
            )
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.GELU(),
        )
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class TemporalConvEncoder(nn.Module):
    """Stack of dilated residual Conv1D blocks."""

    def __init__(
        self,
        channels: int,
        *,
        num_layers: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}.")
        self.blocks = nn.Sequential(
            *[
                ConvBlock1D(
                    channels,
                    kernel_size=kernel_size,
                    dilation=2**layer_idx,
                    dropout=dropout,
                )
                for layer_idx in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


def _normalize_valid_mask(
    valid_mask: torch.Tensor | None, *, batch_size: int, group_size: int, steps: int
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
            f"actions shape prefix {(batch_size, group_size, steps)}."
        )
    return valid_mask.bool()


def _masked_set_stats(
    z: torch.Tensor, valid_mask: torch.Tensor | None, eps: float = 1.0e-6
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    # z: [B, N, M, H]
    if valid_mask is None:
        mean = z.mean(dim=1)
        var = ((z - mean[:, None]) ** 2).mean(dim=1)
        return mean, var.add(eps).sqrt(), None

    mask = valid_mask.to(device=z.device).unsqueeze(-1).to(dtype=z.dtype)
    count = mask.sum(dim=1).clamp_min(1.0)
    mean = (z * mask).sum(dim=1) / count
    var = (((z - mean[:, None]) * mask) ** 2).sum(dim=1) / count
    std = var.clamp_min(0.0).add(eps).sqrt()
    time_mask = valid_mask.any(dim=1).to(device=z.device)
    return mean, std, time_mask


def _masked_temporal_pool(
    u: torch.Tensor, time_mask: torch.Tensor | None
) -> torch.Tensor:
    # u: [B, M, H]
    if time_mask is None:
        return torch.cat([u.mean(dim=1), u.max(dim=1).values], dim=-1)

    mask = time_mask.to(device=u.device).unsqueeze(-1).to(dtype=u.dtype)
    count = mask.sum(dim=1).clamp_min(1.0)
    mean = (u * mask).sum(dim=1) / count
    neg_inf = torch.finfo(u.dtype).min
    max_values = u.masked_fill(mask.bool().logical_not(), neg_inf).max(dim=1).values
    has_valid_step = time_mask.to(device=u.device).any(dim=1, keepdim=True)
    max_values = torch.where(has_valid_step, max_values, torch.zeros_like(max_values))
    return torch.cat([mean, max_values], dim=-1)


class LiteGroupAllFailClassifier(nn.Module):
    """Lightweight temporal set classifier for group all-fail prediction.

    Input shape is [B, N, M, D], where N is the number of sampled trajectories
    in a GRPO group and M is the observed prefix length.
    """

    def __init__(
        self,
        cfg: EarlyStopModelConfig | dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        if cfg is not None and kwargs:
            raise ValueError("Pass either cfg or keyword arguments, not both.")
        self.cfg = build_early_stop_config(cfg if cfg is not None else kwargs)
        hidden_dim = int(self.cfg.hidden_dim)

        self.action_processor = build_action_processor(self.cfg)
        self.in_proj = nn.Linear(self.action_processor.output_dim, hidden_dim)
        self.traj_tcn = TemporalConvEncoder(
            hidden_dim,
            num_layers=int(self.cfg.traj_tcn_layers),
            kernel_size=int(self.cfg.traj_kernel_size),
            dropout=float(self.cfg.dropout),
        )
        self.group_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.group_tcn = TemporalConvEncoder(
            hidden_dim,
            num_layers=int(self.cfg.group_tcn_layers),
            kernel_size=int(self.cfg.group_kernel_size),
            dropout=float(self.cfg.dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(self.cfg.dropout)),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, actions: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if actions.ndim != 4:
            raise ValueError(
                f"actions must have shape [B, N, M, D], got {tuple(actions.shape)}."
            )
        batch_size, group_size, steps, _ = actions.shape
        if steps <= 0:
            raise ValueError("actions must contain at least one time step.")
        valid_mask = _normalize_valid_mask(
            valid_mask, batch_size=batch_size, group_size=group_size, steps=steps
        )

        try:
            x = self.action_processor(actions, valid_mask=valid_mask)
        except TypeError:
            x = self.action_processor(actions)
        x = self.in_proj(x)

        x = x.reshape(batch_size * group_size, steps, -1).transpose(1, 2)
        z = self.traj_tcn(x)
        z = z.transpose(1, 2).reshape(batch_size, group_size, steps, -1)

        g_mean, g_std, time_mask = _masked_set_stats(z, valid_mask)
        g = torch.cat([g_mean, g_std], dim=-1)
        g = self.group_proj(g)

        u = g.transpose(1, 2)
        u = self.group_tcn(u)
        u = u.transpose(1, 2)

        feat = _masked_temporal_pool(u, time_mask)
        return self.head(feat)


def build_early_stop_model(
    cfg: EarlyStopModelConfig | dict[str, Any] | None = None,
    **kwargs: Any,
) -> LiteGroupAllFailClassifier:
    return LiteGroupAllFailClassifier(cfg, **kwargs)
