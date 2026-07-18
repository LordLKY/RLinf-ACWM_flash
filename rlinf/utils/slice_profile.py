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
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlinf.utils.nested_dict_process import clone_nested_to_cpu


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return str(value)


def slice_nested_batch(value: Any, start: int, end: int, batch_size: int) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim > 0 and value.shape[0] == batch_size:
            return value[start:end].detach().cpu().clone().contiguous()
        return value.detach().cpu().clone().contiguous()
    if isinstance(value, np.ndarray):
        if value.ndim > 0 and value.shape[0] == batch_size:
            return value[start:end].copy()
        return value.copy()
    if isinstance(value, dict):
        return {
            key: slice_nested_batch(item, start, end, batch_size)
            for key, item in value.items()
        }
    if isinstance(value, list):
        if len(value) == batch_size:
            return clone_nested_to_cpu(value[start:end])
        return clone_nested_to_cpu(value)
    if isinstance(value, tuple):
        if len(value) == batch_size:
            return tuple(clone_nested_to_cpu(value[start:end]))
        return tuple(clone_nested_to_cpu(item) for item in value)
    return clone_nested_to_cpu(value)


class SliceProfileWriter:
    """Bounded writer for group-wise inference slice samples."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        kind: str,
        worker_label: str,
        max_groups: int = 8,
    ):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.kind = str(kind)
        self.worker_label = str(worker_label)
        self.max_groups = int(max_groups)

        self.sample_dir = self.output_dir / "samples"
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / "manifest.jsonl"
        self.summary_path = self.output_dir / "summary.json"

        self._num_groups = 0
        self._source_counter: Counter[str] = Counter()

    @property
    def num_groups(self) -> int:
        return self._num_groups

    @property
    def has_capacity(self) -> bool:
        return self.max_groups < 0 or self._num_groups < self.max_groups

    def _write_summary(self) -> None:
        summary = {
            "kind": self.kind,
            "worker_label": self.worker_label,
            "output_dir": str(self.output_dir),
            "num_groups": self._num_groups,
            "max_groups": self.max_groups,
            "source_counter": dict(self._source_counter),
        }
        with self.summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def record_group_slices(
        self,
        payload: dict[str, Any],
        *,
        source: str,
        batch_size: int,
        group_size: int,
        eligible_groups: torch.Tensor | np.ndarray | list[bool] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        if not self.has_capacity:
            return 0
        batch_size = int(batch_size)
        group_size = int(group_size)
        if batch_size <= 0 or group_size <= 0 or batch_size % group_size != 0:
            return 0

        num_groups = batch_size // group_size
        if eligible_groups is None:
            eligible = [True] * num_groups
        elif isinstance(eligible_groups, torch.Tensor):
            eligible = [bool(x) for x in eligible_groups.detach().cpu().flatten().tolist()]
        elif isinstance(eligible_groups, np.ndarray):
            eligible = [bool(x) for x in eligible_groups.reshape(-1).tolist()]
        else:
            eligible = [bool(x) for x in eligible_groups]
        if len(eligible) != num_groups:
            raise ValueError(
                f"eligible_groups length {len(eligible)} does not match "
                f"num_groups={num_groups}."
            )

        written = 0
        for group_index, is_eligible in enumerate(eligible):
            if not is_eligible or not self.has_capacity:
                continue
            start = group_index * group_size
            end = start + group_size
            group_payload = slice_nested_batch(payload, start, end, batch_size)
            sample_metadata = {
                "kind": self.kind,
                "worker_label": self.worker_label,
                "source": str(source),
                "sample_index": self._num_groups,
                "group_index_in_batch": group_index,
                "batch_start": start,
                "batch_end": end,
                "batch_size": batch_size,
                "group_size": group_size,
            }
            if metadata:
                sample_metadata.update(json_safe(metadata))

            sample_name = f"{self.kind}_{self._num_groups:06d}.pt"
            sample_path = self.sample_dir / sample_name
            torch.save(
                {
                    "metadata": sample_metadata,
                    "payload": group_payload,
                },
                sample_path,
            )

            manifest_record = {
                **sample_metadata,
                "sample_path": str(sample_path.relative_to(self.output_dir)),
            }
            with self.manifest_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(manifest_record, ensure_ascii=False) + "\n")

            self._num_groups += 1
            self._source_counter[str(source)] += 1
            written += 1

        if written > 0:
            self._write_summary()
        return written

    def finalize(self) -> dict[str, Any]:
        self._write_summary()
        return {
            "kind": self.kind,
            "worker_label": self.worker_label,
            "output_dir": str(self.output_dir),
            "num_groups": self._num_groups,
            "max_groups": self.max_groups,
            "source_counter": dict(self._source_counter),
        }
