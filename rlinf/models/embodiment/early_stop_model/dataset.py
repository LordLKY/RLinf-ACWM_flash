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
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class EarlyStopProfileSample:
    action_file: str
    pt_path: Path
    jsonl_path: str
    jsonl_line: int
    fixed_prefix_steps: int | None = None


@dataclass(frozen=True)
class EarlyStopProfileIndexStats:
    jsonl_records: int
    unique_action_files: int
    duplicate_action_files: int
    discarded_duplicate_records: int
    missing_files: int
    indexed_groups: int
    indexed_samples: int


class EarlyStopProfileDataset(Dataset):
    """Offline dataset produced by profile.profile_early_stop.

    Each item is a GRPO group action prefix. Labels are computed from the saved
    ``.pt`` payload, not trusted from jsonl metadata:

    label = 1 means all trajectories in the group failed.
    label = 0 means at least one trajectory succeeded.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        prefix_mode: str = "random",
        prefix_steps: int | None = None,
        min_prefix_steps: int = 1,
        max_prefix_steps: int | None = None,
        prefix_step_list: list[int] | tuple[int, ...] | None = None,
        deduplicate_action_file: bool = True,
        discard_duplicate_action_files: bool = True,
        label_source: str = "pt_success",
        load_metadata: bool = True,
    ):
        self.root = Path(root).expanduser().resolve()
        self.prefix_mode = str(prefix_mode)
        self.prefix_steps = prefix_steps
        self.min_prefix_steps = int(min_prefix_steps)
        self.max_prefix_steps = None if max_prefix_steps is None else int(max_prefix_steps)
        self.prefix_step_list = (
            [int(x) for x in prefix_step_list] if prefix_step_list is not None else None
        )
        self.deduplicate_action_file = bool(deduplicate_action_file)
        self.discard_duplicate_action_files = bool(discard_duplicate_action_files)
        self.label_source = str(label_source)
        self.load_metadata = bool(load_metadata)

        if self.label_source != "pt_success":
            raise ValueError(
                f"Unsupported label_source={self.label_source!r}; only 'pt_success' is supported."
            )
        if self.prefix_mode not in {"fixed", "random", "list"}:
            raise ValueError(
                f"prefix_mode must be one of ['fixed', 'random', 'list'], got {self.prefix_mode!r}."
            )
        if self.prefix_mode == "fixed" and self.prefix_steps is None:
            raise ValueError("prefix_steps is required when prefix_mode='fixed'.")
        if self.prefix_mode == "list" and not self.prefix_step_list:
            raise ValueError("prefix_step_list is required when prefix_mode='list'.")
        if self.min_prefix_steps <= 0:
            raise ValueError(f"min_prefix_steps must be positive, got {self.min_prefix_steps}.")
        if self.max_prefix_steps is not None and self.max_prefix_steps <= 0:
            raise ValueError(f"max_prefix_steps must be positive, got {self.max_prefix_steps}.")
        if (
            self.max_prefix_steps is not None
            and self.min_prefix_steps > self.max_prefix_steps
        ):
            raise ValueError(
                f"min_prefix_steps={self.min_prefix_steps} exceeds "
                f"max_prefix_steps={self.max_prefix_steps}."
            )

        self.samples, self.index_stats = self._build_index()
        if not self.samples:
            raise ValueError(f"No valid early-stop profile samples found under {self.root}.")

    def _read_jsonl_records(self) -> list[dict[str, Any]]:
        merged_jsonl = self.root / "groups.jsonl"
        jsonl_files = [merged_jsonl] if merged_jsonl.is_file() else []
        if not jsonl_files:
            jsonl_files = sorted(self.root.glob("groups_env_rank*.jsonl"))
        if not jsonl_files:
            raise FileNotFoundError(
                f"No groups.jsonl or groups_env_rank*.jsonl files found under {self.root}."
            )

        records: list[dict[str, Any]] = []
        for jsonl_path in jsonl_files:
            with jsonl_path.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if "action_file" not in record:
                        raise ValueError(
                            f"Missing action_file in {jsonl_path}:{line_no}."
                        )
                    record["_jsonl_path"] = jsonl_path.name
                    record["_jsonl_line"] = line_no
                    records.append(record)
        return records

    def _build_index(
        self,
    ) -> tuple[list[EarlyStopProfileSample], EarlyStopProfileIndexStats]:
        records = self._read_jsonl_records()
        action_counts = Counter(str(record["action_file"]) for record in records)
        duplicate_action_files = {
            action_file for action_file, count in action_counts.items() if count > 1
        }

        samples: list[EarlyStopProfileSample] = []
        missing_files = 0
        discarded_duplicate_records = 0
        seen_action_files: set[str] = set()
        for record in records:
            action_file = str(record["action_file"])
            if action_file in duplicate_action_files:
                discarded_duplicate_records += 1
                if self.discard_duplicate_action_files:
                    continue
            if self.deduplicate_action_file and action_file in seen_action_files:
                continue
            seen_action_files.add(action_file)

            pt_path = self.root / action_file
            if not pt_path.is_file():
                missing_files += 1
                continue

            if self.prefix_mode == "list":
                assert self.prefix_step_list is not None
                for fixed_prefix_steps in self.prefix_step_list:
                    samples.append(
                        EarlyStopProfileSample(
                            action_file=action_file,
                            pt_path=pt_path,
                            jsonl_path=str(record["_jsonl_path"]),
                            jsonl_line=int(record["_jsonl_line"]),
                            fixed_prefix_steps=int(fixed_prefix_steps),
                        )
                    )
            else:
                samples.append(
                    EarlyStopProfileSample(
                        action_file=action_file,
                        pt_path=pt_path,
                        jsonl_path=str(record["_jsonl_path"]),
                        jsonl_line=int(record["_jsonl_line"]),
                    )
                )

        indexed_groups = len({sample.action_file for sample in samples})
        stats = EarlyStopProfileIndexStats(
            jsonl_records=len(records),
            unique_action_files=len(action_counts),
            duplicate_action_files=len(duplicate_action_files),
            discarded_duplicate_records=discarded_duplicate_records,
            missing_files=missing_files,
            indexed_groups=indexed_groups,
            indexed_samples=len(samples),
        )
        return samples, stats

    def __len__(self) -> int:
        return len(self.samples)

    def _choose_prefix_steps(self, max_steps: int, sample: EarlyStopProfileSample) -> int:
        if max_steps <= 0:
            raise ValueError(f"actions must contain at least one step, got {max_steps}.")
        if sample.fixed_prefix_steps is not None:
            prefix_steps = sample.fixed_prefix_steps
        elif self.prefix_mode == "fixed":
            assert self.prefix_steps is not None
            prefix_steps = int(self.prefix_steps)
        elif self.prefix_mode == "random":
            upper = max_steps if self.max_prefix_steps is None else min(
                self.max_prefix_steps, max_steps
            )
            lower = min(self.min_prefix_steps, upper)
            prefix_steps = random.randint(lower, upper)
        else:
            raise ValueError(f"Unexpected prefix_mode={self.prefix_mode!r}.")
        return max(1, min(int(prefix_steps), max_steps))

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        payload = torch.load(sample.pt_path, map_location="cpu")
        actions = payload["actions"].float().contiguous()
        if actions.ndim != 3:
            raise ValueError(
                f"{sample.pt_path} actions must have shape [N, T, D], got {tuple(actions.shape)}."
            )
        prefix_steps = self._choose_prefix_steps(actions.shape[1], sample)
        actions = actions[:, :prefix_steps].contiguous()
        valid_mask = torch.ones(actions.shape[:2], dtype=torch.bool)

        success = payload["success"].bool()
        label = torch.tensor(float(success.sum().item() == 0), dtype=torch.float32)
        metadata = {
            "action_file": sample.action_file,
            "jsonl_path": sample.jsonl_path,
            "jsonl_line": sample.jsonl_line,
            "prefix_steps": prefix_steps,
            "label_all_failed": bool(label.item()),
        }
        if self.load_metadata:
            metadata["pt_metadata"] = payload.get("metadata", {})
            metadata["lengths"] = payload.get("lengths", torch.empty(0)).tolist()
            metadata["done_steps"] = payload.get("done_steps", torch.empty(0)).tolist()
            metadata["success"] = success.tolist()
            metadata["failure"] = payload.get("failure", torch.empty(0)).bool().tolist()
            metadata["done"] = payload.get("done", torch.empty(0)).bool().tolist()

        return {
            "actions": actions,
            "valid_mask": valid_mask,
            "label": label,
            "metadata": metadata,
        }

    def count_labels(self) -> dict[str, int]:
        all_failed = 0
        not_all_failed = 0
        for sample in self.samples:
            payload = torch.load(sample.pt_path, map_location="cpu")
            label = bool(payload["success"].bool().sum().item() == 0)
            all_failed += int(label)
            not_all_failed += int(not label)
        return {
            "all_failed": all_failed,
            "not_all_failed": not_all_failed,
            "total": all_failed + not_all_failed,
        }


def early_stop_profile_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    batch_size = len(batch)
    group_size = int(batch[0]["actions"].shape[0])
    action_dim = int(batch[0]["actions"].shape[-1])
    max_steps = max(int(item["actions"].shape[1]) for item in batch)

    actions = torch.zeros((batch_size, group_size, max_steps, action_dim), dtype=torch.float32)
    valid_mask = torch.zeros((batch_size, group_size, max_steps), dtype=torch.bool)
    labels = torch.stack([item["label"].float() for item in batch], dim=0)
    metadata = [item["metadata"] for item in batch]

    for batch_idx, item in enumerate(batch):
        item_actions = item["actions"].float()
        if item_actions.shape[0] != group_size or item_actions.shape[-1] != action_dim:
            raise ValueError(
                "All samples in a batch must share group_size/action_dim; got "
                f"{tuple(item_actions.shape)} vs group_size={group_size}, action_dim={action_dim}."
            )
        steps = int(item_actions.shape[1])
        actions[batch_idx, :, :steps] = item_actions
        item_mask = item["valid_mask"].bool()
        if tuple(item_mask.shape) != (group_size, steps):
            raise ValueError(
                f"valid_mask shape {tuple(item_mask.shape)} does not match "
                f"actions prefix {(group_size, steps)}."
            )
        valid_mask[batch_idx, :, :steps] = item_mask

    return {
        "actions": actions,
        "valid_mask": valid_mask,
        "labels": labels,
        "metadata": metadata,
    }
