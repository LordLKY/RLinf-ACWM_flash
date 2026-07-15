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
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset

from rlinf.models.embodiment.early_stop_model.dataset import (
    EarlyStopProfileDataset,
    early_stop_profile_collate,
)
from rlinf.models.embodiment.early_stop_model import build_early_stop_model


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _as_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return dict(obj)


def _split_indices(num_samples: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    if num_samples <= 1:
        return list(range(num_samples)), []
    indices = list(range(num_samples))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = int(round(num_samples * float(val_ratio)))
    val_size = max(1, min(val_size, num_samples - 1)) if val_ratio > 0 else 0
    return indices[val_size:], indices[:val_size]


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "actions": batch["actions"].to(device, non_blocking=True),
        "valid_mask": batch["valid_mask"].to(device, non_blocking=True),
        "labels": batch["labels"].to(device, non_blocking=True),
        "metadata": batch["metadata"],
    }


def _binary_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = probs >= 0.5
    labels_bool = labels.bool()
    tp = (preds & labels_bool).sum().item()
    tn = ((~preds) & (~labels_bool)).sum().item()
    fp = (preds & (~labels_bool)).sum().item()
    fn = ((~preds) & labels_bool).sum().item()
    total = max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    accuracy = (tp + tn) / total
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "all_fail_recall": recall,
        "false_positive_rate": fpr,
        "positive_ratio": labels_bool.float().mean().item(),
        "true_positive": float(tp),
        "true_negative": float(tn),
        "false_positive": float(fp),
        "false_negative": float(fn),
    }


def _run_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    pos_weight: torch.Tensor | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_examples = 0
    logits_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []

    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        with torch.set_grad_enabled(training):
            logits = model(batch["actions"], valid_mask=batch["valid_mask"]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(
                logits,
                batch["labels"],
                pos_weight=pos_weight,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = int(batch["labels"].shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        total_examples += batch_size
        logits_list.append(logits.detach().cpu())
        labels_list.append(batch["labels"].detach().cpu())

    if total_examples == 0:
        return {"loss": math.nan}
    logits_all = torch.cat(logits_list, dim=0)
    labels_all = torch.cat(labels_list, dim=0)
    metrics = _binary_metrics(logits_all, labels_all)
    metrics["loss"] = total_loss / total_examples
    metrics["num_examples"] = float(total_examples)
    return metrics


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    cfg: Any,
    dataset_stats: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": OmegaConf.to_container(cfg, resolve=True),
            "dataset_stats": dataset_stats,
        },
        path,
    )


def _create_run_dirs(output_root: Path, run_name: str | None = None) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    base_name = run_name or f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = output_root / base_name
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"{base_name}_{suffix:02d}"
        suffix += 1
    ckpt_dir = run_dir / "ckpt"
    log_dir = run_dir / "logs"
    tensorboard_dir = run_dir / "tensorboard"
    for path in (ckpt_dir, log_dir, tensorboard_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "run": run_dir,
        "ckpt": ckpt_dir,
        "logs": log_dir,
        "tensorboard": tensorboard_dir,
    }


def _log_metrics_to_tensorboard(
    writer: SummaryWriter | None,
    metrics: dict[str, float],
    *,
    split: str,
    epoch: int,
) -> None:
    if writer is None:
        return
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            writer.add_scalar(f"{split}/{key}", float(value), epoch)


@hydra.main(version_base="1.1", config_path=".", config_name="early_stop_model_train")
def main(cfg) -> None:
    _seed_everything(int(cfg.train.seed))
    device_name = str(cfg.train.device)
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    dataset = EarlyStopProfileDataset(
        root=cfg.data.root,
        prefix_mode=str(cfg.data.prefix_mode),
        prefix_steps=cfg.data.get("prefix_steps", None),
        min_prefix_steps=int(cfg.data.min_prefix_steps),
        max_prefix_steps=cfg.data.get("max_prefix_steps", None),
        prefix_step_list=cfg.data.get("prefix_step_list", None),
        deduplicate_action_file=bool(cfg.data.deduplicate_action_file),
        discard_duplicate_action_files=bool(cfg.data.discard_duplicate_action_files),
        label_source=str(cfg.data.label_source),
    )
    label_counts = dataset.count_labels()
    train_indices, val_indices = _split_indices(
        len(dataset), float(cfg.data.val_ratio), int(cfg.train.seed)
    )
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices) if val_indices else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.data.num_workers),
        pin_memory=device.type == "cuda",
        collate_fn=early_stop_profile_collate,
        drop_last=False,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=int(cfg.train.batch_size),
            shuffle=False,
            num_workers=int(cfg.data.num_workers),
            pin_memory=device.type == "cuda",
            collate_fn=early_stop_profile_collate,
            drop_last=False,
        )
        if val_dataset is not None
        else None
    )

    model = build_early_stop_model(cfg.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )

    pos_weight = None
    if bool(cfg.train.use_pos_weight):
        positive = max(label_counts["all_failed"], 1)
        negative = max(label_counts["not_all_failed"], 1)
        pos_weight = torch.tensor(negative / positive, dtype=torch.float32, device=device)

    output_root = Path(str(cfg.train.output_dir)).expanduser().resolve()
    run_dirs = _create_run_dirs(
        output_root,
        run_name=cfg.train.get("run_name", None),
    )
    metrics_path = run_dirs["logs"] / "metrics.jsonl"
    dataset_stats = {
        "index": _as_dict(dataset.index_stats),
        "labels": label_counts,
        "train_samples": len(train_dataset),
        "val_samples": 0 if val_dataset is None else len(val_dataset),
        "pos_weight": None if pos_weight is None else float(pos_weight.item()),
    }
    with (run_dirs["logs"] / "dataset_stats.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_stats, f, indent=2)
    with (run_dirs["logs"] / "config.yaml").open("w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    writer = (
        SummaryWriter(log_dir=str(run_dirs["tensorboard"]))
        if bool(cfg.train.get("tensorboard", True))
        else None
    )

    print(
        json.dumps(
            {
                "run_dir": str(run_dirs["run"]),
                "dataset_stats": dataset_stats,
            },
            indent=2,
        )
    )

    eval_interval = int(cfg.train.get("eval_interval", 1))
    checkpoint_interval = int(cfg.train.get("checkpoint_interval", 1))
    epochs = int(cfg.train.epochs)
    last_record: dict[str, Any] | None = None
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for epoch in range(epochs):
            train_metrics = _run_epoch(
                model=model,
                loader=train_loader,
                device=device,
                optimizer=optimizer,
                pos_weight=pos_weight,
            )
            should_eval = (
                val_loader is not None
                and eval_interval > 0
                and ((epoch + 1) % eval_interval == 0 or epoch == epochs - 1)
            )
            if should_eval:
                val_metrics = _run_epoch(
                    model=model,
                    loader=val_loader,
                    device=device,
                    optimizer=None,
                    pos_weight=pos_weight,
                )
            else:
                val_metrics = {}

            record = {
                "epoch": epoch,
                "run_dir": str(run_dirs["run"]),
                "train": train_metrics,
                "val": val_metrics,
            }
            metrics_file.write(json.dumps(record) + "\n")
            metrics_file.flush()
            print(json.dumps(record))
            last_record = record

            _log_metrics_to_tensorboard(writer, train_metrics, split="train", epoch=epoch)
            if val_metrics:
                _log_metrics_to_tensorboard(writer, val_metrics, split="val", epoch=epoch)
            if writer is not None:
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
                writer.flush()

            should_checkpoint = checkpoint_interval > 0 and (
                (epoch + 1) % checkpoint_interval == 0 or epoch == epochs - 1
            )
            if should_checkpoint:
                ckpt_path = run_dirs["ckpt"] / f"epoch_{epoch + 1:06d}.pt"
                _save_checkpoint(
                    ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=record,
                    cfg=cfg,
                    dataset_stats=dataset_stats,
                )
                _save_checkpoint(
                    run_dirs["ckpt"] / "last.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=record,
                    cfg=cfg,
                    dataset_stats=dataset_stats,
                )

    if writer is not None:
        writer.close()

    with (run_dirs["logs"] / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run_dir": str(run_dirs["run"]),
                "last_epoch": None if last_record is None else last_record["epoch"],
                "last_metrics": last_record,
                "dataset_stats": dataset_stats,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    # Keep Hydra from inheriting unrelated CUDA visibility mutations in tests.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
