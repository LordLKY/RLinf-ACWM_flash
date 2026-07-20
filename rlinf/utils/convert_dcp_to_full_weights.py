# Copyright 2025 The RLinf Authors.
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

"""Convert an RLinf FSDP DCP checkpoint to ``model_state_dict/full_weights.pt``.

Example:
    python -m rlinf.utils.convert_dcp_to_full_weights \
        /path/to/checkpoints/global_step_10/actor/dcp_checkpoint

The input can be either an actor checkpoint directory containing
``dcp_checkpoint/`` or the ``dcp_checkpoint/`` directory itself. By default the
output is written to the sibling ``model_state_dict/full_weights.pt`` path used
by RLinf evaluation configs.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.format_utils import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict


DEFAULT_DCP_MODEL_KEY = "fsdp_checkpoint.model"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an RLinf actor dcp_checkpoint to "
            "actor/model_state_dict/full_weights.pt."
        )
    )
    parser.add_argument(
        "checkpoint_path",
        type=Path,
        help=(
            "Path to an actor checkpoint directory or to its dcp_checkpoint "
            "subdirectory."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Output .pt path. Defaults to "
            "<actor_dir>/model_state_dict/full_weights.pt."
        ),
    )
    parser.add_argument(
        "--dcp-model-key",
        default=DEFAULT_DCP_MODEL_KEY,
        help=(
            "Dotted key for the model state in the DCP checkpoint. "
            f"Default: {DEFAULT_DCP_MODEL_KEY}."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def _resolve_paths(
    checkpoint_path: Path, output_path: Path | None
) -> tuple[Path, Path]:
    checkpoint_path = checkpoint_path.expanduser().resolve()

    if checkpoint_path.name == "dcp_checkpoint":
        dcp_path = checkpoint_path
        actor_dir = checkpoint_path.parent
    else:
        actor_dir = checkpoint_path
        dcp_path = actor_dir / "dcp_checkpoint"

    if not dcp_path.is_dir():
        raise FileNotFoundError(f"DCP checkpoint directory not found: {dcp_path}")

    if not (dcp_path / ".metadata").is_file():
        raise FileNotFoundError(
            f"Invalid DCP checkpoint: missing metadata file {dcp_path / '.metadata'}"
        )

    distcp_files = list(dcp_path.glob("*.distcp"))
    if not distcp_files:
        raise FileNotFoundError(
            f"Invalid DCP checkpoint: no *.distcp files found under {dcp_path}"
        )

    if output_path is None:
        output_path = actor_dir / "model_state_dict" / "full_weights.pt"
    else:
        output_path = output_path.expanduser().resolve()

    return dcp_path, output_path


def _get_nested(data: dict[str, Any], dotted_key: str) -> Any:
    cur: Any = data
    traversed: list[str] = []
    for part in dotted_key.split("."):
        traversed.append(part)
        if not isinstance(cur, dict) or part not in cur:
            available = list(cur.keys()) if isinstance(cur, dict) else type(cur)
            raise KeyError(
                f"Could not find DCP key '{dotted_key}' at "
                f"'{'.'.join(traversed)}'. Available: {available}"
            )
        cur = cur[part]
    return cur


def load_dcp_model_state_dict(
    dcp_path: Path, dcp_model_key: str = DEFAULT_DCP_MODEL_KEY
) -> dict[str, Any]:
    checkpoint: dict[str, Any] = {}
    _load_state_dict(
        checkpoint,
        storage_reader=FileSystemReader(str(dcp_path)),
        planner=_EmptyStateDictLoadPlanner(keys={dcp_model_key}),
        no_dist=True,
    )

    model_state_dict = _get_nested(checkpoint, dcp_model_key)
    if not isinstance(model_state_dict, dict):
        raise TypeError(
            f"DCP key '{dcp_model_key}' did not resolve to a state_dict; "
            f"got {type(model_state_dict)}."
        )
    return model_state_dict


def convert_dcp_to_full_weights(
    checkpoint_path: Path,
    output_path: Path | None = None,
    dcp_model_key: str = DEFAULT_DCP_MODEL_KEY,
    overwrite: bool = False,
) -> Path:
    dcp_path, output_path = _resolve_paths(checkpoint_path, output_path)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass --overwrite to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_name(
        f".{output_path.name}.tmp.{os.getpid()}"
    )
    if tmp_output_path.exists():
        tmp_output_path.unlink()

    model_state_dict = load_dcp_model_state_dict(dcp_path, dcp_model_key)
    torch.save(model_state_dict, tmp_output_path)
    os.replace(tmp_output_path, output_path)
    return output_path


def main() -> None:
    args = _parse_args()
    output_path = convert_dcp_to_full_weights(
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        dcp_model_key=args.dcp_model_key,
        overwrite=args.overwrite,
    )
    print(f"Saved full model weights to {output_path}")


if __name__ == "__main__":
    main()
