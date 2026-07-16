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
from pathlib import Path

import hydra
import torch

from rlinf.models.embodiment.early_stop_model.inference import (
    EarlyStopOnlineInferencer,
)


@hydra.main(version_base="1.1", config_path=".", config_name="early_stop_online_infer")
def main(cfg) -> None:
    inferencer = EarlyStopOnlineInferencer(
        {
            "checkpoint_path": cfg.get("checkpoint_path", None),
            "threshold": cfg.get("threshold", 0.5),
            "device": cfg.get("device", "cuda"),
            "torch_dtype": cfg.get("torch_dtype", "float32"),
            "compile_model": cfg.get("compile_model", False),
            "strict_load": cfg.get("strict_load", True),
            "model": cfg.get("model", None),
        }
    )
    if cfg.get("input_pt", None):
        payload = torch.load(Path(str(cfg.input_pt)).expanduser(), map_location="cpu")
        actions = payload["actions"].float()
        if actions.ndim == 3:
            actions = actions.unsqueeze(0)
        prefix_steps = cfg.get("prefix_steps", None)
        if prefix_steps is not None:
            actions = actions[:, :, : int(prefix_steps)]
        valid_mask = torch.ones(actions.shape[:3], dtype=torch.bool)
    else:
        actions = torch.randn(
            int(cfg.smoke.num_groups),
            int(cfg.smoke.group_size),
            int(cfg.smoke.prefix_steps),
            int(cfg.smoke.action_dim),
            dtype=torch.float32,
        )
        valid_mask = torch.ones(actions.shape[:3], dtype=torch.bool)

    decisions, probabilities = inferencer.predict(actions, valid_mask=valid_mask)
    print(
        json.dumps(
            {
                "checkpoint_path": str(inferencer.checkpoint_path),
                "threshold": inferencer.threshold,
                "actions_shape": list(actions.shape),
                "probabilities": [float(x) for x in probabilities.tolist()],
                "pred_all_fail": [bool(x) for x in decisions.tolist()],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
