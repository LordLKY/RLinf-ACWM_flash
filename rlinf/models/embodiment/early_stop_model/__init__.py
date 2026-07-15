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

from rlinf.models.embodiment.early_stop_model.action_processor import (
    GroupRMSDeltaActionProcessor,
    build_action_processor,
)
from rlinf.models.embodiment.early_stop_model.config import (
    EarlyStopModelConfig,
    GroupRMSActionNormConfig,
    build_early_stop_config,
)
from rlinf.models.embodiment.early_stop_model.dataset import (
    EarlyStopProfileDataset,
    EarlyStopProfileIndexStats,
    EarlyStopProfileSample,
    early_stop_profile_collate,
)
from rlinf.models.embodiment.early_stop_model.temporal_setnet import (
    ConvBlock1D,
    LiteGroupAllFailClassifier,
    TemporalConvEncoder,
    build_early_stop_model,
)


def get_model(cfg, torch_dtype=None):
    model = build_early_stop_model(cfg)
    if torch_dtype is not None:
        model = model.to(dtype=torch_dtype)
    return model


__all__ = [
    "ConvBlock1D",
    "EarlyStopModelConfig",
    "EarlyStopProfileDataset",
    "EarlyStopProfileIndexStats",
    "EarlyStopProfileSample",
    "GroupRMSActionNormConfig",
    "GroupRMSDeltaActionProcessor",
    "LiteGroupAllFailClassifier",
    "TemporalConvEncoder",
    "build_action_processor",
    "build_early_stop_config",
    "build_early_stop_model",
    "early_stop_profile_collate",
    "get_model",
]
