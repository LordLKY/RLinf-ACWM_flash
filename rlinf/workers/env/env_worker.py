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

import asyncio
import gc
import json
import os
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.algorithms.registry import calculate_adv_and_returns
from rlinf.algorithms.rlt.transition import update_rlt_transitions
from rlinf.data.embodied_io_struct import (
    ChunkStepResult,
    ContinuousBatchingRolloutCollector,
    EmbodiedRolloutResult,
    EnvOutput,
    RolloutResult,
    Trajectory,
    convert_trajectories_to_batch,
)
from rlinf.envs import get_env_cls
from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.utils import get_env_attr
from rlinf.envs.wrappers import RecordVideo
from rlinf.scheduler import Channel, Cluster, CommMapper, Worker
from rlinf.utils.distributed import masked_stats, normalize_from_stats
from rlinf.utils.metric_utils import compute_split_num
from rlinf.utils.nested_dict_process import (
    clone_nested_to_cpu,
    copy_dict_tensor,
    split_dict_to_chunk,
    update_nested_cfg,
)
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.utils.utils import (
    flatten_embodied_batch,
    pack_batch,
    preprocess_embodied_batch,
)
from rlinf.workers.env.history_manager import HistoryManager


@dataclass
class SlotGroupState:
    group_id: int
    reset_state_id: int | None = None
    is_dummy: bool = False
    member_trajectory_ids: list[int] = field(default_factory=list)
    completed_member_count: int = 0
    completed_trajectory_ids: set[int] = field(default_factory=set)
    complete: bool = False


class EnvSlotStateManager:
    """Tracks long-lived slot, trajectory, and GRPO group state.

    This is bookkeeping only. It marks slots/trajectories/groups done, but does
    not reset or reuse any slot.
    """

    def __init__(
        self,
        *,
        num_slots: int,
        group_size: int,
        stage_id: int,
        rank: int,
        total_num_envs: int,
        train_batch_size: int,
        train_num_envs_per_stage: int,
        reset_state_ids: torch.Tensor,
    ):
        if num_slots <= 0:
            raise ValueError(f"num_slots must be positive, got {num_slots}")
        if group_size <= 0:
            raise ValueError(f"group_size must be positive, got {group_size}")

        self.num_slots = int(num_slots)
        self.group_size = int(group_size)
        self.stage_id = int(stage_id)
        self.total_num_envs = int(total_num_envs)
        if self.total_num_envs % self.group_size != 0:
            raise ValueError(
                f"total_num_envs={self.total_num_envs} must be divisible by "
                f"group_size={self.group_size}"
            )

        self.slot_ids = torch.arange(self.num_slots, dtype=torch.long)
        global_slot_offset = (
            int(rank) * int(train_batch_size)
            + int(stage_id) * int(train_num_envs_per_stage)
        )
        self.global_slot_ids = self.slot_ids + global_slot_offset
        if int(self.global_slot_ids[0].item()) % self.group_size != 0:
            raise ValueError(
                "Continuous batching currently expects each EnvWorker stage to "
                "start on a group boundary. "
                f"Got global_slot_offset={int(self.global_slot_ids[0].item())}, "
                f"group_size={self.group_size}."
            )
        if self.num_slots % self.group_size != 0:
            raise ValueError(
                "Continuous batching currently expects each EnvWorker stage to "
                f"own complete groups. Got num_slots={self.num_slots}, "
                f"group_size={self.group_size}."
            )

        self.group_id_stride = self.total_num_envs // self.group_size
        self.local_num_groups = self.num_slots // self.group_size
        self.initial_group_id = int(self.global_slot_ids[0].item()) // self.group_size
        self.next_group_id = self.initial_group_id + self.group_id_stride
        self.next_trajectory_id = self.next_group_id * self.group_size

        self.group_ids = torch.empty(self.num_slots, dtype=torch.long)
        self.group_member_ids = torch.empty(self.num_slots, dtype=torch.long)
        self.trajectory_ids = torch.empty(self.num_slots, dtype=torch.long)
        self.assignment_rounds = torch.zeros(self.num_slots, dtype=torch.long)

        self.active = torch.ones(self.num_slots, dtype=torch.bool)
        self.done = torch.zeros(self.num_slots, dtype=torch.bool)
        self.reset_state_ids = self._normalize_reset_state_ids(reset_state_ids)

        self.groups: dict[int, SlotGroupState] = {}
        self.real_group_ids: set[int] = set()
        self.dummy_group_ids: set[int] = set()
        self.target_real_group_count: int | None = None
        self.created_real_group_count = 0
        self.completed_real_group_ids: set[int] = set()
        self.ready_group_ids: list[int] = []
        self.ready_group_id_set: set[int] = set()
        self.done_slot_ids: set[int] = set()
        self.free_slot_ids: set[int] = set()
        self.slot_to_trajectory_id: dict[int, int] = {}
        self.slot_to_group_id: dict[int, int] = {}
        self.slot_to_group_member_id: dict[int, int] = {}
        self.trajectory_done: dict[int, bool] = {}
        self.trajectory_to_group_id: dict[int, int] = {}
        self.open_group_id: int | None = None
        self._assign_initial_slots()

    def _assign_initial_slots(self) -> None:
        trajectory_id = self.initial_group_id * self.group_size
        for local_group_idx in range(self.local_num_groups):
            group_id = self.initial_group_id + local_group_idx
            group = self.groups.setdefault(
                group_id,
                SlotGroupState(
                    group_id,
                    reset_state_id=int(
                        self.reset_state_ids[local_group_idx * self.group_size].item()
                    ),
                ),
            )
            self.real_group_ids.add(group_id)
            self.created_real_group_count += 1
            for member_id in range(self.group_size):
                slot_idx = local_group_idx * self.group_size + member_id
                self.group_ids[slot_idx] = group_id
                self.group_member_ids[slot_idx] = member_id
                self.trajectory_ids[slot_idx] = trajectory_id

                self.slot_to_trajectory_id[slot_idx] = trajectory_id
                self.slot_to_group_id[slot_idx] = group_id
                self.slot_to_group_member_id[slot_idx] = member_id
                self.trajectory_done[trajectory_id] = False
                self.trajectory_to_group_id[trajectory_id] = group_id
                group.member_trajectory_ids.append(trajectory_id)
                trajectory_id += 1

    def _normalize_reset_state_ids(self, reset_state_ids: torch.Tensor) -> torch.Tensor:
        reset_state_ids = torch.as_tensor(reset_state_ids, dtype=torch.long).cpu()
        if reset_state_ids.numel() != self.num_slots:
            raise ValueError(
                f"Expected {self.num_slots} reset_state_ids, "
                f"got {reset_state_ids.numel()}"
            )
        return reset_state_ids.reshape(self.num_slots)

    @staticmethod
    def _done_mask(dones: torch.Tensor | None, num_slots: int) -> torch.Tensor:
        if dones is None:
            return torch.zeros(num_slots, dtype=torch.bool)
        done_mask = dones.detach().cpu().bool()
        if done_mask.dim() > 1:
            done_mask = done_mask.any(dim=tuple(range(1, done_mask.dim())))
        done_mask = done_mask.reshape(-1)
        if done_mask.numel() != num_slots:
            raise ValueError(
                f"Expected {num_slots} done flags, got {done_mask.numel()}"
            )
        return done_mask

    def update_reset_state_ids(self, reset_state_ids: torch.Tensor) -> None:
        self.reset_state_ids = self._normalize_reset_state_ids(reset_state_ids)

    def mark_done(self, dones: torch.Tensor | None) -> torch.Tensor:
        done_mask = self._done_mask(dones, self.num_slots)
        newly_done = done_mask & self.active
        for slot_idx in torch.nonzero(newly_done, as_tuple=False).flatten().tolist():
            trajectory_id = int(self.trajectory_ids[slot_idx].item())
            group_id = int(self.group_ids[slot_idx].item())
            self.active[slot_idx] = False
            self.done[slot_idx] = True
            self.done_slot_ids.add(slot_idx)
            self.free_slot_ids.add(slot_idx)
            self.trajectory_done[trajectory_id] = True

            group = self.groups[group_id]
            if trajectory_id not in group.completed_trajectory_ids:
                group.completed_trajectory_ids.add(trajectory_id)
                group.completed_member_count += 1
                group.complete = group.completed_member_count >= self.group_size
                if group.complete and group_id not in self.ready_group_id_set:
                    self.ready_group_id_set.add(group_id)
                    self.ready_group_ids.append(group_id)
                if group.complete and not group.is_dummy:
                    self.completed_real_group_ids.add(group_id)
        return newly_done

    def set_target_real_group_count(self, target_group_count: int) -> None:
        if target_group_count < self.local_num_groups:
            raise ValueError(
                "target_group_count must be at least the initially assigned "
                f"local group count ({self.local_num_groups}), got {target_group_count}"
            )
        self.target_real_group_count = int(target_group_count)

    @property
    def completed_real_group_count(self) -> int:
        return len(self.completed_real_group_ids)

    @property
    def target_real_groups_complete(self) -> bool:
        if self.target_real_group_count is None:
            return False
        return self.completed_real_group_count >= self.target_real_group_count

    def _can_create_real_group(self) -> bool:
        return (
            self.target_real_group_count is None
            or self.created_real_group_count < self.target_real_group_count
        )

    def _reset_state_id_for_group(
        self, group_id: int, num_reset_states: int
    ) -> int:
        if num_reset_states <= 0:
            raise ValueError(
                f"num_reset_states must be positive, got {num_reset_states}"
            )
        return (int(group_id) * self.group_size) % int(num_reset_states)

    def _new_group(
        self, num_reset_states: int, *, is_dummy: bool = False
    ) -> SlotGroupState:
        group_id = self.next_group_id
        self.next_group_id += self.group_id_stride
        group = SlotGroupState(
            group_id,
            reset_state_id=self._reset_state_id_for_group(
                group_id, num_reset_states
            ),
            is_dummy=is_dummy,
        )
        self.groups[group_id] = group
        if is_dummy:
            self.dummy_group_ids.add(group_id)
        else:
            self.real_group_ids.add(group_id)
            self.created_real_group_count += 1
        self.open_group_id = group_id
        return group

    def reassign_slots(
        self,
        slot_indices: torch.Tensor | list[int],
        num_reset_states: int,
        *,
        allow_real_groups: bool = True,
    ) -> dict[str, torch.Tensor]:
        slots = torch.as_tensor(slot_indices, dtype=torch.long).flatten().cpu()
        if slots.numel() == 0:
            return {
                "slot_indices": slots,
                "episode_indices": torch.empty(0, dtype=torch.long),
            }
        if (slots < 0).any() or (slots >= self.num_slots).any():
            raise IndexError(
                f"slot_indices out of range for num_slots={self.num_slots}: "
                f"{slots.tolist()}"
            )

        group = (
            self.groups.get(self.open_group_id)
            if self.open_group_id is not None
            else None
        )
        if group is not None and len(group.member_trajectory_ids) >= self.group_size:
            group = None
            self.open_group_id = None

        assigned_reset_state_ids = []
        assigned_group_ids = []
        assigned_member_ids = []
        assigned_trajectory_ids = []
        for slot_idx in slots.tolist():
            if slot_idx not in self.free_slot_ids:
                raise ValueError(
                    f"Slot {slot_idx} is not free. It must be marked done before reassignment."
                )
            if group is None or len(group.member_trajectory_ids) >= self.group_size:
                create_real = allow_real_groups and self._can_create_real_group()
                group = self._new_group(num_reset_states, is_dummy=not create_real)

            member_id = len(group.member_trajectory_ids)
            trajectory_id = self.next_trajectory_id
            self.next_trajectory_id += 1
            reset_state_id = int(group.reset_state_id)

            self.group_ids[slot_idx] = group.group_id
            self.group_member_ids[slot_idx] = member_id
            self.trajectory_ids[slot_idx] = trajectory_id
            self.assignment_rounds[slot_idx] += 1
            self.active[slot_idx] = True
            self.done[slot_idx] = False
            self.reset_state_ids[slot_idx] = reset_state_id

            self.free_slot_ids.discard(slot_idx)
            self.done_slot_ids.discard(slot_idx)
            self.slot_to_trajectory_id[slot_idx] = trajectory_id
            self.slot_to_group_id[slot_idx] = group.group_id
            self.slot_to_group_member_id[slot_idx] = member_id
            self.trajectory_done[trajectory_id] = False
            self.trajectory_to_group_id[trajectory_id] = group.group_id
            group.member_trajectory_ids.append(trajectory_id)

            assigned_reset_state_ids.append(reset_state_id)
            assigned_group_ids.append(group.group_id)
            assigned_member_ids.append(member_id)
            assigned_trajectory_ids.append(trajectory_id)

            if len(group.member_trajectory_ids) >= self.group_size:
                self.open_group_id = None
                group = None

        return {
            "slot_indices": slots,
            "episode_indices": torch.as_tensor(
                assigned_reset_state_ids, dtype=torch.long
            ),
            "group_id": torch.as_tensor(assigned_group_ids, dtype=torch.long),
            "group_member_id": torch.as_tensor(assigned_member_ids, dtype=torch.long),
            "trajectory_id": torch.as_tensor(assigned_trajectory_ids, dtype=torch.long),
        }

    def build_metadata(
        self, reset_state_ids: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        if reset_state_ids is not None:
            self.update_reset_state_ids(reset_state_ids)

        group_completed_member_count = torch.tensor(
            [
                self.groups[int(group_id.item())].completed_member_count
                for group_id in self.group_ids
            ],
            dtype=torch.long,
        )
        group_complete = torch.tensor(
            [
                self.groups[int(group_id.item())].complete
                for group_id in self.group_ids
            ],
            dtype=torch.bool,
        )
        is_dummy = torch.tensor(
            [
                self.groups[int(group_id.item())].is_dummy
                for group_id in self.group_ids
            ],
            dtype=torch.bool,
        )

        return {
            "slot_id": self.slot_ids.reshape(self.num_slots, 1).clone(),
            "trajectory_id": self.trajectory_ids.reshape(self.num_slots, 1).clone(),
            "group_id": self.group_ids.reshape(self.num_slots, 1).clone(),
            "group_member_id": self.group_member_ids.reshape(self.num_slots, 1).clone(),
            "reset_state_id": self.reset_state_ids.reshape(self.num_slots, 1).clone(),
            "rollout_epoch_id": self.assignment_rounds.reshape(self.num_slots, 1).clone(),
            "assignment_round": self.assignment_rounds.reshape(self.num_slots, 1).clone(),
            "is_active": self.active.reshape(self.num_slots, 1).clone(),
            "is_done": self.done.reshape(self.num_slots, 1).clone(),
            "is_dummy": is_dummy.reshape(self.num_slots, 1),
            "group_complete": group_complete.reshape(self.num_slots, 1),
            "group_completed_member_count": group_completed_member_count.reshape(
                self.num_slots, 1
            ),
        }


class EnvWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.train_video_cnt = 0
        self.eval_video_cnt = 0
        self.should_stop = False

        self.env_list = []
        self.eval_env_list = []

        self.last_obs_list = []
        self.last_intervened_info_list = []
        self._prefetched_train_bootstrap: list[EnvOutput] | None = None
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        self.collect_transitions = self.cfg.rollout.get("collect_transitions", False)
        self.collect_prev_infos = self.cfg.rollout.get("collect_prev_infos", True)
        self.stage_num = self.cfg.rollout.pipeline_stage_num
        self.enable_rlt = (
            OmegaConf.select(self.cfg, "algorithm.loss_type", default="") == "rlt_ac"
        )

        self.reward_mode = self.cfg.get("reward", {}).get("reward_mode", "per_step")
        self.history_reward_assign = self.cfg.get("reward", {}).get(
            "history_reward_assign", False
        )
        self.use_reward_model = self.cfg.get("reward", {}).get(
            "use_reward_model", False
        )
        self.use_realworld_reward = self.cfg.get("reward", {}).get(
            "standalone_realworld", False
        )
        self.use_external_reward_model = (
            self.use_reward_model and not self.use_realworld_reward
        )
        self.env_infos_reward_keys = ("success", "episode", "final_info")
        if self.use_external_reward_model:
            self.reward_weight = self.cfg.reward.get("reward_weight", 1.0)
            self.env_reward_weight = self.cfg.reward.get("env_reward_weight", 0.0)

        # Env configurations
        self.use_training_pipeline = self.cfg.runner.get("use_training_pipeline", False)
        self.only_eval = getattr(self.cfg.runner, "only_eval", False)
        self.model_cfg = (
            self.cfg.rollout.model if self.only_eval else self.cfg.actor.model
        )
        train_env_cfg = self.cfg.env.get("train", None)
        eval_env_cfg = self.cfg.env.get("eval", None)
        self.enable_train = not self.only_eval and train_env_cfg is not None
        self.enable_eval = (
            self.cfg.runner.get("val_check_interval", -1) > 0 or self.only_eval
        )
        self.rollout_epoch = (
            train_env_cfg.rollout_epoch if train_env_cfg is not None else 1
        )
        self.eval_rollout_epoch = eval_env_cfg.rollout_epoch if self.enable_eval else 1
        continuous_batching_cfg = (
            train_env_cfg.get("continuous_batching", {})
            if train_env_cfg is not None
            else {}
        )
        self.continuous_batching_enabled = bool(
            continuous_batching_cfg.get("enabled", False)
        )
        self.collect_slot_metadata = self.continuous_batching_enabled
        self.slot_state_managers: list[EnvSlotStateManager] | None = None

        profile_cfg = self.cfg.get("profile", {})
        self.profile_early_stop_enabled = bool(
            profile_cfg.get("profile_early_stop", False)
        )
        self._profile_early_stop_groups: dict[tuple[int, int, int, int], dict] = {}
        self._profile_early_stop_history: list[dict[str, Any]] = []
        self._profile_early_stop_step_id = 0
        self._profile_early_stop_dir = os.path.join(
            str(OmegaConf.select(self.cfg, "runner.logger.log_path", default=".")),
            "profile_early_stop",
        )

        self.train_enable_offload = (
            train_env_cfg.get("enable_offload", False)
            if train_env_cfg is not None
            else False
        )
        self.eval_enable_offload = (
            eval_env_cfg.get("enable_offload", False)
            if eval_env_cfg is not None
            else False
        )
        if self.enable_train:
            self.train_num_envs_per_stage = (
                self.cfg.env.train.total_num_envs // self._world_size // self.stage_num
            )
            self.train_batch_size = self.cfg.env.train.total_num_envs // self.stage_num
        if self.enable_eval:
            self.eval_num_envs_per_stage = (
                self.cfg.env.eval.total_num_envs // self._world_size // self.stage_num
            )
            self.eval_batch_size = self.cfg.env.eval.total_num_envs // self.stage_num
        self.n_train_chunk_steps = 0
        if self.enable_train:
            self.n_train_chunk_steps = (
                self.cfg.env.train.max_steps_per_rollout_epoch
                // self.model_cfg.num_action_chunks
            )
        self.n_eval_chunk_steps = 0
        if self.enable_eval:
            self.n_eval_chunk_steps = (
                self.cfg.env.eval.max_steps_per_rollout_epoch
                // self.model_cfg.num_action_chunks
            )
        self.actor_split_num = (
            1 if not self.enable_train else self.get_actor_split_num()
        )
        if self.use_training_pipeline and self.enable_train:
            self._init_pipeline_params()

        if self.enable_train:
            self.train_prev_done: list[torch.Tensor] = [
                torch.zeros(self.train_num_envs_per_stage, dtype=torch.bool)
                for _ in range(self.stage_num)
            ]
        if self.enable_eval:
            self.eval_prev_done: list[torch.Tensor] = [
                torch.zeros(self.eval_num_envs_per_stage, dtype=torch.bool)
                for _ in range(self.stage_num)
            ]
        self.env_decoupled_mode = self.cfg.runner.get("enable_decoupled_mode", False)

        if self.env_decoupled_mode:
            # Init the batch_router for env decoupled mode
            # The batch_router is a dictionary that maps the tag to the list of batch_index.
            self.batch_router = {}
            assert self._component_placement.get_world_size(
                "env"
            ) >= self._component_placement.get_world_size("rollout"), (
                "the world size of env must be greater than the world size of rollout in env_decoupled_mode"
            )

    def init_worker(self):
        # This is a barrier to ensure all envs' initial setup upon import is done
        # Essential for RealWorld env to ensure initial ROS node setup is done
        self.broadcast(
            True,
            groups=[(self._group_name, list(range(self._world_size)))],
        )

        self.update_env_cfg()

        if self.enable_train:
            train_env_cls = get_env_cls(self.cfg.env.train.env_type, self.cfg.env.train)
            self.env_list = self._setup_env_and_wrappers(
                env_cls=train_env_cls,
                env_cfg=self.cfg.env.train,
                num_envs_per_stage=self.train_num_envs_per_stage,
            )
            if self.train_enable_offload:
                assert all(
                    callable(get_env_attr(env, "offload")) for env in self.env_list
                ), "train envs must have an offload method to enable offload!"

        if self.enable_eval:
            eval_env_cls = get_env_cls(self.cfg.env.eval.env_type, self.cfg.env.eval)
            self.eval_env_list = self._setup_env_and_wrappers(
                env_cls=eval_env_cls,
                env_cfg=self.cfg.env.eval,
                num_envs_per_stage=self.eval_num_envs_per_stage,
            )
            if self.eval_enable_offload:
                assert all(
                    callable(get_env_attr(env, "offload")) for env in self.eval_env_list
                ), "eval envs must have an offload method to enable offload!"

        if self.enable_train:
            if self.reward_mode == "history_buffer":
                self.train_history_managers = [
                    HistoryManager(self.cfg.reward, self.train_num_envs_per_stage)
                    for _ in range(self.stage_num)
                ]
                self.history_lengths = [{} for _ in range(self.stage_num)]

        self._init_env()

    def update_env_cfg(self):
        if self.enable_train:
            # train env
            train_override_cfgs = self.cfg.env.train.get("override_cfgs", None)
            if train_override_cfgs is not None:
                assert len(train_override_cfgs) > self._rank, (
                    f"{len(train_override_cfgs)=} > {self._rank=}"
                )

                general_train_override_cfg = OmegaConf.to_container(
                    self.cfg.env.train.get("override_cfg", {}), resolve=True
                )
                override_cfg = OmegaConf.to_container(
                    train_override_cfgs[self._rank], resolve=True
                ).copy()

                base_cfg = {}
                base_cfg = update_nested_cfg(base_cfg, general_train_override_cfg)
                base_cfg = update_nested_cfg(base_cfg, override_cfg)
                setattr(self.cfg.env.train, "override_cfg", OmegaConf.create(base_cfg))
            self._inject_realworld_reward_cfg(self.cfg.env.train)
        if self.enable_eval:
            eval_override_cfgs = self.cfg.env.eval.get("override_cfgs", None)
            if eval_override_cfgs is not None:
                assert len(eval_override_cfgs) > self._rank, (
                    f"{len(eval_override_cfgs)=} > {self._rank=}"
                )

                general_eval_override_cfg = OmegaConf.to_container(
                    self.cfg.env.eval.get("override_cfg", {}), resolve=True
                )
                eval_override_cfg = OmegaConf.to_container(
                    eval_override_cfgs[self._rank], resolve=True
                ).copy()
                base_eval_cfg = {}
                base_eval_cfg = update_nested_cfg(
                    base_eval_cfg, general_eval_override_cfg
                )
                base_eval_cfg = update_nested_cfg(base_eval_cfg, eval_override_cfg)
                setattr(
                    self.cfg.env.eval, "override_cfg", OmegaConf.create(base_eval_cfg)
                )
            self._inject_realworld_reward_cfg(self.cfg.env.eval)

    def _init_pipeline_params(self):
        actor_ws = self._component_placement.get_world_size("actor")
        logical_env_ws = self._world_size * self.stage_num
        self.shuffle_rollout = self.cfg.algorithm.get("shuffle_rollout", True)
        self.pipeline_stage_actor_splits = [
            CommMapper.get_dst_ranks(
                batch_size=self.cfg.env.train.total_num_envs,
                src_world_size=logical_env_ws,
                dst_world_size=actor_ws,
                src_rank=self._rank * self.stage_num + stage_id,
            )
            for stage_id in range(self.stage_num)
        ]
        local_actor_ranks = {
            actor_rank
            for actor_splits in self.pipeline_stage_actor_splits
            for actor_rank, _ in actor_splits
        }
        self.pipeline_actor_env_ranks = {
            actor_rank: sorted(
                {
                    logical_src_rank // self.stage_num
                    for logical_src_rank, _ in CommMapper.get_src_ranks(
                        batch_size=self.cfg.env.train.total_num_envs,
                        src_world_size=logical_env_ws,
                        dst_world_size=actor_ws,
                        dst_rank=actor_rank,
                    )
                }
            )
            for actor_rank in range(actor_ws)
        }
        self.pipeline_actor_keys = {
            actor_rank: CommMapper.build_channel_key(
                actor_rank, actor_rank, "pipeline_actor"
            )
            for actor_rank in local_actor_ranks
        }
        if self.shuffle_rollout:
            self.shuffle_generators = {
                actor_rank: torch.Generator().manual_seed(
                    self.cfg.actor.seed + actor_rank + self._rank * actor_ws
                )
                for actor_rank in local_actor_ranks
            }

    def _inject_realworld_reward_cfg(self, env_cfg: DictConfig):
        if not (self.use_reward_model and self.use_realworld_reward):
            return
        if env_cfg.env_type != "realworld":
            return

        reward_placements = self._component_placement.get_strategy(
            "reward"
        ).get_placement(Cluster())
        assert len(reward_placements) > 0, (
            "Reward placement must contain at least one worker."
        )
        reward_placement = reward_placements[0]
        reward_hardware_ranks = self._component_placement.get_hardware_ranks("reward")
        assert len(reward_hardware_ranks) > 0, (
            "Reward placement must contain at least one hardware rank."
        )

        override_cfg = OmegaConf.to_container(
            env_cfg.get("override_cfg", {}), resolve=True
        )
        override_cfg["use_reward_model"] = True
        override_cfg["reward_worker_cfg"] = OmegaConf.to_container(
            self.cfg.reward, resolve=True
        )
        override_cfg["reward_worker_hardware_rank"] = reward_hardware_ranks[0]
        override_cfg["reward_worker_node_rank"] = reward_placement.cluster_node_rank
        override_cfg["reward_worker_node_group"] = reward_placement.node_group_label
        override_cfg["reward_image_key"] = env_cfg.main_image_key
        setattr(env_cfg, "override_cfg", OmegaConf.create(override_cfg))

    def _setup_env_and_wrappers(self, env_cls, env_cfg, num_envs_per_stage: int):
        env_list = []

        for stage_id in range(self.stage_num):
            env = env_cls(
                cfg=env_cfg,
                num_envs=num_envs_per_stage,
                seed_offset=self._rank * self.stage_num + stage_id,
                total_num_processes=self._world_size * self.stage_num,
                worker_info=self.worker_info,
            )
            if env_cfg.video_cfg.save_video:
                env = RecordVideo(env, env_cfg.video_cfg)
            if env_cfg.get("data_collection", None) and getattr(
                env_cfg.data_collection, "enabled", False
            ):
                from rlinf.envs.wrappers import CollectEpisode

                env = CollectEpisode(
                    env,
                    save_dir=env_cfg.data_collection.save_dir,
                    rank=self._rank,
                    num_envs=num_envs_per_stage,
                    export_format=getattr(
                        env_cfg.data_collection, "export_format", "pickle"
                    ),
                    robot_type=getattr(env_cfg.data_collection, "robot_type", "panda"),
                    fps=getattr(env_cfg.data_collection, "fps", 10),
                    only_success=getattr(
                        env_cfg.data_collection, "only_success", False
                    ),
                    finalize_interval=getattr(
                        env_cfg.data_collection, "finalize_interval", 100
                    ),
                )
            env_list.append(env)
        return env_list

    def _init_env(self):
        for i in range(self.stage_num):
            if self.enable_train:
                if self.cfg.env.train.auto_reset:
                    extracted_obs, _ = self.env_list[i].reset()
                    self.last_obs_list.append(extracted_obs)
                    self.last_intervened_info_list.append((None, None))
                if self.train_enable_offload and self.cfg.env.train.get(
                    "enable_init_offload", True
                ):
                    get_env_attr(self.env_list[i], "offload")()
            if self.enable_eval:
                if self.eval_enable_offload:
                    get_env_attr(self.eval_env_list[i], "offload")()

    @Worker.timer("env_interact_step")
    def env_interact_step(
        self, chunk_actions: torch.Tensor, stage_id: int
    ) -> tuple[EnvOutput, dict[str, Any]]:
        """
        This function is used to interact with the environment.
        """
        chunk_actions = prepare_actions(
            raw_chunk_actions=chunk_actions,
            env_type=self.cfg.env.train.env_type,
            model_type=self.model_cfg.model_type,
            num_action_chunks=self.model_cfg.num_action_chunks,
            action_dim=self.model_cfg.action_dim,
            policy=self.model_cfg.get("policy_setup", None),
            wm_env_type=self.cfg.env.train.get("wm_env_type", None),
        )
        env_info = {}

        obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list = (
            self.env_list[stage_id].chunk_step(chunk_actions)
        )
        if isinstance(obs_list, (list, tuple)):
            extracted_obs = obs_list[-1] if obs_list else None
        if isinstance(infos_list, (list, tuple)):
            infos = infos_list[-1] if infos_list else None
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)
        final_obs = (
            self._build_chunk_final_obs(obs_list, infos_list)
            if self.use_external_reward_model
            else (
                infos["final_observation"]
                if isinstance(infos, dict) and "final_observation" in infos
                else None
            )
        )
        if not self.cfg.env.train.auto_reset:
            if self.cfg.env.train.ignore_terminations:
                if chunk_truncations[:, -1].any():
                    assert chunk_truncations[:, -1].all()
                    if "episode" in infos:
                        for key in infos["episode"]:
                            env_info[key] = infos["episode"][key].cpu()
            else:
                if "episode" in infos:
                    for key in infos["episode"]:
                        env_info[key] = infos["episode"][key].cpu()
        elif chunk_dones.any():
            if "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info["episode"]:
                    env_info[key] = final_info["episode"][key][chunk_dones[:, -1]].cpu()

        intervene_actions = (
            infos["intervene_action"] if "intervene_action" in infos else None
        )
        intervene_flags = infos["intervene_flag"] if "intervene_flag" in infos else None
        rlt_switch_flags = (
            infos["rlt_switch_flags"] if "rlt_switch_flags" in infos else None
        )
        if self.cfg.env.train.auto_reset and chunk_dones.any():
            if "intervene_action" in infos["final_info"]:
                intervene_actions = infos["final_info"]["intervene_action"]
                intervene_flags = infos["final_info"]["intervene_flag"]

        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=final_obs,
            rewards=chunk_rewards,
            env_infos=infos if isinstance(infos, dict) else None,
            dones=chunk_dones,
            terminations=chunk_terminations,
            truncations=chunk_truncations,
            intervene_actions=intervene_actions,
            intervene_flags=intervene_flags,
            rlt_switch_flags=rlt_switch_flags,
        )
        return env_output, env_info

    def _build_slot_metadata(
        self,
        *,
        rollout_epoch_id: int,
        stage_id: int,
        is_active: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Build batch-slot metadata without changing rollout execution."""
        num_slots = self.train_num_envs_per_stage
        group_size = int(self.cfg.algorithm.group_size)
        groups_per_epoch = int(self.cfg.env.train.total_num_envs // group_size)

        reset_state_ids = self._get_stage_reset_state_ids(stage_id)
        if (
            self.slot_state_managers is not None
            and self.slot_state_managers[stage_id] is not None
        ):
            return self.slot_state_managers[stage_id].build_metadata(reset_state_ids)

        local_slot_ids = torch.arange(num_slots, dtype=torch.long)
        global_slot_offset = (
            self._rank * self.train_batch_size
            + stage_id * self.train_num_envs_per_stage
        )
        global_slot_ids = local_slot_ids + int(global_slot_offset)

        if is_active is None:
            is_active = torch.ones(num_slots, dtype=torch.bool)
        active = is_active.detach().cpu().bool()
        if active.numel() != num_slots:
            raise ValueError(
                f"Expected {num_slots} active flags for slot metadata, "
                f"got {active.numel()}"
            )

        group_ids = rollout_epoch_id * groups_per_epoch + global_slot_ids // group_size

        return {
            "slot_id": local_slot_ids.reshape(num_slots, 1),
            "group_id": group_ids.reshape(num_slots, 1),
            "group_member_id": (global_slot_ids % group_size).reshape(num_slots, 1),
            "reset_state_id": reset_state_ids.reshape(num_slots, 1),
            "rollout_epoch_id": torch.full(
                (num_slots, 1), rollout_epoch_id, dtype=torch.long
            ),
            "is_active": active.reshape(num_slots, 1),
            "is_dummy": torch.zeros((num_slots, 1), dtype=torch.bool),
        }

    def _get_stage_reset_state_ids(self, stage_id: int) -> torch.Tensor:
        num_slots = self.train_num_envs_per_stage
        reset_state_ids = get_env_attr(
            self.env_list[stage_id], "reset_state_ids", None
        )
        if isinstance(reset_state_ids, torch.Tensor):
            reset_state_ids = reset_state_ids.detach().cpu().long()
        elif reset_state_ids is not None:
            reset_state_ids = torch.as_tensor(reset_state_ids, dtype=torch.long)
        else:
            reset_state_ids = torch.full((num_slots,), -1, dtype=torch.long)

        if reset_state_ids.numel() != num_slots:
            raise ValueError(
                f"Expected {num_slots} reset_state_ids for slot metadata, "
                f"got {reset_state_ids.numel()}"
            )
        return reset_state_ids.reshape(num_slots)

    def _new_slot_state_manager(self, *, stage_id: int) -> EnvSlotStateManager:
        group_size = int(self.cfg.algorithm.group_size)
        total_num_envs = int(self.cfg.env.train.total_num_envs)
        if total_num_envs % group_size != 0:
            raise ValueError(
                f"total_num_envs={total_num_envs} must be divisible by "
                f"group_size={group_size}"
            )
        return EnvSlotStateManager(
            num_slots=self.train_num_envs_per_stage,
            group_size=group_size,
            stage_id=stage_id,
            rank=self._rank,
            total_num_envs=total_num_envs,
            train_batch_size=self.train_batch_size,
            train_num_envs_per_stage=self.train_num_envs_per_stage,
            reset_state_ids=self._get_stage_reset_state_ids(stage_id),
        )

    def _ensure_slot_state_managers(self) -> None:
        if not self.collect_slot_metadata:
            self.slot_state_managers = None
            return
        if self.slot_state_managers is not None:
            return
        self.slot_state_managers = [
            self._new_slot_state_manager(stage_id=stage_id)
            for stage_id in range(self.stage_num)
        ]

    def _reset_slot_state_managers(self) -> None:
        if not self.collect_slot_metadata:
            self.slot_state_managers = None
            return
        self.slot_state_managers = [
            self._new_slot_state_manager(stage_id=stage_id)
            for stage_id in range(self.stage_num)
        ]

    def _mark_slot_managers_done(
        self, *, stage_id: int, dones: torch.Tensor | None
    ) -> torch.Tensor | None:
        if (
            self.slot_state_managers is None
            or self.slot_state_managers[stage_id] is None
        ):
            return None
        return self.slot_state_managers[stage_id].mark_done(dones)

    def _num_env_reset_states(self, stage_id: int) -> int:
        dataset = get_env_attr(self.env_list[stage_id], "dataset", None)
        if dataset is None:
            raise RuntimeError(
                "continuous_batching slot reassignment requires env.dataset "
                "to derive reset_state_id."
            )
        return len(dataset)

    def _reassign_done_slots(
        self,
        *,
        stage_id: int,
        newly_done: torch.Tensor | None,
        allow_real_groups: bool = True,
    ) -> dict[str, torch.Tensor] | None:
        if newly_done is None or not newly_done.any():
            return None
        if self.slot_state_managers is None:
            return None
        done_slots = torch.nonzero(
            newly_done.detach().cpu().bool(), as_tuple=False
        ).flatten()
        if done_slots.numel() == 0:
            return None
        return self.slot_state_managers[stage_id].reassign_slots(
            done_slots,
            num_reset_states=self._num_env_reset_states(stage_id),
            allow_real_groups=allow_real_groups,
        )

    def _reset_reassigned_env_slots(
        self, *, stage_id: int, reassignment: dict[str, torch.Tensor] | None
    ) -> dict[str, Any] | None:
        if reassignment is None:
            return None
        reset_slots = get_env_attr(self.env_list[stage_id], "reset_slots", None)
        if not callable(reset_slots):
            raise RuntimeError(
                "continuous_batching.enabled requires env.reset_slots(...) "
                "for slot reuse."
            )
        return reset_slots(
            reassignment["slot_indices"],
            reassignment["episode_indices"],
        )

    def _new_rollout_collector(self):
        if self.continuous_batching_enabled:
            return ContinuousBatchingRolloutCollector(
                max_episode_length=self.cfg.env.train.max_episode_steps,
                group_size=int(self.cfg.algorithm.group_size),
                batch_size=self.train_num_envs_per_stage,
                rollout_epoch=self.rollout_epoch,
                target_chunk_steps=self.n_train_chunk_steps,
            )
        return EmbodiedRolloutResult(
            max_episode_length=self.cfg.env.train.max_episode_steps
        )

    def _profile_early_stop_active(self) -> bool:
        return self.profile_early_stop_enabled and not self.continuous_batching_enabled

    def _profile_early_stop_ensure_dir(self) -> None:
        os.makedirs(os.path.join(self._profile_early_stop_dir, "groups"), exist_ok=True)

    def _profile_early_stop_group_key(
        self, *, rollout_epoch_id: int, stage_id: int, slot_idx: int
    ) -> tuple[int, int, int, int]:
        group_size = int(self.cfg.algorithm.group_size)
        groups_per_epoch = int(self.cfg.env.train.total_num_envs // group_size)
        global_slot_offset = (
            self._rank * self.train_batch_size
            + stage_id * self.train_num_envs_per_stage
        )
        global_slot_id = int(global_slot_offset + slot_idx)
        group_id = rollout_epoch_id * groups_per_epoch + global_slot_id // group_size
        return self._profile_early_stop_step_id, rollout_epoch_id, stage_id, group_id

    def _profile_early_stop_member_id(self, *, stage_id: int, slot_idx: int) -> int:
        group_size = int(self.cfg.algorithm.group_size)
        global_slot_offset = (
            self._rank * self.train_batch_size
            + stage_id * self.train_num_envs_per_stage
        )
        return int((global_slot_offset + slot_idx) % group_size)

    def _profile_early_stop_actions(
        self, actions: torch.Tensor | None, env_output: EnvOutput
    ) -> torch.Tensor | None:
        if actions is None:
            return None
        raw_actions = actions.detach().cpu().float().contiguous()
        if raw_actions.dim() == 2:
            raw_actions = raw_actions.reshape(
                raw_actions.shape[0], int(self.model_cfg.num_action_chunks), -1
            )
        if raw_actions.dim() != 3:
            raise ValueError(
                f"profile_early_stop expects actions with 2 or 3 dims, got "
                f"{tuple(raw_actions.shape)}."
            )

        prepared_actions = prepare_actions(
            raw_chunk_actions=raw_actions,
            env_type=self.cfg.env.train.env_type,
            model_type=self.model_cfg.model_type,
            num_action_chunks=self.model_cfg.num_action_chunks,
            action_dim=self.model_cfg.action_dim,
            policy=self.model_cfg.get("policy_setup", None),
            wm_env_type=self.cfg.env.train.get("wm_env_type", None),
        )
        if isinstance(prepared_actions, np.ndarray):
            prepared_actions = torch.from_numpy(prepared_actions)
        prepared_actions = prepared_actions.detach().cpu().float().contiguous()

        if env_output.intervene_actions is not None:
            intervene_actions = env_output.intervene_actions.detach().cpu().float()
            if intervene_actions.dim() == 2:
                intervene_actions = intervene_actions.reshape(
                    prepared_actions.shape[0], prepared_actions.shape[1], -1
                )
            intervene_flags = env_output.intervene_flags
            if intervene_flags is not None:
                intervene_flags = intervene_flags.detach().cpu().bool()
                if intervene_flags.dim() == 1:
                    intervene_flags = intervene_flags[:, None]
                intervene_flags = intervene_flags.reshape(
                    prepared_actions.shape[0], prepared_actions.shape[1], 1
                )
                prepared_actions = torch.where(
                    intervene_flags.expand_as(prepared_actions),
                    intervene_actions.to(prepared_actions.dtype),
                    prepared_actions,
                )
        return prepared_actions

    @staticmethod
    def _profile_early_stop_bool_matrix(
        value: torch.Tensor | None, batch_size: int, chunk_size: int
    ) -> torch.Tensor:
        if value is None:
            return torch.zeros((batch_size, chunk_size), dtype=torch.bool)
        value = value.detach().cpu().bool()
        if value.dim() == 1:
            value = value[:, None]
        if value.shape[0] != batch_size:
            raise ValueError(
                f"profile_early_stop expected batch size {batch_size}, "
                f"got {value.shape[0]}."
            )
        if value.shape[1] != chunk_size:
            if value.shape[1] == 1:
                padded = torch.zeros((batch_size, chunk_size), dtype=torch.bool)
                padded[:, -1:] = value
                value = padded
            else:
                raise ValueError(
                    f"profile_early_stop expected chunk size {chunk_size}, "
                    f"got {value.shape[1]}."
                )
        return value

    def _profile_early_stop_new_group(
        self, *, key: tuple[int, int, int, int], action_dim: int
    ) -> dict[str, Any]:
        step_id, rollout_epoch_id, stage_id, group_id = key
        group_size = int(self.cfg.algorithm.group_size)
        max_episode_steps = int(self.cfg.env.train.max_episode_steps)
        return {
            "metadata": {
                "global_step": step_id,
                "rollout_epoch_id": rollout_epoch_id,
                "rank": self._rank,
                "stage_id": stage_id,
                "group_id": group_id,
                "group_size": group_size,
                "max_episode_steps": max_episode_steps,
                "num_action_chunks": int(self.model_cfg.num_action_chunks),
                "action_dim": int(action_dim),
            },
            "actions": torch.zeros(
                (group_size, max_episode_steps, action_dim), dtype=torch.float32
            ),
            "lengths": torch.zeros(group_size, dtype=torch.long),
            "done_steps": torch.full((group_size,), -1, dtype=torch.long),
            "success": torch.zeros(group_size, dtype=torch.bool),
            "failure": torch.zeros(group_size, dtype=torch.bool),
            "done": torch.zeros(group_size, dtype=torch.bool),
        }

    def _profile_early_stop_write_group(
        self, key: tuple[int, int, int, int], group: dict[str, Any]
    ) -> None:
        self._profile_early_stop_ensure_dir()
        metadata = dict(group["metadata"])
        success = group["success"].cpu().bool()
        failure = group["failure"].cpu().bool()
        done = group["done"].cpu().bool()
        lengths = group["lengths"].cpu().long()
        done_steps = group["done_steps"].cpu().long()
        group_filename = (
            f"step{metadata['global_step']:06d}_"
            f"epoch{metadata['rollout_epoch_id']:03d}_"
            f"rank{metadata['rank']:03d}_"
            f"stage{metadata['stage_id']:02d}_"
            f"group{metadata['group_id']:06d}.pt"
        )
        group_path = os.path.join(
            self._profile_early_stop_dir, "groups", group_filename
        )
        torch.save(
            {
                "metadata": metadata,
                "actions": group["actions"].cpu().contiguous(),
                "lengths": lengths,
                "done_steps": done_steps,
                "success": success,
                "failure": failure,
                "done": done,
            },
            group_path,
        )

        record = {
            **metadata,
            "action_file": os.path.relpath(group_path, self._profile_early_stop_dir),
            "success": [bool(x) for x in success.tolist()],
            "failure": [bool(x) for x in failure.tolist()],
            "done": [bool(x) for x in done.tolist()],
            "lengths": [int(x) for x in lengths.tolist()],
            "done_steps": [int(x) for x in done_steps.tolist()],
            "success_count": int(success.sum().item()),
            "failure_count": int(failure.sum().item()),
            "all_failed": bool(success.sum().item() == 0),
        }
        self._profile_early_stop_history.append(record)
        with open(
            os.path.join(
                self._profile_early_stop_dir,
                f"groups_env_rank{self._rank}.jsonl",
            ),
            "a",
            encoding="utf-8",
        ) as f:
            f.write(json.dumps(record) + "\n")

    def _profile_early_stop_record_chunk(
        self,
        *,
        rollout_epoch_id: int,
        stage_id: int,
        actions: torch.Tensor | None,
        env_output: EnvOutput,
    ) -> None:
        if not self._profile_early_stop_active():
            return
        prepared_actions = self._profile_early_stop_actions(actions, env_output)
        if prepared_actions is None:
            return

        batch_size, chunk_size, action_dim = prepared_actions.shape
        terminations = self._profile_early_stop_bool_matrix(
            env_output.terminations, batch_size, chunk_size
        )
        truncations = self._profile_early_stop_bool_matrix(
            env_output.truncations, batch_size, chunk_size
        )
        dones = terminations | truncations
        max_episode_steps = int(self.cfg.env.train.max_episode_steps)

        for slot_idx in range(batch_size):
            key = self._profile_early_stop_group_key(
                rollout_epoch_id=rollout_epoch_id,
                stage_id=stage_id,
                slot_idx=slot_idx,
            )
            member_id = self._profile_early_stop_member_id(
                stage_id=stage_id, slot_idx=slot_idx
            )
            group = self._profile_early_stop_groups.get(key)
            if group is None:
                group = self._profile_early_stop_new_group(
                    key=key, action_dim=action_dim
                )
                self._profile_early_stop_groups[key] = group

            if bool(group["done"][member_id].item()):
                continue

            start = int(group["lengths"][member_id].item())
            if start >= max_episode_steps:
                group["done"][member_id] = True
                group["failure"][member_id] = True
                group["done_steps"][member_id] = max_episode_steps
                continue

            remaining = max_episode_steps - start
            copy_len = min(chunk_size, remaining)
            done_positions = torch.nonzero(
                dones[slot_idx, :copy_len], as_tuple=False
            ).flatten()
            if done_positions.numel() > 0:
                copy_len = int(done_positions[0].item()) + 1

            if copy_len > 0:
                group["actions"][member_id, start : start + copy_len] = (
                    prepared_actions[slot_idx, :copy_len]
                )
                group["lengths"][member_id] = start + copy_len

            if done_positions.numel() > 0:
                done_idx = int(done_positions[0].item())
                succeeded = bool(terminations[slot_idx, done_idx].item())
                group["done"][member_id] = True
                group["success"][member_id] = succeeded
                group["failure"][member_id] = not succeeded
                group["done_steps"][member_id] = int(group["lengths"][member_id].item())
            elif int(group["lengths"][member_id].item()) >= max_episode_steps:
                group["done"][member_id] = True
                group["failure"][member_id] = True
                group["done_steps"][member_id] = max_episode_steps

            if bool(group["done"].all().item()):
                self._profile_early_stop_write_group(key, group)
                del self._profile_early_stop_groups[key]

    def finalize_profile_early_stop(self) -> dict[str, Any]:
        if not self.profile_early_stop_enabled or self.continuous_batching_enabled:
            return {}
        self._profile_early_stop_ensure_dir()
        for key, group in list(self._profile_early_stop_groups.items()):
            not_done = ~group["done"]
            group["failure"][not_done] = True
            group["done_steps"][not_done] = group["lengths"][not_done]
            self._profile_early_stop_write_group(key, group)
            del self._profile_early_stop_groups[key]

        total_groups = len(self._profile_early_stop_history)
        total_trajectories = sum(
            int(record["group_size"]) for record in self._profile_early_stop_history
        )
        success_trajectories = sum(
            int(record["success_count"]) for record in self._profile_early_stop_history
        )
        failure_trajectories = sum(
            int(record["failure_count"]) for record in self._profile_early_stop_history
        )
        all_failed_groups = sum(
            1 for record in self._profile_early_stop_history if record["all_failed"]
        )
        summary = {
            "rank": self._rank,
            "total_groups": total_groups,
            "total_trajectories": total_trajectories,
            "success_trajectories": success_trajectories,
            "failure_trajectories": failure_trajectories,
            "trajectory_success_rate": (
                success_trajectories / total_trajectories
                if total_trajectories > 0
                else 0.0
            ),
            "all_failed_groups": all_failed_groups,
            "all_failed_group_ratio": (
                all_failed_groups / total_groups if total_groups > 0 else 0.0
            ),
            "groups": self._profile_early_stop_history,
        }
        with open(
            os.path.join(
                self._profile_early_stop_dir, f"summary_env_rank{self._rank}.json"
            ),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(summary, f, indent=2)
        return summary

    def env_evaluate_step(
        self, raw_actions: torch.Tensor, stage_id: int
    ) -> tuple[EnvOutput, dict[str, Any]]:
        """
        This function is used to evaluate the environment.
        """
        chunk_actions = prepare_actions(
            raw_chunk_actions=raw_actions,
            env_type=self.cfg.env.eval.env_type,
            model_type=self.model_cfg.model_type,
            num_action_chunks=self.model_cfg.num_action_chunks,
            action_dim=self.model_cfg.action_dim,
            policy=self.model_cfg.get("policy_setup", None),
            wm_env_type=self.cfg.env.eval.get("wm_env_type", None),
        )
        env_info = {}

        obs_list, _, chunk_terminations, chunk_truncations, infos_list = (
            self.eval_env_list[stage_id].chunk_step(chunk_actions)
        )
        if isinstance(obs_list, (list, tuple)):
            extracted_obs = obs_list[-1] if obs_list else None
        if isinstance(infos_list, (list, tuple)):
            infos = infos_list[-1] if infos_list else None
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)
        final_obs = (
            self._build_chunk_final_obs(obs_list, infos_list)
            if self.use_external_reward_model
            else (
                infos["final_observation"]
                if isinstance(infos, dict) and "final_observation" in infos
                else None
            )
        )

        current_dones = chunk_dones[:, -1]  # [num_envs] bool
        if self.cfg.env.eval.auto_reset:
            newly_done = current_dones
        else:
            prev = self.eval_prev_done[stage_id].to(current_dones.device)
            newly_done = current_dones & ~prev
            self.eval_prev_done[stage_id] = prev | current_dones

        if newly_done.any():
            if "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info["episode"]:
                    env_info[key] = final_info["episode"][key][newly_done].cpu()
            elif "episode" in infos:
                for key in infos["episode"]:
                    env_info[key] = infos["episode"][key][newly_done].cpu()

        rlt_switch_flags = (
            infos["rlt_switch_flags"] if "rlt_switch_flags" in infos else None
        )

        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=final_obs,
            rlt_switch_flags=rlt_switch_flags,
        )
        return env_output, env_info

    def _build_chunk_final_obs(self, obs_list, infos_list):
        """Build per-env terminal observations for a whole chunk.

        Matches the old wrapper semantics:
        - default to the last rollout observation for each env
        - if an env terminated earlier in the chunk, replace that env's observation
          with the true `final_observation` captured at that substep
        """
        if not isinstance(obs_list, (list, tuple)) or len(obs_list) == 0:
            return None

        last_obs = obs_list[-1]
        if not isinstance(last_obs, dict):
            return None

        merged_final_obs = copy_dict_tensor(last_obs)

        if not isinstance(infos_list, (list, tuple)):
            return merged_final_obs

        for step_infos in infos_list:
            if not isinstance(step_infos, dict):
                continue
            if (
                "final_observation" not in step_infos
                or "_final_observation" not in step_infos
            ):
                continue

            final_obs = step_infos["final_observation"]
            reset_mask = step_infos["_final_observation"]
            if final_obs is None or reset_mask is None:
                continue
            reset_mask = (
                reset_mask.detach().cpu().numpy()
                if isinstance(reset_mask, torch.Tensor)
                else np.asarray(reset_mask)
            )
            done_mask = (
                reset_mask.any(axis=-1)
                if reset_mask.ndim > 1
                else reset_mask.astype(bool)
            )
            if not done_mask.any():
                continue

            for key, value in merged_final_obs.items():
                if key not in final_obs:
                    continue

                final_value = final_obs[key]
                if isinstance(value, torch.Tensor) and isinstance(
                    final_value, torch.Tensor
                ):
                    dst_mask = torch.as_tensor(done_mask, device=value.device)
                    src_mask = dst_mask.to(device=final_value.device)
                    merged_final_obs[key][dst_mask] = final_value[src_mask]
                elif isinstance(value, np.ndarray) and isinstance(
                    final_value, np.ndarray
                ):
                    merged_final_obs[key][done_mask] = final_value[done_mask]

        return merged_final_obs

    @staticmethod
    def _infer_rollout_batch_size(data: Any) -> int:
        """Infer batch dim for routed shards; supports RolloutResult and plain tensor payloads.

        When the channel carries a non-``RolloutResult`` shard (e.g. reward tensor or eval
        actions) into a rollout recv, avoid assuming dataclass fields and delegate or use
        the leading dimension of dense arrays.
        """

        if isinstance(data, torch.Tensor) or isinstance(data, np.ndarray):
            return int(data.shape[0])
        if isinstance(data, RolloutResult):
            for field_name in (
                "actions",
                "prev_logprobs",
                "prev_values",
                "bootstrap_values",
                "versions",
            ):
                value = getattr(data, field_name, None)
                if isinstance(value, torch.Tensor):
                    return int(value.shape[0])
            forward_inputs = getattr(data, "forward_inputs", None)
            if forward_inputs:
                first_tensor = next(iter(forward_inputs.values()))
                if isinstance(first_tensor, torch.Tensor):
                    return int(first_tensor.shape[0])
            raise ValueError("Cannot infer batch size from rollout result.")
        from rlinf.scheduler import infer_batch_size

        return infer_batch_size(data)

    @Worker.timer("compute_bootstrap_rewards")
    def compute_bootstrap_rewards(
        self,
        env_output: EnvOutput,
        bootstrap_values: torch.Tensor | None,
        reward_model_output: torch.Tensor | None,
    ) -> torch.Tensor | None:
        rewards = env_output.rewards
        if rewards is None:
            return None

        if reward_model_output is not None:
            reward_model_output = reward_model_output.to(rewards.dtype)
            rewards = (
                self.env_reward_weight * rewards
                + self.reward_weight * reward_model_output
            )

        adjusted_rewards = rewards.clone()
        if (
            bootstrap_values is None
            or not self.cfg.env.train.auto_reset
            or env_output.dones is None
        ):
            return adjusted_rewards

        bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "standard")
        if bootstrap_type == "standard":
            last_step_truncations = env_output.truncations[:, -1]
        else:
            last_step_truncations = env_output.dones[:, -1]

        if not last_step_truncations.any():
            return adjusted_rewards

        final_values = torch.zeros_like(adjusted_rewards[:, -1], dtype=torch.float32)
        final_values[last_step_truncations] = (
            bootstrap_values[last_step_truncations].reshape(-1).to(torch.float32)
        )
        adjusted_rewards[:, -1] += self.cfg.algorithm.gamma * final_values
        return adjusted_rewards

    def finish_rollout(self, mode="train"):
        # reset
        if mode == "train":
            for i in range(self.stage_num):
                if self.cfg.env.train.video_cfg.save_video:
                    flush_video = get_env_attr(self.env_list[i], "flush_video")
                    if callable(flush_video):
                        flush_video()
                self.env_list[i].update_reset_state_ids()
        elif mode == "eval":
            for i in range(self.stage_num):
                if self.cfg.env.eval.video_cfg.save_video:
                    flush_video = get_env_attr(self.eval_env_list[i], "flush_video")
                    if callable(flush_video):
                        flush_video()
                if not self.cfg.env.eval.auto_reset:
                    self.eval_env_list[i].update_reset_state_ids()

    @Worker.timer("get_reward_model_output")
    def get_reward_model_output(
        self,
        env_output: EnvOutput,
        send_channel: Channel,
        recv_channel: Channel,
        stage_id: int | None = None,
        last_run: bool = False,
    ):
        if self.reward_mode in {"per_step", "history_buffer"}:
            observations = (
                env_output.final_obs
                if env_output.final_obs is not None
                else env_output.obs
            )
        elif self.reward_mode == "terminal" and env_output.final_obs is not None:
            observations = env_output.final_obs
        else:
            return None
        reward_input = dict(observations)
        if env_output.env_infos is not None:
            reward_input["env_infos"] = self._select_reward_env_infos(
                env_output.env_infos
            )

        dones = env_output.dones
        if dones is not None and getattr(dones, "ndim", 0) > 1:
            dones = dones[:, -1]
            reward_input.update({"dones": dones})

        if self.reward_mode == "history_buffer":
            if stage_id is None:
                raise ValueError("stage_id is required for history-buffer reward.")
            history_manager = self.train_history_managers[stage_id]
            history_manager.append_to_history_entries(observations)
            history_input, history_lengths = history_manager.build_history_input(
                dones=dones
            )
            reward_input["history_input"] = history_input
            self.history_lengths[stage_id] = dict(history_lengths)

        if last_run:
            reward_input.update(
                {
                    "last_run": torch.ones(
                        (self.train_num_envs_per_stage, 1), dtype=torch.bool
                    )
                }
            )
        self.send_to(
            group_name=self.cfg.reward.group_name,
            channel=send_channel,
            data=reward_input,
            tag="train_reward_obs",
            async_op=True,
            decoupled_mode=self.env_decoupled_mode,
        )
        reward_output = self.recv_from(
            group_name=self.cfg.reward.group_name,
            channel=recv_channel,
            tag="train_reward_obs",
            batch_size=self.train_batch_size,
            decoupled_mode=self.env_decoupled_mode,
        )
        if self.reward_mode != "terminal" or reward_output is None:
            return reward_output
        return self._scatter_terminal_reward_output(
            env_output=env_output, reward_output=reward_output
        )

    def _select_reward_env_infos(self, env_infos: dict[str, Any]) -> dict[str, Any]:
        reward_env_infos = {}
        for key in self.env_infos_reward_keys:
            if key not in env_infos:
                continue
            reward_env_infos[key] = clone_nested_to_cpu(env_infos[key])
        return reward_env_infos

    def _scatter_terminal_reward_output(
        self,
        env_output: EnvOutput,
        reward_output: torch.Tensor,
    ) -> torch.Tensor:
        if env_output.rewards is None or env_output.dones is None:
            return reward_output

        done_envs = env_output.dones.any(dim=1)
        sparse_rewards = torch.zeros_like(env_output.rewards, dtype=reward_output.dtype)
        if not done_envs.any():
            return sparse_rewards

        done_steps = env_output.dones.to(torch.int64).argmax(dim=1)
        sparse_rewards[done_envs, done_steps[done_envs]] = (
            reward_output[done_envs].reshape(-1).to(sparse_rewards.dtype)
        )
        return sparse_rewards

    def assign_history_reward(self, stage_id: int, reward_model_output: torch.Tensor):
        reward_assign_lengths = [
            min(
                history_buffer_length[env_id]
                for history_buffer_length in self.history_lengths[stage_id].values()
            )
            for env_id in range(self.train_num_envs_per_stage)
        ]
        rollout_rewards = self.rollout_results[stage_id].rewards
        rollout_rewards_length = len(rollout_rewards)
        reward_assign_lengths = [
            min(reward_assign_length, rollout_rewards_length)
            for reward_assign_length in reward_assign_lengths
        ]
        if not any(reward_assign_lengths):
            return
        reward = (self.reward_weight * reward_model_output).to(
            rollout_rewards[-1].dtype
        )
        for env_id, reward_assign_length in enumerate(reward_assign_lengths):
            for reward_assign_step in range(2, reward_assign_length + 1):
                rollout_rewards[-reward_assign_step][env_id] += reward[env_id]

    @Worker.timer("env/bootstrap_step")
    def bootstrap_step(self) -> list[EnvOutput]:
        def get_zero_dones() -> torch.Tensor:
            return (
                torch.zeros((self.train_num_envs_per_stage,), dtype=bool)
                .unsqueeze(1)
                .repeat(1, self.model_cfg.num_action_chunks)
            )

        env_outputs: list[EnvOutput] = []
        if not self.cfg.env.train.auto_reset:
            for stage_id in range(self.stage_num):
                self.env_list[stage_id].is_start = True
                extracted_obs, infos = self.env_list[stage_id].reset()
                dones = get_zero_dones()
                terminations = dones.clone()
                truncations = dones.clone()

                env_output = EnvOutput(
                    obs=extracted_obs,
                    dones=dones,
                    terminations=terminations,
                    truncations=truncations,
                    final_obs=(
                        infos["final_observation"]
                        if "final_observation" in infos
                        else None
                    ),
                    intervene_actions=None,
                    intervene_flags=None,
                )
                env_outputs.append(env_output)
        else:
            dones = get_zero_dones()
            terminations = dones.clone()
            truncations = dones.clone()

            for stage_id in range(self.stage_num):
                env_output = EnvOutput(
                    obs=self.last_obs_list[stage_id],
                    rewards=None,
                    dones=dones,
                    terminations=terminations,
                    truncations=truncations,
                    intervene_actions=self.last_intervened_info_list[stage_id][0],
                    intervene_flags=self.last_intervened_info_list[stage_id][1],
                )
                env_outputs.append(env_output)

        return env_outputs

    def _send_train_bootstrap(
        self, rollout_channel: Channel, env_outputs: list[EnvOutput]
    ) -> None:
        for stage_id in range(self.stage_num):
            self._send_train_env_output(
                rollout_channel, stage_id, env_outputs[stage_id]
            )

    def _send_train_env_output(
        self,
        rollout_channel: Channel,
        stage_id: int,
        env_output: EnvOutput,
        *,
        continuous_done: bool = False,
    ) -> None:
        env_batch = env_output.to_dict()
        data = {
            "obs": env_batch["obs"],
            "final_obs": env_batch["final_obs"],
        }
        if continuous_done:
            from rlinf.scheduler import infer_batch_size

            data["continuous_done"] = torch.ones(
                (infer_batch_size(env_batch["obs"]),), dtype=torch.bool
            )
        if self.enable_rlt:
            data["rlt_switch_flags"] = env_batch.get("rlt_switch_flags", None)
        self.send_to(
            group_name=self.cfg.rollout.group_name,
            channel=rollout_channel,
            data=data,
            mode="train",
            tag="rollout_results",
            route_key=stage_id if not self.env_decoupled_mode else None,
            decoupled_mode=self.env_decoupled_mode,
        )

    def _bootstrap_and_send_train(self, rollout_channel: Channel) -> list[EnvOutput]:
        env_outputs = self.bootstrap_step()
        self._send_train_bootstrap(rollout_channel, env_outputs)
        return env_outputs

    def prefetch_train_bootstrap(self, rollout_channel: Channel) -> None:
        """Prepare and send the first env batch for the next training rollout."""
        if self._prefetched_train_bootstrap is not None:
            raise RuntimeError(
                "A prefetched train bootstrap already exists. "
                "Call interact() to consume it before prefetching again."
            )
        self._prefetched_train_bootstrap = self._bootstrap_and_send_train(
            rollout_channel
        )

    def record_env_metrics(
        self,
        env_metrics: dict[str, list],
        env_info: dict[str, Any],
    ):
        for key, value in env_info.items():
            env_metrics.setdefault(key, []).append(value)

    def store_last_obs_and_intervened_info(self, env_output_list: list[EnvOutput]):
        self.last_obs_list = [env_output.obs for env_output in env_output_list]
        self.last_intervened_info_list = [
            (env_output.intervene_actions, env_output.intervene_flags)
            for env_output in env_output_list
        ]

    def _continuous_target_groups_per_stage(self) -> int:
        group_size = int(self.cfg.algorithm.group_size)
        target_trajectories = self.train_num_envs_per_stage * self.rollout_epoch
        if target_trajectories % group_size != 0:
            raise ValueError(
                "continuous_batching requires per-stage target trajectories to be "
                f"divisible by group_size. Got target_trajectories={target_trajectories}, "
                f"group_size={group_size}."
            )
        return target_trajectories // group_size

    def _continuous_max_chunk_steps(self) -> int:
        return max(1, self.n_train_chunk_steps * max(1, self.rollout_epoch))

    def _continuous_targets_complete(self) -> bool:
        if self.slot_state_managers is None:
            return False
        return all(
            manager.target_real_groups_complete
            for manager in self.slot_state_managers
        )

    @staticmethod
    def _slice_metadata_slots(
        metadata: dict[str, torch.Tensor], slot_indices: torch.Tensor | list[int]
    ) -> dict[str, torch.Tensor]:
        slots = torch.as_tensor(slot_indices, dtype=torch.long).flatten()
        return {
            key: value.index_select(0, slots.to(value.device)).cpu().contiguous()
            for key, value in metadata.items()
        }

    def _append_continuous_initial_boundary(
        self,
        *,
        stage_id: int,
        slot_indices: torch.Tensor | list[int] | None = None,
    ) -> None:
        metadata = self._build_slot_metadata(rollout_epoch_id=0, stage_id=stage_id)
        if slot_indices is not None:
            slots = torch.as_tensor(slot_indices, dtype=torch.long).flatten()
            if slots.numel() == 0:
                return
            metadata = self._slice_metadata_slots(metadata, slots)
            num_slots = int(slots.numel())
        else:
            num_slots = self.train_num_envs_per_stage

        dones = torch.zeros(
            (num_slots, self.model_cfg.num_action_chunks), dtype=torch.bool
        )
        self.rollout_results[stage_id].append_step_result(
            ChunkStepResult(
                dones=dones,
                terminations=dones.clone(),
                truncations=dones.clone(),
                slot_metadata=metadata,
            )
        )

    def _append_continuous_step_result(
        self,
        *,
        stage_id: int,
        rollout_result: RolloutResult,
        env_output: EnvOutput,
        rewards: torch.Tensor | None,
        slot_metadata: dict[str, torch.Tensor],
    ) -> None:
        actions = rollout_result.forward_inputs.get("action", rollout_result.actions)
        action_tokens = rollout_result.forward_inputs.get("action_tokens", None)
        if actions is not None and action_tokens is not None:
            if actions.shape[:2] != action_tokens.shape[:2]:
                raise ValueError(
                    "continuous_batching action/action_tokens length mismatch: "
                    f"actions.shape={tuple(actions.shape)}, "
                    f"action_tokens.shape={tuple(action_tokens.shape)}."
                )
        chunk_step_result = ChunkStepResult(
            actions=actions,
            prev_logprobs=(
                rollout_result.prev_logprobs if self.collect_prev_infos else None
            ),
            # Continuous batching v1 aligns with auto_reset=False and GRPO actor loss.
            # Do not collect prev_values until final-value/bootstrap semantics are added.
            prev_values=None,
            forward_inputs=rollout_result.forward_inputs,
            versions=rollout_result.versions,
            dones=env_output.dones,
            truncations=env_output.truncations,
            terminations=env_output.terminations,
            rewards=rewards,
            slot_metadata=slot_metadata,
        )
        self.rollout_results[stage_id].append_step_result(chunk_step_result)
        if rollout_result.save_flags is not None:
            self.rollout_results[stage_id].mark_last_step_with_flags(
                rollout_result.save_flags
            )

    def _send_continuous_stop(self, rollout_channel: Channel, env_outputs: list[EnvOutput]):
        for stage_id, env_output in enumerate(env_outputs):
            self._send_train_env_output(
                rollout_channel,
                stage_id,
                env_output,
                continuous_done=True,
            )

    async def _run_interact_once_continuous(
        self,
        input_channel: Channel,
        rollout_channel: Channel,
        reward_channel: Channel | None,
        actor_channel: Channel | None,
        *,
        cooperative_yield: bool,
    ) -> dict[str, torch.Tensor]:
        if self.use_training_pipeline:
            raise NotImplementedError(
                "continuous_batching does not support use_training_pipeline yet."
            )
        if self.collect_transitions:
            raise NotImplementedError(
                "continuous_batching does not support collect_transitions yet."
            )

        self.rollout_results = [
            self._new_rollout_collector() for _ in range(self.stage_num)
        ]
        env_metrics = defaultdict(list)

        env_outputs = self.bootstrap_step()
        self._reset_slot_state_managers()
        target_groups_per_stage = self._continuous_target_groups_per_stage()
        assert self.slot_state_managers is not None
        for manager in self.slot_state_managers:
            manager.set_target_real_group_count(target_groups_per_stage)

        for stage_id in range(self.stage_num):
            self._append_continuous_initial_boundary(stage_id=stage_id)
            self._send_train_env_output(rollout_channel, stage_id, env_outputs[stage_id])

        max_chunk_steps = self._continuous_max_chunk_steps()
        chunk_step_idx = 0
        stopped_stages = [False for _ in range(self.stage_num)]
        while not all(stopped_stages):
            if chunk_step_idx >= max_chunk_steps:
                raise RuntimeError(
                    "continuous_batching did not complete target groups within "
                    f"{max_chunk_steps} chunks. Completed per stage: "
                    f"{[m.completed_real_group_count for m in self.slot_state_managers]}, "
                    f"target={target_groups_per_stage}."
                )

            next_env_outputs: list[EnvOutput] = [None] * self.stage_num
            for stage_id in range(self.stage_num):
                if stopped_stages[stage_id]:
                    next_env_outputs[stage_id] = env_outputs[stage_id]
                    continue
                if cooperative_yield:
                    await asyncio.sleep(0)

                manager = self.slot_state_managers[stage_id]
                stage_already_complete = manager.target_real_groups_complete
                step_metadata = self._build_slot_metadata(
                    rollout_epoch_id=0,
                    stage_id=stage_id,
                )

                rollout_result = self.recv_from(
                    group_name=self.cfg.rollout.group_name,
                    channel=input_channel,
                    tag="train_rollout_results",
                    route_key=stage_id if not self.env_decoupled_mode else None,
                    batch_size=self.train_batch_size,
                    merge_fn=RolloutResult.merge_rollout_results,
                    infer_batch_size_fn=self._infer_rollout_batch_size,
                    decoupled_mode=self.env_decoupled_mode,
                )

                env_output, env_info = self.env_interact_step(
                    rollout_result.actions, stage_id
                )

                reward_model_output = None
                if reward_channel is not None and not stage_already_complete:
                    reward_model_output = self.get_reward_model_output(
                        env_output,
                        send_channel=reward_channel,
                        recv_channel=input_channel,
                        stage_id=stage_id,
                    )
                    if reward_model_output is not None:
                        env_metrics["reward_model_output"].append(
                            reward_model_output.detach().float().reshape(-1).cpu()
                        )

                if not stage_already_complete:
                    rewards = self.compute_bootstrap_rewards(
                        env_output,
                        bootstrap_values=None,
                        reward_model_output=reward_model_output,
                    )
                    self._append_continuous_step_result(
                        stage_id=stage_id,
                        rollout_result=rollout_result,
                        env_output=env_output,
                        rewards=rewards,
                        slot_metadata=step_metadata,
                    )
                    if (
                        self.reward_mode == "history_buffer"
                        and self.history_reward_assign
                        and reward_model_output is not None
                    ):
                        self.assign_history_reward(stage_id, reward_model_output)

                newly_done = self._mark_slot_managers_done(
                    stage_id=stage_id, dones=env_output.dones
                )
                reassignment = self._reassign_done_slots(
                    stage_id=stage_id,
                    newly_done=newly_done,
                    allow_real_groups=not manager.target_real_groups_complete,
                )
                reset_obs = self._reset_reassigned_env_slots(
                    stage_id=stage_id, reassignment=reassignment
                )
                if reset_obs is not None:
                    env_output.obs = reset_obs
                    self._append_continuous_initial_boundary(
                        stage_id=stage_id,
                        slot_indices=reassignment["slot_indices"],
                    )

                if env_info:
                    self.record_env_metrics(env_metrics, env_info)

                next_env_outputs[stage_id] = env_output

            env_outputs = next_env_outputs
            chunk_step_idx += 1
            for stage_id, env_output in enumerate(env_outputs):
                if stopped_stages[stage_id]:
                    continue
                manager = self.slot_state_managers[stage_id]
                if manager.target_real_groups_complete:
                    self._send_train_env_output(
                        rollout_channel,
                        stage_id,
                        env_output,
                        continuous_done=True,
                    )
                    stopped_stages[stage_id] = True
                else:
                    self._send_train_env_output(rollout_channel, stage_id, env_output)

        if not self._continuous_targets_complete():
            raise RuntimeError(
                "continuous_batching did not complete target groups within "
                f"{max_chunk_steps} chunks. Completed per stage: "
                f"{[m.completed_real_group_count for m in self.slot_state_managers]}, "
                f"target={target_groups_per_stage}."
            )
        self.store_last_obs_and_intervened_info(env_outputs)
        self.finish_rollout()

        if actor_channel is not None:
            for stage_id in range(self.stage_num):
                await self.send_rollout_trajectories(
                    self.rollout_results[stage_id], actor_channel
                )
            self.rollout_results = []
            gc.collect()

        for key, value in env_metrics.items():
            env_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return env_metrics

    @Worker.timer("env/send_rollout_trajectories")
    async def send_rollout_trajectories(
        self, rollout_result: EmbodiedRolloutResult, channel: Channel
    ):
        trajectories: list[Trajectory] = rollout_result.to_splited_trajectories(
            self.actor_split_num
        )
        rollout_result.clear()
        for trajectory in trajectories:
            channel.put(trajectory, async_op=True)
        del trajectories
        gc.collect()

    @Worker.timer("run_interact_once")
    async def _run_interact_once(
        self,
        input_channel: Channel,
        rollout_channel: Channel,
        reward_channel: Channel | None,
        actor_channel: Channel | None,
        *,
        cooperative_yield: bool,
    ) -> dict[str, torch.Tensor]:
        if self.continuous_batching_enabled:
            return await self._run_interact_once_continuous(
                input_channel,
                rollout_channel,
                reward_channel,
                actor_channel,
                cooperative_yield=cooperative_yield,
            )

        self.rollout_results = [
            self._new_rollout_collector() for _ in range(self.stage_num)
        ]
        env_metrics = defaultdict(list)
        rlt_pending_obs: list[dict[str, Any] | None] = [None] * self.stage_num

        for epoch in range(self.rollout_epoch):
            if epoch == 0 and self._prefetched_train_bootstrap is not None:
                env_outputs = self._prefetched_train_bootstrap
                self._prefetched_train_bootstrap = None
            else:
                env_outputs = self._bootstrap_and_send_train(rollout_channel)
            self._ensure_slot_state_managers()

            for chunk_step_idx in range(self.n_train_chunk_steps):
                for stage_id in range(self.stage_num):
                    if cooperative_yield:
                        await asyncio.sleep(0)

                    env_output = env_outputs[stage_id]
                    curr_obs = env_output.obs
                    if env_output.intervene_actions is not None:
                        self.rollout_results[stage_id].update_last_actions(
                            env_output.intervene_actions,
                            env_output.intervene_flags,
                        )

                    reward_model_output = None
                    if reward_channel is not None and chunk_step_idx != 0:
                        reward_model_output = self.get_reward_model_output(
                            env_output,
                            send_channel=reward_channel,
                            recv_channel=input_channel,
                            stage_id=stage_id,
                        )
                        if reward_model_output is not None:
                            env_metrics["reward_model_output"].append(
                                reward_model_output.detach().float().reshape(-1).cpu()
                            )

                    rollout_result = self.recv_from(
                        group_name=self.cfg.rollout.group_name,
                        channel=input_channel,
                        tag="train_rollout_results",
                        route_key=stage_id if not self.env_decoupled_mode else None,
                        batch_size=self.train_batch_size,
                        merge_fn=RolloutResult.merge_rollout_results,
                        infer_batch_size_fn=self._infer_rollout_batch_size,
                        decoupled_mode=self.env_decoupled_mode,
                    )
                    rewards = self.compute_bootstrap_rewards(
                        env_output, rollout_result.bootstrap_values, reward_model_output
                    )
                    slot_metadata = {}
                    if self.collect_slot_metadata:
                        slot_metadata = self._build_slot_metadata(
                            rollout_epoch_id=epoch,
                            stage_id=stage_id,
                        )
                    chunk_step_result = ChunkStepResult(
                        actions=rollout_result.forward_inputs.get("action", None),
                        prev_logprobs=(
                            rollout_result.prev_logprobs
                            if self.collect_prev_infos
                            else None
                        ),
                        prev_values=(
                            rollout_result.prev_values
                            if self.collect_prev_infos
                            else None
                        ),
                        forward_inputs=rollout_result.forward_inputs,
                        versions=rollout_result.versions,
                        dones=env_output.dones,
                        truncations=env_output.truncations,
                        terminations=env_output.terminations,
                        rewards=rewards,
                        slot_metadata=slot_metadata,
                    )
                    self.rollout_results[stage_id].append_step_result(chunk_step_result)
                    if (
                        self.reward_mode == "history_buffer"
                        and self.history_reward_assign
                        and reward_model_output is not None
                    ):
                        self.assign_history_reward(stage_id, reward_model_output)
                    if rollout_result.save_flags is not None:
                        self.rollout_results[stage_id].mark_last_step_with_flags(
                            rollout_result.save_flags
                        )
                    if self.enable_rlt and self.collect_transitions:
                        update_rlt_transitions(
                            stage_id,
                            rlt_pending_obs,
                            self.rollout_results,
                            rollout_result,
                            cache_current=True,
                        )

                    env_output, env_info = self.env_interact_step(
                        rollout_result.actions, stage_id
                    )
                    self._profile_early_stop_record_chunk(
                        rollout_epoch_id=epoch,
                        stage_id=stage_id,
                        actions=rollout_result.actions,
                        env_output=env_output,
                    )
                    if self.collect_slot_metadata:
                        newly_done = self._mark_slot_managers_done(
                            stage_id=stage_id, dones=env_output.dones
                        )
                        reassignment = self._reassign_done_slots(
                            stage_id=stage_id, newly_done=newly_done
                        )
                        reset_obs = self._reset_reassigned_env_slots(
                            stage_id=stage_id, reassignment=reassignment
                        )
                        if reset_obs is not None:
                            env_output.obs = reset_obs
                    env_batch = env_output.to_dict()
                    data = {
                        "obs": env_batch["obs"],
                        "final_obs": env_batch["final_obs"],
                    }
                    if self.enable_rlt:
                        data["rlt_switch_flags"] = env_batch.get(
                            "rlt_switch_flags", None
                        )
                    self.send_to(
                        group_name=self.cfg.rollout.group_name,
                        channel=rollout_channel,
                        data=data,
                        mode="train",
                        tag="rollout_results",
                        route_key=stage_id if not self.env_decoupled_mode else None,
                        decoupled_mode=self.env_decoupled_mode,
                    )
                    if self.collect_transitions and not self.enable_rlt:
                        next_obs = (
                            env_output.final_obs
                            if env_output.dones.any() and self.cfg.env.train.auto_reset
                            else env_output.obs
                        )
                        self.rollout_results[stage_id].append_transitions(
                            curr_obs, next_obs
                        )

                    env_outputs[stage_id] = env_output
                    should_record = (
                        self.cfg.env.train.auto_reset
                        or self.cfg.env.train.ignore_terminations
                        or chunk_step_idx == self.n_train_chunk_steps - 1
                    )
                    if should_record:
                        self.record_env_metrics(env_metrics, env_info)
            for stage_id in range(self.stage_num):
                env_output = env_outputs[stage_id]
                if env_output.intervene_actions is not None:
                    self.rollout_results[stage_id].update_last_actions(
                        env_output.intervene_actions,
                        env_output.intervene_flags,
                    )

                reward_model_output = None
                if reward_channel is not None:
                    last_run = epoch == self.rollout_epoch - 1
                    reward_model_output = self.get_reward_model_output(
                        env_output,
                        send_channel=reward_channel,
                        recv_channel=input_channel,
                        stage_id=stage_id,
                        last_run=last_run,
                    )
                    if reward_model_output is not None:
                        env_metrics["reward_model_output"].append(
                            reward_model_output.detach().float().reshape(-1).cpu()
                        )
                rollout_result = self.recv_from(
                    group_name=self.cfg.rollout.group_name,
                    channel=input_channel,
                    tag="train_rollout_results",
                    route_key=stage_id if not self.env_decoupled_mode else None,
                    batch_size=self.train_batch_size,
                    merge_fn=RolloutResult.merge_rollout_results,
                    infer_batch_size_fn=self._infer_rollout_batch_size,
                    decoupled_mode=self.env_decoupled_mode,
                )
                rewards = self.compute_bootstrap_rewards(
                    env_output, rollout_result.bootstrap_values, reward_model_output
                )
                slot_metadata = {}
                if self.collect_slot_metadata:
                    slot_metadata = self._build_slot_metadata(
                        rollout_epoch_id=epoch,
                        stage_id=stage_id,
                    )
                chunk_step_result = ChunkStepResult(
                    prev_values=(
                        rollout_result.prev_values if self.collect_prev_infos else None
                    ),
                    dones=env_output.dones,
                    truncations=env_output.truncations,
                    terminations=env_output.terminations,
                    rewards=rewards,
                    slot_metadata=slot_metadata,
                )
                self.rollout_results[stage_id].append_step_result(chunk_step_result)
                if (
                    self.reward_mode == "history_buffer"
                    and self.history_reward_assign
                    and reward_model_output is not None
                ):
                    self.assign_history_reward(stage_id, reward_model_output)
                if self.enable_rlt and self.collect_transitions:
                    update_rlt_transitions(
                        stage_id,
                        rlt_pending_obs,
                        self.rollout_results,
                        rollout_result,
                        cache_current=False,
                    )

            if self.use_training_pipeline and actor_channel is not None:
                await self.send_rollout_trajectories_pipeline(
                    self.rollout_results, actor_channel
                )
                self.rollout_results = [
                    self._new_rollout_collector() for _ in range(self.stage_num)
                ]

            self.store_last_obs_and_intervened_info(env_outputs)
            self.finish_rollout()

        if not self.use_training_pipeline and actor_channel is not None:
            for stage_id in range(self.stage_num):
                await self.send_rollout_trajectories(
                    self.rollout_results[stage_id], actor_channel
                )
            # reduce memory peak
            self.rollout_results = []
            gc.collect()

        for key, value in env_metrics.items():
            env_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        if self._profile_early_stop_active():
            self._profile_early_stop_step_id += 1

        return env_metrics

    @Worker.timer("interact")
    async def interact(
        self,
        input_channel: Channel,
        rollout_channel: Channel,
        reward_channel: Channel | None,
        actor_channel: Channel | None = None,
    ):
        env_metrics = await self._run_interact_once(
            input_channel,
            rollout_channel,
            reward_channel,
            actor_channel,
            cooperative_yield=False,
        )

        for env in self.env_list:
            if self.train_enable_offload:
                get_env_attr(env, "offload")()

        return env_metrics

    def evaluate(self, input_channel: Channel, rollout_channel: Channel):
        eval_metrics = defaultdict(list)
        for eval_rollout_epoch in range(self.eval_rollout_epoch):
            if not self.cfg.env.eval.auto_reset or eval_rollout_epoch == 0:
                for stage_id in range(self.stage_num):
                    self.eval_env_list[stage_id].is_start = True
                    self.eval_prev_done[stage_id] = torch.zeros(
                        self.eval_num_envs_per_stage, dtype=torch.bool
                    )
                    extracted_obs, infos = self.eval_env_list[stage_id].reset()
                    env_output = EnvOutput(
                        obs=extracted_obs,
                        final_obs=(
                            infos["final_observation"]
                            if "final_observation" in infos
                            else None
                        ),
                    )
                    env_batch = env_output.to_dict()
                    data = {
                        "obs": env_batch["obs"],
                        "final_obs": env_batch["final_obs"],
                    }
                    if self.enable_rlt:
                        data["rlt_switch_flags"] = env_batch.get(
                            "rlt_switch_flags", None
                        )
                    self.send_to(
                        group_name=self.cfg.rollout.group_name,
                        channel=rollout_channel,
                        data=data,
                        mode="eval",
                        tag="rollout_results",
                        route_key=stage_id if not self.env_decoupled_mode else None,
                        decoupled_mode=self.env_decoupled_mode,
                    )

            for eval_step in range(self.n_eval_chunk_steps):
                for stage_id in range(self.stage_num):
                    rollout_results = self.recv_from(
                        group_name=self.cfg.rollout.group_name,
                        channel=input_channel,
                        tag="eval_rollout_results",
                        route_key=stage_id if not self.env_decoupled_mode else None,
                        batch_size=self.eval_batch_size,
                        infer_batch_size_fn=self._infer_rollout_batch_size
                        if self.env_decoupled_mode
                        else None,
                        decoupled_mode=self.env_decoupled_mode,
                    )
                    raw_chunk_actions = (
                        rollout_results.actions
                        if hasattr(rollout_results, "actions")
                        else rollout_results
                    )
                    if isinstance(raw_chunk_actions, torch.Tensor):
                        raw_chunk_actions = raw_chunk_actions.detach().cpu().numpy()
                    else:
                        raw_chunk_actions = np.asarray(raw_chunk_actions)
                    env_output, env_info = self.env_evaluate_step(
                        raw_chunk_actions, stage_id
                    )

                    for key, value in env_info.items():
                        eval_metrics[key].append(value)

                    if self.cfg.env.eval.auto_reset:
                        if (
                            eval_rollout_epoch == self.eval_rollout_epoch - 1
                            and eval_step == self.n_eval_chunk_steps - 1
                        ):
                            continue
                    else:
                        if eval_step == self.n_eval_chunk_steps - 1:
                            continue
                    env_batch = env_output.to_dict()
                    data = {
                        "obs": env_batch["obs"],
                        "final_obs": env_batch["final_obs"],
                    }
                    if self.enable_rlt:
                        data["rlt_switch_flags"] = env_batch.get(
                            "rlt_switch_flags", None
                        )
                    self.send_to(
                        group_name=self.cfg.rollout.group_name,
                        channel=rollout_channel,
                        data=data,
                        mode="eval",
                        tag="rollout_results",
                        route_key=stage_id if not self.env_decoupled_mode else None,
                        decoupled_mode=self.env_decoupled_mode,
                    )

            self.finish_rollout(mode="eval")
        for stage_id in range(self.stage_num):
            if self.eval_enable_offload:
                get_env_attr(self.eval_env_list[stage_id], "offload")()

        for key, value in eval_metrics.items():
            eval_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return eval_metrics

    def get_actor_split_num(self):
        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(recv_num, send_num)
        return split_num

    def compute_advantages_and_returns(
        self, rollout_batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # Advantages/returns are rollout-level quantities, so compute them before
        # splitting. After this point each channel item is an actor micro-batch that can
        # be trained directly without reconstructing the full rollout batch on actor.
        kwargs = {
            "task_type": self.cfg.runner.task_type,
            "adv_type": self.cfg.algorithm.adv_type,
            "rewards": rollout_batch["rewards"],
            "dones": rollout_batch["dones"],
            "values": rollout_batch.get("prev_values", None),
            "gamma": self.cfg.algorithm.get("gamma", 1),
            "gae_lambda": self.cfg.algorithm.get("gae_lambda", 1),
            "group_size": self.cfg.algorithm.get("group_size", 8),
            "reward_type": self.cfg.algorithm.reward_type,
            "loss_mask": rollout_batch.get("loss_mask", None),
            "loss_mask_sum": rollout_batch.get("loss_mask_sum", None),
            "normalize_advantages": self.cfg.algorithm.get("normalize_advantages", True)
            and not self.use_training_pipeline,
        }
        advantages_and_returns = calculate_adv_and_returns(**kwargs)
        rollout_batch.update(advantages_and_returns)
        if kwargs["loss_mask"] is not None:
            rollout_batch["loss_mask"] = kwargs["loss_mask"]
        if kwargs["loss_mask_sum"] is not None:
            rollout_batch["loss_mask_sum"] = kwargs["loss_mask_sum"]
        return rollout_batch

    def prepare_pipeline_batch(self, trajectory: Trajectory) -> dict[str, torch.Tensor]:
        batch = convert_trajectories_to_batch([trajectory])
        batch = preprocess_embodied_batch(
            batch,
            rollout_epoch=1,
            auto_reset=self.cfg.env.train.auto_reset,
            ignore_terminations=self.cfg.env.train.ignore_terminations,
            reward_type=self.cfg.algorithm.reward_type,
            filter_rewards=self.cfg.algorithm.get("filter_rewards", False),
            group_size=self.cfg.algorithm.group_size,
            rewards_lower_bound=self.cfg.algorithm.get("rewards_lower_bound", None),
            rewards_upper_bound=self.cfg.algorithm.get("rewards_upper_bound", None),
        )
        return self.compute_advantages_and_returns(batch)

    def pack_pipeline_micro_batches(
        self, batch: dict[str, torch.Tensor], actor_rank: int
    ) -> list[dict]:
        batch_size = batch["prev_logprobs"].shape[0] * batch["prev_logprobs"].shape[1]
        if self.shuffle_rollout:
            shuffle_id = torch.randperm(
                batch_size, generator=self.shuffle_generators[actor_rank]
            )
        else:
            shuffle_id = torch.arange(batch_size)

        flatten_batch = flatten_embodied_batch(batch, shuffle_id)
        micro_batch_size = self.cfg.actor.micro_batch_size
        assert batch_size % micro_batch_size == 0, (
            f"Batch size {batch_size} is not divisible by micro_batch_size {micro_batch_size}."
        )
        num_micro_batches = batch_size // micro_batch_size
        micro_batches = split_dict_to_chunk(flatten_batch, num_micro_batches, dim=0)
        return [pack_batch(micro_batch) for micro_batch in micro_batches]

    async def send_rollout_trajectories_pipeline(
        self,
        rollout_results: list[EmbodiedRolloutResult],
        channel: Channel,
    ) -> None:
        pending_batches: list[tuple[int, dict[str, torch.Tensor]]] = []
        batches_by_actor_rank: dict[int, list[dict[str, torch.Tensor]]] = defaultdict(
            list
        )

        with self.worker_timer("prepare_micro_batches"):
            for stage_id, rollout_result in enumerate(rollout_results):
                actor_splits = self.pipeline_stage_actor_splits[stage_id]
                trajectories = rollout_result.to_splited_trajectories_by_sizes(
                    [split_size for _, split_size in actor_splits]
                )

                for (actor_rank, _), trajectory in zip(actor_splits, trajectories):
                    batch = self.prepare_pipeline_batch(trajectory)
                    pending_batches.append((actor_rank, batch))
                    batches_by_actor_rank[actor_rank].append(batch)

            if self.cfg.algorithm.get("normalize_advantages", True):
                for actor_rank, batches in sorted(batches_by_actor_rank.items()):
                    local_adv_stats = sum(
                        masked_stats(batch["advantages"], batch.get("loss_mask"))
                        for batch in batches
                    )
                    env_ranks = self.pipeline_actor_env_ranks[actor_rank]
                    global_adv_stats = sum(
                        self.broadcast(
                            local_adv_stats if self._rank == src_rank else None,
                            groups=[(self._group_name, env_ranks)],
                            src=(self._group_name, src_rank),
                        )
                        for src_rank in env_ranks
                    )
                    for batch in batches:
                        batch["advantages"] = normalize_from_stats(
                            batch["advantages"], global_adv_stats
                        )

            for actor_rank, batch in pending_batches:
                for micro_batch in self.pack_pipeline_micro_batches(batch, actor_rank):
                    channel.put(
                        micro_batch,
                        key=self.pipeline_actor_keys[actor_rank],
                        async_op=True,
                    )
