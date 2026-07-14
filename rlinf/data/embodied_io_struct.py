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

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    pass

from rlinf.utils.nested_dict_process import (
    cat_list_of_dict_tensor,
    put_tensor_device,
    split_dict,
    split_dict_to_chunk,
    stack_list_of_dict_tensor,
)


def get_model_weights_id(versions: torch.Tensor) -> str:
    """
    Get the model weights id from the tensor.

    Args:
        versions (torch.Tensor): The tensor to get the model weights id from.

    Returns:
        str: The model weights id.
    """

    name_bytes = versions.cpu().numpy().tobytes()
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, name_bytes.hex()))


@dataclass(kw_only=True)
class EnvOutput:
    """Environment output for a single chunk step."""

    obs: dict[str, Any]
    final_obs: Optional[dict[str, Any]] = None
    dones: Optional[torch.Tensor] = None  # [B]
    terminations: Optional[torch.Tensor] = None  # [B]
    truncations: Optional[torch.Tensor] = None  # [B]
    rewards: Optional[torch.Tensor] = None  # [B]
    env_infos: Optional[dict[str, Any]] = None

    intervene_actions: Optional[torch.Tensor] = None  # [B]
    intervene_flags: Optional[torch.Tensor] = None  # [B]
    rlt_switch_flags: Optional[torch.Tensor] = None  # [B] or [B, action_chunk]

    def __post_init__(self):
        self.obs = put_tensor_device(self.obs, "cpu")
        self.final_obs = (
            put_tensor_device(self.final_obs, "cpu")
            if self.final_obs is not None
            else None
        )
        self.dones = self.dones.cpu().contiguous() if self.dones is not None else None
        self.terminations = (
            self.terminations.cpu().contiguous()
            if self.terminations is not None
            else None
        )
        self.truncations = (
            self.truncations.cpu().contiguous()
            if self.truncations is not None
            else None
        )
        self.rewards = (
            self.rewards.cpu().contiguous() if self.rewards is not None else None
        )
        self.env_infos = (
            put_tensor_device(self.env_infos, "cpu")
            if self.env_infos is not None
            else None
        )
        self.intervene_actions = (
            self.intervene_actions.cpu().contiguous()
            if self.intervene_actions is not None
            else None
        )
        self.intervene_flags = (
            self.intervene_flags.cpu().contiguous()
            if self.intervene_flags is not None
            else None
        )
        self.rlt_switch_flags = (
            self.rlt_switch_flags.cpu().contiguous()
            if self.rlt_switch_flags is not None
            else None
        )

    def prepare_observations(self, obs: dict[str, Any]) -> dict[str, Any]:
        image_tensor = obs["main_images"] if "main_images" in obs else None
        wrist_image_tensor = obs["wrist_images"] if "wrist_images" in obs else None
        extra_view_image_tensor = (
            obs["extra_view_images"] if "extra_view_images" in obs else None
        )
        states = obs["states"] if "states" in obs else None
        task_descriptions = (
            list(obs["task_descriptions"])
            if "task_descriptions" in obs and obs["task_descriptions"] is not None
            else None
        )

        return {
            "main_images": image_tensor,  # [N_ENV, H, W, C]
            "wrist_images": wrist_image_tensor,  # [N_ENV, H, W, C] or [N_ENV, N_IMG, H, W, C]
            "extra_view_images": extra_view_image_tensor,  # [N_ENV, N_IMG, H, W, C]
            "states": states,
            "task_descriptions": task_descriptions,
        }

    @staticmethod
    def merge_env_outputs(env_outputs: list[dict]) -> dict[str, Any]:
        """Merge multiple env output dicts into one batch-aligned env output.

        Merge strategy:

        - Tensor fields: concatenate on batch dimension.
        - List fields: flatten in source order.
        - ``None`` fields: keep ``None``.
        - ``final_obs`` supports partial ``None`` across shards. For shards
            without ``final_obs``, use the corresponding ``obs`` as fallback to
            keep batch alignment.

        Args:
            env_outputs: Per-source env output dicts that share the same schema.

        Returns:
            A merged env output dict produced via ``EnvOutput(...).to_dict()``.
        """

        def _get_batch_size(env_output: dict[str, Any]) -> int:
            dones = env_output.get("dones")
            if isinstance(dones, torch.Tensor):
                return dones.shape[0]

            obs = env_output["obs"]
            for key in ("states", "main_images", "task_descriptions"):
                value = obs.get(key)
                if isinstance(value, torch.Tensor):
                    return value.shape[0]
                if isinstance(value, list):
                    return len(value)
            raise ValueError("Cannot infer batch size from env output.")

        def _merge_obs_dicts(obs_dicts: list[dict[str, Any]]) -> dict[str, Any]:
            merged_obs = {}
            for key in obs_dicts[0].keys():
                obs_elements = [obs_dict[key] for obs_dict in obs_dicts]
                first_non_none = next(
                    (element for element in obs_elements if element is not None), None
                )
                if first_non_none is None:
                    merged_obs[key] = None
                elif isinstance(first_non_none, torch.Tensor):
                    merged_obs[key] = torch.cat(obs_elements, dim=0)
                elif isinstance(first_non_none, list):
                    merged_obs[key] = [
                        item for sublist in obs_elements for item in sublist
                    ]
                else:
                    merged_obs[key] = obs_elements
            return merged_obs

        def _merge_optional_tensor_field(
            field_name: str,
            *,
            allow_partial_none: bool = False,
            fill_value: float | bool = 0,
        ) -> torch.Tensor | None:
            values = [env_output[field_name] for env_output in env_outputs]
            if all(value is None for value in values):
                return None

            if any(value is None for value in values):
                if not allow_partial_none:
                    raise ValueError(
                        f"Inconsistent field '{field_name}': some shards are None while others are tensors."
                    )

                ref_tensor = next(value for value in values if value is not None)
                filled_values = []
                for env_output, value in zip(env_outputs, values):
                    if value is None:
                        batch_size = _get_batch_size(env_output)
                        fill_shape = (batch_size, *ref_tensor.shape[1:])
                        filled_values.append(
                            torch.full(
                                fill_shape,
                                fill_value=fill_value,
                                dtype=ref_tensor.dtype,
                            )
                        )
                    else:
                        filled_values.append(value)
                values = filled_values

            return torch.cat(values, dim=0)

        merged_obs = _merge_obs_dicts([env_output["obs"] for env_output in env_outputs])

        merged_final_obs = None
        final_obs_list = [env_output["final_obs"] for env_output in env_outputs]
        if any(final_obs is not None for final_obs in final_obs_list):
            # Some shards may not have done episodes in this step, so their final_obs
            # is None. Use obs as fallback to keep merged batch shape aligned.
            final_obs_or_obs = [
                final_obs if final_obs is not None else env_output["obs"]
                for env_output, final_obs in zip(env_outputs, final_obs_list)
            ]
            merged_final_obs = _merge_obs_dicts(final_obs_or_obs)

        merged_dones = _merge_optional_tensor_field("dones")
        merged_terminations = _merge_optional_tensor_field("terminations")
        merged_truncations = _merge_optional_tensor_field("truncations")
        merged_rewards = _merge_optional_tensor_field("rewards")
        merged_intervene_actions = _merge_optional_tensor_field(
            "intervene_actions",
            allow_partial_none=True,
            fill_value=0.0,
        )
        merged_intervene_flags = _merge_optional_tensor_field(
            "intervene_flags",
            allow_partial_none=True,
            fill_value=False,
        )
        merged_rlt_switch_flags = _merge_optional_tensor_field(
            "rlt_switch_flags",
            allow_partial_none=True,
            fill_value=False,
        )
        # turn to EnvOutput and turn to dict to call post init for tensor processing
        return EnvOutput(
            obs=merged_obs,
            final_obs=merged_final_obs,
            dones=merged_dones,
            terminations=merged_terminations,
            truncations=merged_truncations,
            rewards=merged_rewards,
            intervene_actions=merged_intervene_actions,
            intervene_flags=merged_intervene_flags,
            rlt_switch_flags=merged_rlt_switch_flags,
        ).to_dict()

    def to_dict(self) -> dict[str, Any]:
        env_output_dict = {}

        env_output_dict["obs"] = self.prepare_observations(self.obs)
        env_output_dict["final_obs"] = (
            self.prepare_observations(self.final_obs)
            if self.final_obs is not None
            else None
        )
        env_output_dict["dones"] = self.dones
        env_output_dict["terminations"] = self.terminations
        env_output_dict["truncations"] = self.truncations
        env_output_dict["rewards"] = self.rewards
        env_output_dict["env_infos"] = self.env_infos
        env_output_dict["intervene_actions"] = self.intervene_actions
        env_output_dict["intervene_flags"] = self.intervene_flags
        env_output_dict["rlt_switch_flags"] = self.rlt_switch_flags

        return env_output_dict


@dataclass(kw_only=True)
class RolloutResult:
    """Rollout result for a single chunk step."""

    actions: torch.Tensor = None  # [B, action_dim]
    prev_logprobs: torch.Tensor = None  # [B, action_dim]
    prev_values: torch.Tensor = None  # [B, 1]

    bootstrap_values: torch.Tensor = None  # [B, 1]
    save_flags: torch.Tensor = None  # [B, num_action_chunks]
    forward_inputs: dict[str, torch.Tensor] = field(default_factory=dict)
    versions: torch.Tensor = None  # [B, 1]

    def __post_init__(self):
        if self.actions is not None:
            self.actions = self.actions.cpu().contiguous()
        if self.prev_logprobs is not None:
            self.prev_logprobs = self.prev_logprobs.cpu().contiguous()
        if self.prev_values is not None:
            self.prev_values = self.prev_values.cpu().contiguous()
        if self.bootstrap_values is not None:
            self.bootstrap_values = self.bootstrap_values.cpu().contiguous()
        if self.save_flags is not None:
            self.save_flags = self.save_flags.cpu().contiguous()
        if self.forward_inputs:
            self.forward_inputs = put_tensor_device(self.forward_inputs, "cpu")
        if self.versions is not None:
            self.versions = self.versions.cpu().contiguous()

    @staticmethod
    def merge_rollout_results(
        rollout_results: list["RolloutResult"],
    ) -> "RolloutResult":
        def _merge_optional_tensor(field_name: str) -> torch.Tensor | None:
            values = [
                getattr(rollout_result, field_name)
                for rollout_result in rollout_results
            ]
            if all(value is None for value in values):
                return None
            if any(value is None for value in values):
                raise ValueError(
                    f"Inconsistent field '{field_name}': some shards are None while others are tensors."
                )
            return torch.cat(values, dim=0)

        merged_actions = _merge_optional_tensor("actions")
        merged_prev_logprobs = _merge_optional_tensor("prev_logprobs")
        merged_prev_values = _merge_optional_tensor("prev_values")
        merged_bootstrap_values = _merge_optional_tensor("bootstrap_values")
        merged_save_flags = _merge_optional_tensor("save_flags")
        merged_versions = _merge_optional_tensor("versions")

        forward_inputs_list = [
            rollout_result.forward_inputs for rollout_result in rollout_results
        ]
        if all(not forward_inputs for forward_inputs in forward_inputs_list):
            merged_forward_inputs = {}
        else:
            merged_forward_inputs = cat_list_of_dict_tensor(forward_inputs_list)
        return RolloutResult(
            actions=merged_actions,
            prev_logprobs=merged_prev_logprobs,
            prev_values=merged_prev_values,
            bootstrap_values=merged_bootstrap_values,
            save_flags=merged_save_flags,
            forward_inputs=merged_forward_inputs,
            versions=merged_versions,
        )


@dataclass(kw_only=True)
class ChunkStepResult:
    """Model outputs, env outputs (without observations), and training forward inputs for a chunk step."""

    actions: torch.Tensor = None  # [B, action_dim]
    prev_logprobs: torch.Tensor = None  # [B, action_dim]
    prev_values: torch.Tensor = None  # [B, 1]
    dones: torch.Tensor = None  # [B, 1]
    truncations: torch.Tensor = None  # [B, 1]
    terminations: torch.Tensor = None  # [B, 1]
    rewards: torch.Tensor = None  # [B, 1]
    forward_inputs: dict[str, torch.Tensor] = field(default_factory=dict)
    versions: torch.Tensor = None  # [B, 1]
    slot_metadata: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self):
        if self.actions is not None:
            self.actions = self.actions.cpu().contiguous()
        if self.prev_logprobs is not None:
            self.prev_logprobs = self.prev_logprobs.cpu().contiguous()
        if self.prev_values is not None:
            self.prev_values = self.prev_values.cpu().contiguous()
        if self.dones is not None:
            self.dones = self.dones.cpu().contiguous()
        if self.terminations is not None:
            self.terminations = self.terminations.cpu().contiguous()
        if self.truncations is not None:
            self.truncations = self.truncations.cpu().contiguous()
        if self.rewards is not None:
            self.rewards = self.rewards.cpu().contiguous()
        if self.slot_metadata:
            self.slot_metadata = put_tensor_device(self.slot_metadata, "cpu")
        if self.forward_inputs:
            self.forward_inputs = put_tensor_device(self.forward_inputs, "cpu")
        if self.versions is not None:
            self.versions = self.versions.cpu().contiguous()


@dataclass
class Trajectory:
    """
    trajectory contains multiple episodes.
    """

    max_episode_length: int = 0  # max episode length
    model_weights_id: str = ""  # str(uuid(versions))
    actions: torch.Tensor = None
    intervene_flags: torch.Tensor = None
    rewards: torch.Tensor = None
    terminations: torch.Tensor = None
    truncations: torch.Tensor = None
    dones: torch.Tensor = None
    prev_logprobs: torch.Tensor = None
    prev_values: torch.Tensor = None
    versions: torch.Tensor = None
    forward_inputs: dict[str, Any] = field(default_factory=dict)
    slot_metadata: dict[str, Any] = field(default_factory=dict)

    curr_obs: dict[str, Any] = field(default_factory=dict)
    next_obs: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _generate_field_mask(
        ref_tensor: torch.Tensor, mask: torch.Tensor, traj_len: int
    ) -> torch.Tensor:
        """
        Generate a mask for terminations/truncations/dones based on their original shape.
        """
        assert mask.dim() == 1, f"Expected 1D mask, got {mask.shape=}"
        if ref_tensor.shape[0] == traj_len:
            return mask
        elif ref_tensor.shape[0] > traj_len:
            extra = int(ref_tensor.shape[0] - traj_len)
            assert traj_len % extra == 0, (
                f"Trajectory length {traj_len} is not divisible by extra {extra} for terminations/truncations/dones"
            )
            epoch_len = traj_len // extra

            field_mask = torch.zeros(
                ref_tensor.shape[0], dtype=torch.bool, device=mask.device
            )
            original_indices = torch.arange(ref_tensor.shape[0], device=mask.device)
            epoch_idx = original_indices // (epoch_len + 1)
            step_idx = original_indices % (epoch_len + 1)

            # Keep the first position of each epoch (step_idx == 0)
            field_mask[step_idx == 0] = True

            # Map positions with step_idx >= 1 to mask
            valid_mask = step_idx >= 1
            mask_idx = epoch_idx[valid_mask] * epoch_len + (step_idx[valid_mask] - 1)
            valid_original_indices = original_indices[valid_mask]
            valid_mask_idx = mask_idx < len(mask)
            field_mask[valid_original_indices[valid_mask_idx]] = mask[
                mask_idx[valid_mask_idx]
            ].to(dtype=torch.bool)

            return field_mask
        else:
            raise ValueError(
                f"Reference tensor length {ref_tensor.shape[0]} < traj_len {traj_len}"
            )

    def extract_intervene_traj(self, mode="any"):
        if self.intervene_flags is None or (~self.intervene_flags).all():
            return None

        if mode == "any":
            mask = self.intervene_flags.any(dim=-1)
        elif mode == "all":
            mask = self.intervene_flags.all(dim=-1)
        else:
            raise NotImplementedError(
                f"Unsupported extract_intervene_traj mode: {mode}"
            )
        assert mask.dim() == 2, (
            f"Expected 2D mask after processing (traj len, bsz), got {mask.shape=}"
        )
        traj_len = int(mask.shape[0])

        def apply_mask(tensor, i):
            return tensor[:, i][mask[:, i]].unsqueeze(1) if tensor is not None else None

        def apply_mask_to_dict(d, i):
            return (
                {k: v[:, i][mask[:, i]].unsqueeze(1) for k, v in d.items()} if d else {}
            )

        filtered_trajectories = []
        for i in range(mask.shape[1]):
            if not mask[:, i].any():
                continue

            actions = apply_mask(self.actions, i)
            rewards = apply_mask(self.rewards, i)
            prev_logprobs = apply_mask(self.prev_logprobs, i)
            prev_values = apply_mask(self.prev_values, i)
            intervene_flags = apply_mask(self.intervene_flags, i)

            forward_inputs = apply_mask_to_dict(self.forward_inputs, i)
            slot_metadata = apply_mask_to_dict(self.slot_metadata, i)
            curr_obs = apply_mask_to_dict(self.curr_obs, i)
            next_obs = apply_mask_to_dict(self.next_obs, i)

            terminations = truncations = dones = None
            if self.terminations is not None:
                field_mask = self._generate_field_mask(
                    self.terminations[:, i : i + 1], mask[:, i], traj_len
                )
                terminations = self.terminations[:, i : i + 1][field_mask]
                truncations = self.truncations[:, i : i + 1][field_mask]
                dones = self.dones[:, i : i + 1][field_mask]

            filtered_trajectories.append(
                Trajectory(
                    max_episode_length=self.max_episode_length,
                    model_weights_id=self.model_weights_id,
                    actions=actions,
                    intervene_flags=intervene_flags,
                    rewards=rewards,
                    terminations=terminations,
                    truncations=truncations,
                    dones=dones,
                    prev_logprobs=prev_logprobs,
                    prev_values=prev_values,
                    forward_inputs=forward_inputs,
                    slot_metadata=slot_metadata,
                    curr_obs=curr_obs,
                    next_obs=next_obs,
                )
            )

        return filtered_trajectories if filtered_trajectories else None


@dataclass(kw_only=True)
class EmbodiedRolloutResult:
    """
    Collect chunk-step results and transitions during rollout,
    and convert them into trajectory tensors.
    """

    max_episode_length: int = 0

    actions: list[torch.Tensor] = field(default_factory=list)  # trajectory_length
    intervene_flags: list[torch.Tensor] = field(
        default_factory=list
    )  # trajectory_length
    rewards: list[torch.Tensor] = field(default_factory=list)  # trajectory_length
    terminations: list[torch.Tensor] = field(
        default_factory=list
    )  # trajectory_length + rollout_epoch
    truncations: list[torch.Tensor] = field(
        default_factory=list
    )  # trajectory_length + rollout_epoch
    dones: list[torch.Tensor] = field(
        default_factory=list
    )  # trajectory_length + rollout_epoch
    prev_logprobs: list[torch.Tensor] = field(default_factory=list)  # trajectory_length
    prev_values: list[torch.Tensor] = field(
        default_factory=list
    )  # trajectory_length + rollout_epoch
    versions: list[torch.Tensor] = field(default_factory=list)  # trajectory_length
    forward_inputs: list[dict[str, Any]] = field(
        default_factory=list
    )  # trajectory_length
    slot_metadata: list[dict[str, torch.Tensor]] = field(
        default_factory=list
    )  # trajectory_length

    curr_obs: list[dict[str, Any]] = field(default_factory=list)  # trajectory_length
    next_obs: list[dict[str, Any]] = field(default_factory=list)  # trajectory_length

    def append_step_result(self, result: ChunkStepResult):
        if result.actions is not None:
            self.actions.append(result.actions)
            self.intervene_flags.append(
                torch.zeros_like(result.actions, dtype=torch.bool)
            )
        if result.rewards is not None:
            self.rewards.append(result.rewards)
        if result.terminations is not None:
            self.terminations.append(result.terminations)
        if result.truncations is not None:
            self.truncations.append(result.truncations)
        if result.dones is not None:
            self.dones.append(result.dones)
        if result.prev_logprobs is not None:
            self.prev_logprobs.append(result.prev_logprobs)
        if result.prev_values is not None:
            self.prev_values.append(result.prev_values)
        if result.versions is not None:
            self.versions.append(result.versions)
        if result.forward_inputs:
            self.forward_inputs.append(result.forward_inputs)
        if result.slot_metadata:
            self.slot_metadata.append(result.slot_metadata)

    def mark_last_step_with_flags(self, save_flags: torch.Tensor):
        if not self.intervene_flags:
            return

        if save_flags.dim() == 1:
            save_flags = save_flags[:, None]
        assert save_flags.dim() == 2, f"Expected 2D tensor, got {save_flags.shape=}"

        last_action = self.actions[-1]
        bsz, num_action_chunks = save_flags.shape
        expanded_flags = save_flags.reshape(bsz, num_action_chunks, 1).expand_as(
            last_action.reshape(bsz, num_action_chunks, -1)
        )
        self.intervene_flags[-1] = expanded_flags.reshape(bsz, -1).to(torch.bool)

    def update_last_actions(
        self, intervene_actions: torch.Tensor, intervene_flags: torch.Tensor
    ):
        # action: [bsz, num-chunk-size x action-dim]
        # intervene_actions: [bsz, num-chunk-size x action-dim]
        # intervene_flags: [bsz, num-chunk-size]

        if self.actions and len(self.actions) > 0:
            last_action = self.actions[-1]
            assert last_action.dim() == 2, (
                f"Expected 2D tensor, got {last_action.shape=}"
            )
            assert intervene_actions.dim() == 2, (
                f"Expected 2D tensor, got {intervene_actions.shape=}"
            )

            # Normalize intervene_flags dimensions
            if intervene_flags.dim() == 1:
                intervene_flags = intervene_flags[:, None]
            assert intervene_flags.dim() == 2, (
                f"Expected 2D tensor, got {intervene_flags.shape=}"
            )

            bsz, num_action_chunks = intervene_flags.shape[:2]
            flags = intervene_flags.reshape(-1, num_action_chunks, 1)

            # Combine intervene_actions and last_action based on flags
            last_full_action = intervene_actions.reshape(
                bsz, num_action_chunks, -1
            ) * flags + last_action.reshape(bsz, num_action_chunks, -1) * (~flags)
            self.actions[-1] = last_full_action.reshape(bsz, -1)

            full_flags = flags.expand_as(last_full_action).reshape(bsz, -1)
            self.intervene_flags[-1] = full_flags

            if self.forward_inputs:
                last_fi = self.forward_inputs[-1]
                if "action" in last_fi:
                    last_fi["action"] = (
                        last_full_action.reshape(bsz, -1).cpu().contiguous()
                    )
                last_fi.pop("model_action", None)

    def append_transitions(self, curr_obs=None, next_obs=None):
        assert curr_obs is not None and next_obs is not None
        if "task_descriptions" in curr_obs:
            curr_obs.pop("task_descriptions")
        if "task_descriptions" in next_obs:
            next_obs.pop("task_descriptions")
        self.curr_obs.append(curr_obs)
        self.next_obs.append(next_obs)

    def clear(self):
        self.actions.clear()
        self.intervene_flags.clear()
        self.rewards.clear()
        self.terminations.clear()
        self.truncations.clear()
        self.dones.clear()
        self.prev_logprobs.clear()
        self.prev_values.clear()
        self.versions.clear()
        self.forward_inputs.clear()
        self.slot_metadata.clear()
        self.curr_obs.clear()
        self.next_obs.clear()

    def to_trajectory(self) -> Trajectory:
        # return [trajectory_length, B, ...]
        trajectory = Trajectory(
            max_episode_length=self.max_episode_length,
        )
        if len(self.actions) > 0:
            trajectory.actions = torch.stack(self.actions, dim=0).cpu().contiguous()
        if len(self.intervene_flags) > 0:
            trajectory.intervene_flags = (
                torch.stack(self.intervene_flags, dim=0).cpu().contiguous()
            )
        if len(self.rewards) > 0:
            trajectory.rewards = torch.stack(self.rewards, dim=0).cpu().contiguous()
        if len(self.terminations) > 0:
            trajectory.terminations = (
                torch.stack(self.terminations, dim=0).cpu().contiguous()
            )
        if len(self.truncations) > 0:
            trajectory.truncations = (
                torch.stack(self.truncations, dim=0).cpu().contiguous()
            )
        if len(self.dones) > 0:
            trajectory.dones = torch.stack(self.dones, dim=0).cpu().contiguous()
        if len(self.prev_logprobs) > 0:
            trajectory.prev_logprobs = (
                torch.stack(self.prev_logprobs, dim=0).cpu().contiguous()
            )
        if len(self.prev_values) > 0:
            trajectory.prev_values = (
                torch.stack(self.prev_values, dim=0).cpu().contiguous()
            )
        if len(self.versions) > 0:
            trajectory.versions = torch.stack(self.versions, dim=0).cpu().contiguous()
        if len(self.forward_inputs) > 0:
            trajectory.forward_inputs = stack_list_of_dict_tensor(self.forward_inputs)
            for key in trajectory.forward_inputs.keys():
                trajectory.forward_inputs[key] = (
                    trajectory.forward_inputs[key].cpu().contiguous()
                )
        if len(self.slot_metadata) > 0:
            trajectory.slot_metadata = stack_list_of_dict_tensor(self.slot_metadata)
            for key in trajectory.slot_metadata.keys():
                trajectory.slot_metadata[key] = (
                    trajectory.slot_metadata[key].cpu().contiguous()
                )

        if len(self.curr_obs) > 0:
            trajectory.curr_obs = stack_list_of_dict_tensor(self.curr_obs)
            for key in trajectory.curr_obs.keys():
                trajectory.curr_obs[key] = trajectory.curr_obs[key].cpu().contiguous()
        if len(self.next_obs) > 0:
            trajectory.next_obs = stack_list_of_dict_tensor(self.next_obs)
            for key in trajectory.next_obs.keys():
                trajectory.next_obs[key] = trajectory.next_obs[key].cpu().contiguous()

        trajectory.model_weights_id = get_model_weights_id(
            trajectory.versions
            if trajectory.versions is not None
            else torch.zeros(1, dtype=torch.float32)
        )

        return trajectory

    def to_splited_trajectories(self, split_size: int) -> list[Trajectory]:
        all_trajectory: Trajectory = self.to_trajectory()
        splited_trajectories: list[Trajectory] = [
            Trajectory() for _ in range(split_size)
        ]

        if len(all_trajectory.curr_obs) > 0:
            splited_obs = split_dict_to_chunk(
                all_trajectory.curr_obs, split_size, dim=1
            )
            for i in range(split_size):
                splited_trajectories[i].curr_obs = splited_obs[i]
        if len(all_trajectory.next_obs) > 0:
            splited_obs = split_dict_to_chunk(
                all_trajectory.next_obs, split_size, dim=1
            )
            for i in range(split_size):
                splited_trajectories[i].next_obs = splited_obs[i]

        if (
            all_trajectory.forward_inputs is not None
            and len(all_trajectory.forward_inputs) > 0
        ):
            splited_forward_inputs = split_dict_to_chunk(
                all_trajectory.forward_inputs, split_size, dim=1
            )
            for i in range(split_size):
                splited_trajectories[i].forward_inputs = splited_forward_inputs[i]

        if (
            all_trajectory.slot_metadata is not None
            and len(all_trajectory.slot_metadata) > 0
        ):
            splited_slot_metadata = split_dict_to_chunk(
                all_trajectory.slot_metadata, split_size, dim=1
            )
            for i in range(split_size):
                splited_trajectories[i].slot_metadata = splited_slot_metadata[i]

        for field_name in all_trajectory.__dataclass_fields__.keys():
            value = getattr(all_trajectory, field_name)

            if value is None or isinstance(value, dict):
                continue

            if isinstance(value, int) or isinstance(value, str):
                for i in range(split_size):
                    setattr(splited_trajectories[i], field_name, value)
                continue
            elif isinstance(value, torch.Tensor):
                chunks = torch.chunk(value, split_size, dim=1)
                for i in range(split_size):
                    setattr(splited_trajectories[i], field_name, chunks[i].contiguous())
            else:
                raise ValueError(
                    f"Unsupported value type: {type(value)} for field_name: {field_name}"
                )

        del all_trajectory
        return splited_trajectories

    def to_splited_trajectories_by_sizes(
        self, split_sizes: list[int]
    ) -> list[Trajectory]:
        trajectory = self.to_trajectory()
        trajectories = [Trajectory() for _ in split_sizes]

        for field_name in trajectory.__dataclass_fields__:
            value = getattr(trajectory, field_name)
            if value is None:
                continue
            if isinstance(value, (int, str)):
                for split_trajectory in trajectories:
                    setattr(split_trajectory, field_name, value)
            elif isinstance(value, torch.Tensor):
                for split_trajectory, split_value in zip(
                    trajectories, torch.split(value, split_sizes, dim=1)
                ):
                    setattr(split_trajectory, field_name, split_value.contiguous())
            elif isinstance(value, dict):
                for split_trajectory, split_value in zip(
                    trajectories, split_dict(value, split_sizes, dim=1)
                ):
                    setattr(split_trajectory, field_name, split_value)
            else:
                raise ValueError(
                    f"Unsupported value type: {type(value)} for field_name: {field_name}"
                )

        return trajectories


class ContinuousBatchingRolloutCollector:
    """
    Trajectory-major rollout collector used as a stepping stone for slot-level
    continuous batching.

    During collection, each batch slot is appended to the trajectory identified
    by ``(group_id, group_member_id)`` in ``ChunkStepResult.slot_metadata``.
    When exported, trajectories are sorted by that key and rebuilt into the
    legacy time-major ``Trajectory`` format: [T, B, ...].
    """

    def __init__(
        self,
        max_episode_length: int = 0,
        *,
        group_size: int | None = None,
        batch_size: int | None = None,
        rollout_epoch: int = 1,
        target_chunk_steps: int | None = None,
    ):
        self.max_episode_length = max_episode_length
        self.group_size = group_size
        self.batch_size = batch_size
        self.rollout_epoch = int(rollout_epoch)
        self.target_chunk_steps = target_chunk_steps
        self._buffers: dict[tuple[int, int], EmbodiedRolloutResult] = {}
        self._slot_keys: list[tuple[int, int] | None] = []
        self._key_to_epoch_slot: dict[tuple[int, int], tuple[int, int]] = {}
        self.group_to_trajectory_keys: dict[int, set[tuple[int, int]]] = {}

    @property
    def _fixed_shape_export_enabled(self) -> bool:
        return (
            self.group_size is not None
            and self.batch_size is not None
            and self.target_chunk_steps is not None
        )

    @staticmethod
    def _slice_batch_value(value: Any, slot_idx: int) -> Any:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value[slot_idx : slot_idx + 1].clone().contiguous()
        if isinstance(value, dict):
            return {
                key: ContinuousBatchingRolloutCollector._slice_batch_value(
                    item, slot_idx
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [value[slot_idx]]
        return value

    @staticmethod
    def _infer_batch_size(result: ChunkStepResult) -> int:
        for value in (
            result.actions,
            result.rewards,
            result.dones,
            result.terminations,
            result.truncations,
            result.prev_logprobs,
            result.prev_values,
            result.versions,
        ):
            if isinstance(value, torch.Tensor):
                return int(value.shape[0])
        for data in (result.forward_inputs, result.slot_metadata):
            for value in data.values():
                if isinstance(value, torch.Tensor):
                    return int(value.shape[0])
        raise ValueError("Cannot infer batch size from ChunkStepResult.")

    @staticmethod
    def _metadata_key(
        slot_metadata: dict[str, torch.Tensor], slot_idx: int
    ) -> tuple[int, int]:
        group_id = int(slot_metadata["group_id"][slot_idx].reshape(-1)[0].item())
        group_member_id = int(
            slot_metadata["group_member_id"][slot_idx].reshape(-1)[0].item()
        )
        return group_id, group_member_id

    @staticmethod
    def _metadata_epoch_slot(
        slot_metadata: dict[str, torch.Tensor], slot_idx: int
    ) -> tuple[int, int]:
        rollout_epoch_id = int(
            slot_metadata["rollout_epoch_id"][slot_idx].reshape(-1)[0].item()
        )
        slot_id = int(slot_metadata["slot_id"][slot_idx].reshape(-1)[0].item())
        return rollout_epoch_id, slot_id

    @staticmethod
    def _metadata_is_dummy(
        slot_metadata: dict[str, torch.Tensor], slot_idx: int
    ) -> bool:
        if "is_dummy" not in slot_metadata:
            return False
        return bool(slot_metadata["is_dummy"][slot_idx].reshape(-1)[0].item())

    def _get_buffer(
        self, key: tuple[int, int], epoch_slot: tuple[int, int] | None = None
    ) -> EmbodiedRolloutResult:
        if key not in self._buffers:
            self._buffers[key] = EmbodiedRolloutResult(
                max_episode_length=self.max_episode_length
            )
            self.group_to_trajectory_keys.setdefault(key[0], set()).add(key)
            if epoch_slot is not None:
                self._key_to_epoch_slot[key] = epoch_slot
        elif epoch_slot is not None and self._key_to_epoch_slot.get(key) != epoch_slot:
            raise ValueError(
                f"Trajectory key {key} changed epoch/slot from "
                f"{self._key_to_epoch_slot.get(key)} to {epoch_slot}"
            )
        return self._buffers[key]

    def append_step_result(self, result: ChunkStepResult):
        if not result.slot_metadata:
            raise ValueError(
                "ContinuousBatchingRolloutCollector requires slot_metadata for every ChunkStepResult."
            )

        batch_size = self._infer_batch_size(result)
        self._slot_keys = []
        for slot_idx in range(batch_size):
            if self._metadata_is_dummy(result.slot_metadata, slot_idx):
                self._slot_keys.append(None)
                continue
            key = self._metadata_key(result.slot_metadata, slot_idx)
            epoch_slot = self._metadata_epoch_slot(result.slot_metadata, slot_idx)
            self._slot_keys.append(key)
            slot_result = ChunkStepResult(
                actions=self._slice_batch_value(result.actions, slot_idx),
                prev_logprobs=self._slice_batch_value(result.prev_logprobs, slot_idx),
                prev_values=self._slice_batch_value(result.prev_values, slot_idx),
                dones=self._slice_batch_value(result.dones, slot_idx),
                truncations=self._slice_batch_value(result.truncations, slot_idx),
                terminations=self._slice_batch_value(result.terminations, slot_idx),
                rewards=self._slice_batch_value(result.rewards, slot_idx),
                forward_inputs=self._slice_batch_value(
                    result.forward_inputs, slot_idx
                ),
                versions=self._slice_batch_value(result.versions, slot_idx),
                slot_metadata=self._slice_batch_value(
                    result.slot_metadata, slot_idx
                ),
            )
            self._get_buffer(key, epoch_slot).append_step_result(slot_result)

    def mark_last_step_with_flags(self, save_flags: torch.Tensor):
        if not self._slot_keys:
            return
        for slot_idx, key in enumerate(self._slot_keys):
            if key is None:
                continue
            self._get_buffer(key).mark_last_step_with_flags(
                save_flags[slot_idx : slot_idx + 1]
            )

    def update_last_actions(
        self, intervene_actions: torch.Tensor, intervene_flags: torch.Tensor
    ):
        if not self._slot_keys:
            return
        for slot_idx, key in enumerate(self._slot_keys):
            if key is None:
                continue
            self._get_buffer(key).update_last_actions(
                intervene_actions[slot_idx : slot_idx + 1],
                intervene_flags[slot_idx : slot_idx + 1],
            )

    def append_transitions(self, curr_obs=None, next_obs=None):
        assert curr_obs is not None and next_obs is not None
        if not self._slot_keys:
            raise ValueError(
                "Cannot append transitions before appending a slot-metadata step."
            )
        for slot_idx, key in enumerate(self._slot_keys):
            if key is None:
                continue
            self._get_buffer(key).append_transitions(
                self._slice_batch_value(curr_obs, slot_idx),
                self._slice_batch_value(next_obs, slot_idx),
            )

    def clear(self):
        for buffer in self._buffers.values():
            buffer.clear()
        self._buffers.clear()
        self._slot_keys.clear()
        self._key_to_epoch_slot.clear()
        self.group_to_trajectory_keys.clear()

    @staticmethod
    def _cat_nested_batches(epoch_batches: list[dict[str, Any]]) -> dict[str, Any]:
        if not epoch_batches:
            return {}
        merged: dict[str, Any] = {}
        keys = set()
        for batch in epoch_batches:
            keys.update(batch.keys())
        for key in keys:
            values = [batch[key] for batch in epoch_batches if key in batch]
            if not values:
                continue
            if isinstance(values[0], torch.Tensor):
                merged[key] = torch.cat(values, dim=0).contiguous()
            elif isinstance(values[0], dict):
                merged[key] = ContinuousBatchingRolloutCollector._cat_nested_batches(
                    values
                )
            else:
                raise ValueError(
                    f"Unsupported value type while concatenating trajectory batches: {type(values[0])}"
                )
        return merged

    def _epoch_batches(self) -> list[dict[str, Any]]:
        if not self._buffers:
            return []
        keys_by_epoch: dict[int, list[tuple[int, int]]] = {}
        for key in self._buffers:
            if key not in self._key_to_epoch_slot:
                raise ValueError(f"Missing rollout epoch/slot metadata for key {key}")
            epoch_id, _ = self._key_to_epoch_slot[key]
            keys_by_epoch.setdefault(epoch_id, []).append(key)

        epoch_batches = []
        for epoch_id in sorted(keys_by_epoch):
            ordered_keys = sorted(
                keys_by_epoch[epoch_id],
                key=lambda key: self._key_to_epoch_slot[key][1],
            )
            trajectories = [
                self._buffers[key].to_trajectory() for key in ordered_keys
            ]
            epoch_batches.append(convert_trajectories_to_batch(trajectories))
        return epoch_batches

    @staticmethod
    def _pad_tensor_time_dim(
        tensor: torch.Tensor,
        target_len: int,
        *,
        repeat_last: bool = False,
        fill_value: bool | int | float = 0,
    ) -> torch.Tensor:
        current_len = int(tensor.shape[0])
        if current_len > target_len:
            raise ValueError(
                f"Cannot pad tensor with time length {current_len} to shorter "
                f"target length {target_len}."
            )
        if current_len == target_len:
            return tensor.cpu().contiguous()

        pad_len = target_len - current_len
        if repeat_last:
            if current_len <= 0:
                raise ValueError("Cannot repeat-pad an empty time dimension.")
            pad = tensor[-1:].expand(pad_len, *tensor.shape[1:]).clone()
        else:
            pad = torch.full(
                (pad_len, *tensor.shape[1:]),
                fill_value,
                dtype=tensor.dtype,
                device=tensor.device,
            )
        return torch.cat([tensor, pad], dim=0).cpu().contiguous()

    @classmethod
    def _pad_nested_time_dim(
        cls,
        data: dict[str, Any],
        target_len: int,
        *,
        repeat_last: bool = False,
    ) -> dict[str, Any]:
        padded = {}
        for key, value in data.items():
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                padded[key] = cls._pad_tensor_time_dim(
                    value,
                    target_len,
                    repeat_last=repeat_last,
                )
            elif isinstance(value, dict):
                padded[key] = cls._pad_nested_time_dim(
                    value,
                    target_len,
                    repeat_last=repeat_last,
                )
            else:
                raise ValueError(
                    f"Unsupported nested value type while padding: {type(value)}"
                )
        return padded

    @staticmethod
    def _trim_slot_metadata_to_actions(
        slot_metadata: dict[str, Any], action_len: int
    ) -> dict[str, Any]:
        trimmed = {}
        for key, value in slot_metadata.items():
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                if int(value.shape[0]) == action_len + 1:
                    trimmed[key] = value[-action_len:].cpu().contiguous()
                elif int(value.shape[0]) == action_len:
                    trimmed[key] = value.cpu().contiguous()
                elif int(value.shape[0]) > action_len:
                    trimmed[key] = value[-action_len:].cpu().contiguous()
                else:
                    raise ValueError(
                        f"slot_metadata[{key}] time length {value.shape[0]} is "
                        f"shorter than action length {action_len}."
                    )
            elif isinstance(value, dict):
                trimmed[key] = ContinuousBatchingRolloutCollector._trim_slot_metadata_to_actions(
                    value, action_len
                )
            else:
                raise ValueError(
                    f"Unsupported slot_metadata value type while trimming: {type(value)}"
                )
        return trimmed

    @classmethod
    def _pad_slot_metadata(
        cls, slot_metadata: dict[str, Any], target_len: int, action_len: int
    ) -> dict[str, Any]:
        trimmed = cls._trim_slot_metadata_to_actions(slot_metadata, action_len)
        padded = cls._pad_nested_time_dim(trimmed, target_len, repeat_last=True)
        if target_len > action_len:
            for key, value in padded.items():
                if not isinstance(value, torch.Tensor):
                    continue
                if key == "is_active":
                    value[action_len:] = False
                elif key in {"is_done", "group_complete"}:
                    value[action_len:] = True
        return padded

    @staticmethod
    def _trajectory_is_complete(trajectory: Trajectory) -> bool:
        return (
            trajectory.dones is not None
            and trajectory.dones.shape[0] > 0
            and bool(trajectory.dones[-1].bool().any().item())
        )

    def _pad_trajectory_to_fixed_chunks(
        self, trajectory: Trajectory, key: tuple[int, int]
    ) -> Trajectory:
        assert self.target_chunk_steps is not None
        target_len = int(self.target_chunk_steps)
        action_len = (
            int(trajectory.actions.shape[0])
            if isinstance(trajectory.actions, torch.Tensor)
            else 0
        )
        if action_len <= 0:
            raise ValueError(f"Trajectory {key} has no actions.")
        if action_len > target_len:
            raise ValueError(
                f"Trajectory {key} has {action_len} chunks, exceeding target "
                f"chunk steps {target_len}."
            )
        if not self._trajectory_is_complete(trajectory):
            raise ValueError(f"Trajectory {key} is not complete.")

        padded = Trajectory(
            max_episode_length=self.max_episode_length,
            model_weights_id=trajectory.model_weights_id,
        )
        action_fields = (
            "actions",
            "intervene_flags",
            "rewards",
            "prev_logprobs",
        )
        for field_name in action_fields:
            value = getattr(trajectory, field_name)
            if isinstance(value, torch.Tensor):
                setattr(
                    padded,
                    field_name,
                    self._pad_tensor_time_dim(value, target_len),
                )
        if isinstance(trajectory.versions, torch.Tensor):
            padded.versions = self._pad_tensor_time_dim(
                trajectory.versions,
                target_len,
                repeat_last=True,
            )

        if isinstance(trajectory.prev_values, torch.Tensor):
            prev_value_target_len = (
                target_len + 1
                if int(trajectory.prev_values.shape[0]) == action_len + 1
                else target_len
            )
            padded.prev_values = self._pad_tensor_time_dim(
                trajectory.prev_values, prev_value_target_len
            )

        boundary_fields = ("dones", "terminations", "truncations")
        for field_name in boundary_fields:
            value = getattr(trajectory, field_name)
            if not isinstance(value, torch.Tensor):
                continue
            expected_len = action_len + 1
            if int(value.shape[0]) != expected_len:
                raise ValueError(
                    f"Trajectory {key} field {field_name} has time length "
                    f"{value.shape[0]}, expected {expected_len}."
                )
            fill_value = field_name == "dones"
            setattr(
                padded,
                field_name,
                self._pad_tensor_time_dim(
                    value,
                    target_len + 1,
                    fill_value=fill_value,
                ),
            )

        if trajectory.forward_inputs:
            padded.forward_inputs = self._pad_nested_time_dim(
                trajectory.forward_inputs,
                target_len,
                repeat_last=True,
            )
        if trajectory.slot_metadata:
            padded.slot_metadata = self._pad_slot_metadata(
                trajectory.slot_metadata,
                target_len,
                action_len,
            )
        if trajectory.curr_obs:
            padded.curr_obs = self._pad_nested_time_dim(
                trajectory.curr_obs,
                target_len,
                repeat_last=True,
            )
        if trajectory.next_obs:
            padded.next_obs = self._pad_nested_time_dim(
                trajectory.next_obs,
                target_len,
                repeat_last=True,
            )
        return padded

    def _ordered_completed_group_keys(self) -> list[tuple[int, int]]:
        assert self.group_size is not None
        ordered_keys: list[tuple[int, int]] = []
        for group_id in sorted(self.group_to_trajectory_keys):
            group_keys = sorted(
                self.group_to_trajectory_keys[group_id],
                key=lambda item: item[1],
            )
            if len(group_keys) != self.group_size:
                raise ValueError(
                    f"Group {group_id} has {len(group_keys)} trajectories, "
                    f"expected group_size={self.group_size}."
                )
            expected_members = list(range(self.group_size))
            actual_members = [member_id for _, member_id in group_keys]
            if actual_members != expected_members:
                raise ValueError(
                    f"Group {group_id} has members {actual_members}, "
                    f"expected {expected_members}."
                )
            for key in group_keys:
                trajectory = self._buffers[key].to_trajectory()
                if not self._trajectory_is_complete(trajectory):
                    raise ValueError(f"Trajectory {key} in group {group_id} is incomplete.")
            ordered_keys.extend(group_keys)
        return ordered_keys

    def _fixed_shape_epoch_batches(self) -> list[dict[str, Any]]:
        if not self._buffers:
            return []
        assert self.batch_size is not None
        assert self.group_size is not None
        if self.batch_size % self.group_size != 0:
            raise ValueError(
                f"batch_size={self.batch_size} must be divisible by "
                f"group_size={self.group_size}."
            )

        ordered_keys = self._ordered_completed_group_keys()
        expected_trajectories = self.batch_size * self.rollout_epoch
        if len(ordered_keys) != expected_trajectories:
            raise ValueError(
                f"Continuous batching collected {len(ordered_keys)} complete "
                f"trajectories, expected {expected_trajectories} "
                f"({self.batch_size=} * {self.rollout_epoch=})."
            )

        epoch_batches = []
        for epoch_idx in range(self.rollout_epoch):
            begin = epoch_idx * self.batch_size
            end = begin + self.batch_size
            epoch_keys = ordered_keys[begin:end]
            trajectories = [
                self._pad_trajectory_to_fixed_chunks(
                    self._buffers[key].to_trajectory(), key
                )
                for key in epoch_keys
            ]
            epoch_batches.append(convert_trajectories_to_batch(trajectories))
        return epoch_batches

    def to_trajectory(self) -> Trajectory:
        epoch_batches = (
            self._fixed_shape_epoch_batches()
            if self._fixed_shape_export_enabled
            else self._epoch_batches()
        )
        if not epoch_batches:
            return Trajectory(max_episode_length=self.max_episode_length)

        batch = self._cat_nested_batches(epoch_batches)
        trajectory = Trajectory(max_episode_length=self.max_episode_length)
        for key, value in batch.items():
            setattr(trajectory, key, value)
        trajectory.model_weights_id = get_model_weights_id(
            trajectory.versions
            if trajectory.versions is not None
            else torch.zeros(1, dtype=torch.float32)
        )
        return trajectory

    def to_splited_trajectories(self, split_size: int) -> list[Trajectory]:
        trajectory = self.to_trajectory()
        return self._split_trajectory_by_chunks(trajectory, split_size)

    def to_splited_trajectories_by_sizes(
        self, split_sizes: list[int]
    ) -> list[Trajectory]:
        trajectory = self.to_trajectory()
        return self._split_trajectory_by_sizes(trajectory, split_sizes)

    @staticmethod
    def _split_trajectory_by_chunks(
        trajectory: Trajectory, split_size: int
    ) -> list[Trajectory]:
        trajectories = [Trajectory() for _ in range(split_size)]

        if len(trajectory.curr_obs) > 0:
            split_obs = split_dict_to_chunk(trajectory.curr_obs, split_size, dim=1)
            for i in range(split_size):
                trajectories[i].curr_obs = split_obs[i]
        if len(trajectory.next_obs) > 0:
            split_obs = split_dict_to_chunk(trajectory.next_obs, split_size, dim=1)
            for i in range(split_size):
                trajectories[i].next_obs = split_obs[i]
        if trajectory.forward_inputs is not None and len(trajectory.forward_inputs) > 0:
            split_forward_inputs = split_dict_to_chunk(
                trajectory.forward_inputs, split_size, dim=1
            )
            for i in range(split_size):
                trajectories[i].forward_inputs = split_forward_inputs[i]
        if trajectory.slot_metadata is not None and len(trajectory.slot_metadata) > 0:
            split_slot_metadata = split_dict_to_chunk(
                trajectory.slot_metadata, split_size, dim=1
            )
            for i in range(split_size):
                trajectories[i].slot_metadata = split_slot_metadata[i]

        for field_name in trajectory.__dataclass_fields__.keys():
            value = getattr(trajectory, field_name)
            if value is None or isinstance(value, dict):
                continue
            if isinstance(value, (int, str)):
                for split_trajectory in trajectories:
                    setattr(split_trajectory, field_name, value)
            elif isinstance(value, torch.Tensor):
                chunks = torch.chunk(value, split_size, dim=1)
                for split_trajectory, split_value in zip(trajectories, chunks):
                    setattr(split_trajectory, field_name, split_value.contiguous())
            else:
                raise ValueError(
                    f"Unsupported value type: {type(value)} for field_name: {field_name}"
                )

        return trajectories

    @staticmethod
    def _split_trajectory_by_sizes(
        trajectory: Trajectory, split_sizes: list[int]
    ) -> list[Trajectory]:
        trajectories = [Trajectory() for _ in split_sizes]

        for field_name in trajectory.__dataclass_fields__:
            value = getattr(trajectory, field_name)
            if value is None:
                continue
            if isinstance(value, (int, str)):
                for split_trajectory in trajectories:
                    setattr(split_trajectory, field_name, value)
            elif isinstance(value, torch.Tensor):
                for split_trajectory, split_value in zip(
                    trajectories, torch.split(value, split_sizes, dim=1)
                ):
                    setattr(split_trajectory, field_name, split_value.contiguous())
            elif isinstance(value, dict):
                for split_trajectory, split_value in zip(
                    trajectories, split_dict(value, split_sizes, dim=1)
                ):
                    setattr(split_trajectory, field_name, split_value)
            else:
                raise ValueError(
                    f"Unsupported value type: {type(value)} for field_name: {field_name}"
                )

        return trajectories


def convert_trajectories_to_batch(
    trajectories: list[Trajectory],
) -> dict[str, torch.Tensor]:
    """
    convert a list of trajectories to a batch dict, the shape of the batch is [T, B, ...].
    """
    if not trajectories:
        return {}

    batch: dict[str, torch.Tensor] = {}

    # -------- obs / forward_inputs: dict[str, Tensor] --------
    if trajectories[0].curr_obs:
        all_keys: set[str] = set()
        for traj in trajectories:
            all_keys.update(traj.curr_obs.keys())
        batch["curr_obs"] = {}
        for key in all_keys:
            tensors = [
                traj.curr_obs[key] for traj in trajectories if key in traj.curr_obs
            ]
            if tensors:
                batch["curr_obs"][key] = torch.cat(tensors, dim=1)

    if trajectories[0].next_obs:
        all_keys: set[str] = set()
        for traj in trajectories:
            all_keys.update(traj.next_obs.keys())
        batch["next_obs"] = {}
        for key in all_keys:
            tensors = [
                traj.next_obs[key] for traj in trajectories if key in traj.next_obs
            ]
            if tensors:
                batch["next_obs"][key] = torch.cat(tensors, dim=1)

    if trajectories[0].forward_inputs:
        all_keys: set[str] = set()
        for traj in trajectories:
            all_keys.update(traj.forward_inputs.keys())
        batch["forward_inputs"] = {}
        for key in all_keys:
            tensors = [
                traj.forward_inputs[key]
                for traj in trajectories
                if key in traj.forward_inputs
            ]
            if tensors:
                batch["forward_inputs"][key] = torch.cat(tensors, dim=1)

    if trajectories[0].slot_metadata:
        all_keys: set[str] = set()
        for traj in trajectories:
            all_keys.update(traj.slot_metadata.keys())
        batch["slot_metadata"] = {}
        for key in all_keys:
            tensors = [
                traj.slot_metadata[key]
                for traj in trajectories
                if key in traj.slot_metadata
            ]
            if tensors:
                batch["slot_metadata"][key] = torch.cat(tensors, dim=1)

    # -------- tensor fields --------
    reference_trajectory = trajectories[0]
    for field_name in reference_trajectory.__dataclass_fields__.keys():
        if not isinstance(getattr(reference_trajectory, field_name), torch.Tensor):
            continue
        field_list = [
            getattr(traj, field_name)
            for traj in trajectories
            if getattr(traj, field_name) is not None
        ]
        if field_list:
            batch[field_name] = torch.cat(field_list, dim=1)

    return batch
