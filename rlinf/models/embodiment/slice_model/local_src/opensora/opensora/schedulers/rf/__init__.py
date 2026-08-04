from __future__ import annotations

# ruff: noqa: I001

from pkgutil import extend_path
from typing import Any

import torch
from tqdm import tqdm

from opensora.registry import SCHEDULERS

__path__ = extend_path(__path__, __name__)

from .rectified_flow import RFlowScheduler, timestep_transform


_OPENSORA_STEP_RECORDER = None
_OPENSORA_SHARED_INITIAL_NOISE = False
_OPENSORA_SHARED_INITIAL_NOISE_REFERENCE_BATCH_ID = 0
_OPENSORA_PREFIX_STEPS = 0
_OPENSORA_PREFIX_REFERENCE_BATCH_ID = 0


def set_opensora_step_recorder(recorder: Any | None) -> None:
    global _OPENSORA_STEP_RECORDER
    _OPENSORA_STEP_RECORDER = recorder


def get_opensora_step_recorder() -> Any | None:
    return _OPENSORA_STEP_RECORDER


def set_opensora_shared_initial_noise(
    enabled: bool,
    reference_batch_id: int = 0,
) -> None:
    global _OPENSORA_SHARED_INITIAL_NOISE
    global _OPENSORA_SHARED_INITIAL_NOISE_REFERENCE_BATCH_ID
    _OPENSORA_SHARED_INITIAL_NOISE = bool(enabled)
    _OPENSORA_SHARED_INITIAL_NOISE_REFERENCE_BATCH_ID = int(reference_batch_id)


def set_opensora_prefix_steps(
    prefix_steps: int,
    reference_batch_id: int = 0,
) -> None:
    global _OPENSORA_PREFIX_STEPS
    global _OPENSORA_PREFIX_REFERENCE_BATCH_ID
    _OPENSORA_PREFIX_STEPS = int(prefix_steps)
    _OPENSORA_PREFIX_REFERENCE_BATCH_ID = int(reference_batch_id)


def clear_opensora_prefix_steps() -> None:
    set_opensora_prefix_steps(0, 0)


def _record_opensora_step(method_name: str, **payload: Any) -> None:
    recorder = _OPENSORA_STEP_RECORDER
    if recorder is None:
        return
    method = getattr(recorder, method_name, None)
    if method is not None:
        method(**payload)
        return
    update = getattr(recorder, "update", None)
    if update is not None:
        update(method_name, payload)


def _valid_reference_batch_id(batch_size: int, reference_batch_id: int) -> int:
    if batch_size <= 0:
        raise ValueError(f"Invalid batch size: {batch_size}.")
    if reference_batch_id < 0 or reference_batch_id >= batch_size:
        raise ValueError(
            f"reference_batch_id={reference_batch_id} is outside batch size {batch_size}."
        )
    return int(reference_batch_id)


def _copy_reference_latents(
    latents: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    reference_batch_id: int,
) -> torch.Tensor:
    batch_size = int(latents.shape[0])
    if batch_size <= 1:
        return latents
    reference_batch_id = _valid_reference_batch_id(batch_size, reference_batch_id)
    reference_latents = latents[reference_batch_id : reference_batch_id + 1].expand_as(
        latents
    )
    if mask is None:
        return reference_latents.clone()

    mask = mask.to(device=latents.device).bool()
    if mask.ndim != 2:
        raise ValueError(f"Expected OpenSora denoise mask [B, T], got {tuple(mask.shape)}.")
    mask = mask[:, None, :, None, None]
    return torch.where(mask, reference_latents, latents)


def _maybe_share_initial_noise(
    latents: torch.Tensor,
    *,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    if not _OPENSORA_SHARED_INITIAL_NOISE:
        return latents
    return _copy_reference_latents(
        latents,
        mask=mask,
        reference_batch_id=_OPENSORA_SHARED_INITIAL_NOISE_REFERENCE_BATCH_ID,
    )


def _maybe_apply_prefix_step(
    latents: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    step_index: int,
) -> torch.Tensor:
    if _OPENSORA_PREFIX_STEPS <= 0 or step_index >= _OPENSORA_PREFIX_STEPS:
        return latents
    return _copy_reference_latents(
        latents,
        mask=mask,
        reference_batch_id=_OPENSORA_PREFIX_REFERENCE_BATCH_ID,
    )


def _prepare_timesteps(
    *,
    num_sampling_steps: int,
    num_timesteps: int,
    use_discrete_timesteps: bool,
    use_timestep_transform: bool,
    z: torch.Tensor,
    device: torch.device | str,
    additional_args: dict[str, Any] | None,
) -> list[torch.Tensor]:
    timesteps = [
        (1.0 - i / num_sampling_steps) * num_timesteps
        for i in range(num_sampling_steps)
    ]
    if use_discrete_timesteps:
        timesteps = [int(round(t)) for t in timesteps]
    timestep_tensors = [
        torch.tensor([t] * z.shape[0], device=device) for t in timesteps
    ]
    if use_timestep_transform:
        timestep_tensors = [
            timestep_transform(t, additional_args, num_timesteps=num_timesteps)
            for t in timestep_tensors
        ]
    return timestep_tensors


@SCHEDULERS.register_module("rflow_cache")
class RFLOWCache:
    def __init__(
        self,
        num_sampling_steps=10,
        num_timesteps=1000,
        cfg_scale=4.0,
        use_discrete_timesteps=False,
        use_timestep_transform=False,
        **kwargs,
    ):
        self.num_sampling_steps = num_sampling_steps
        self.num_timesteps = num_timesteps
        self.cfg_scale = cfg_scale
        self.use_discrete_timesteps = use_discrete_timesteps
        self.use_timestep_transform = use_timestep_transform

        self.scheduler = RFlowScheduler(
            num_timesteps=num_timesteps,
            num_sampling_steps=num_sampling_steps,
            use_discrete_timesteps=use_discrete_timesteps,
            use_timestep_transform=use_timestep_transform,
            **kwargs,
        )

    def sample(
        self,
        model,
        z,
        y,
        device,
        additional_args=None,
        mask=None,
        guidance_scale=None,
        progress=True,
    ):
        if guidance_scale is None:
            guidance_scale = self.cfg_scale

        model_args = {"y": y.to(device, torch.long)}
        if additional_args is not None:
            model_args.update(additional_args)

        timesteps = _prepare_timesteps(
            num_sampling_steps=self.num_sampling_steps,
            num_timesteps=self.num_timesteps,
            use_discrete_timesteps=self.use_discrete_timesteps,
            use_timestep_transform=self.use_timestep_transform,
            z=z,
            device=device,
            additional_args=additional_args,
        )

        if mask is not None:
            noise_added = torch.zeros_like(mask, dtype=torch.bool)
            noise_added = noise_added | (mask == 1)
        z = _maybe_share_initial_noise(z, mask=mask)

        progress_wrap = tqdm if progress else (lambda x: x)
        kv_caches = None

        for i, t in progress_wrap(enumerate(timesteps)):
            mask_t_upper = None
            x0 = None
            if mask is not None:
                mask_t = mask * self.num_timesteps
                x0 = z.clone()
                x_noise = self.scheduler.add_noise(x0, torch.randn_like(x0), t)

                mask_t_upper = mask_t >= t.unsqueeze(1)
                model_args["x_mask"] = mask_t_upper
                mask_add_noise = mask_t_upper & ~noise_added
                assert sum(sum(mask_add_noise)).item() == 0
                z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
                noise_added = mask_t_upper

            z_in = z
            if kv_caches is not None:
                model_args["kv_caches"] = kv_caches
                model_args["x_mask"] = None
                z_in = z_in[:, :, -1:]

            _record_opensora_step(
                "begin_step",
                scheduler=self,
                step_index=i,
                timestep=t,
                latents=z,
                model_input=z_in,
                model_args=dict(model_args),
                mask_t_upper=mask_t_upper,
                mask=mask,
            )

            if kv_caches is not None:
                pred = model(z_in, t, **model_args)
            else:
                pred, kv_caches = model(z_in, t, **model_args)
            pred = pred.chunk(2, dim=1)[0]
            v_pred = pred

            dt = (
                timesteps[i] - timesteps[i + 1]
                if i < len(timesteps) - 1
                else timesteps[i]
            )
            dt = dt / self.num_timesteps
            _record_opensora_step(
                "after_model",
                scheduler=self,
                step_index=i,
                timestep=t,
                latents=z,
                model_input=z_in,
                pred=pred,
                velocity=v_pred,
                dt=dt,
                mask_t_upper=mask_t_upper,
                mask=mask,
            )

            z = z + v_pred * dt[:, None, None, None, None]

            if mask is not None:
                z = torch.where(mask_t_upper[:, None, :, None, None], z, x0)
            z = _maybe_apply_prefix_step(z, mask=mask, step_index=i)

            _record_opensora_step(
                "end_step",
                scheduler=self,
                step_index=i,
                timestep=t,
                latents=z,
                pred=pred,
                velocity=v_pred,
                dt=dt,
                mask_t_upper=mask_t_upper,
                mask=mask,
            )

        return z

    def training_losses(
        self, model, x_start, model_kwargs=None, noise=None, mask=None, weights=None, t=None
    ):
        return self.scheduler.training_losses(
            model, x_start, model_kwargs, noise, mask, weights, t
        )


@SCHEDULERS.register_module("rflow")
class RFLOW:
    def __init__(
        self,
        num_sampling_steps=10,
        num_timesteps=1000,
        cfg_scale=4.0,
        use_discrete_timesteps=False,
        use_timestep_transform=False,
        **kwargs,
    ):
        self.num_sampling_steps = num_sampling_steps
        self.num_timesteps = num_timesteps
        self.cfg_scale = cfg_scale
        self.use_discrete_timesteps = use_discrete_timesteps
        self.use_timestep_transform = use_timestep_transform

        self.scheduler = RFlowScheduler(
            num_timesteps=num_timesteps,
            num_sampling_steps=num_sampling_steps,
            use_discrete_timesteps=use_discrete_timesteps,
            use_timestep_transform=use_timestep_transform,
            **kwargs,
        )

    def sample(
        self,
        model,
        z,
        y,
        device,
        additional_args=None,
        mask=None,
        guidance_scale=None,
        progress=True,
    ):
        if guidance_scale is None:
            guidance_scale = self.cfg_scale

        model_args = {"y": y}
        if additional_args is not None:
            model_args.update(additional_args)

        timesteps = _prepare_timesteps(
            num_sampling_steps=self.num_sampling_steps,
            num_timesteps=self.num_timesteps,
            use_discrete_timesteps=self.use_discrete_timesteps,
            use_timestep_transform=self.use_timestep_transform,
            z=z,
            device=device,
            additional_args=additional_args,
        )

        if mask is not None:
            noise_added = torch.zeros_like(mask, dtype=torch.bool)
            noise_added = noise_added | (mask == 1)
        z = _maybe_share_initial_noise(z, mask=mask)

        progress_wrap = tqdm if progress else (lambda x: x)

        for i, t in progress_wrap(enumerate(timesteps)):
            mask_t_upper = None
            x0 = None
            if mask is not None:
                mask_t = mask * self.num_timesteps
                x0 = z.clone()
                base_noise = torch.randn_like(x0[0])
                noise = torch.stack([base_noise for _ in range(x0.shape[0])], dim=0)
                x_noise = self.scheduler.add_noise(x0, noise, t)

                mask_t_upper = mask_t >= t.unsqueeze(1)
                model_args["x_mask"] = mask_t_upper
                mask_add_noise = mask_t_upper & ~noise_added
                assert sum(sum(mask_add_noise)).item() == 0
                z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
                noise_added = mask_t_upper

            z_in = z
            _record_opensora_step(
                "begin_step",
                scheduler=self,
                step_index=i,
                timestep=t,
                latents=z,
                model_input=z_in,
                model_args=dict(model_args),
                mask_t_upper=mask_t_upper,
                mask=mask,
            )

            pred = model(z_in, t, **model_args).chunk(2, dim=1)[0]
            v_pred = pred

            dt = (
                timesteps[i] - timesteps[i + 1]
                if i < len(timesteps) - 1
                else timesteps[i]
            )
            dt = dt / self.num_timesteps
            _record_opensora_step(
                "after_model",
                scheduler=self,
                step_index=i,
                timestep=t,
                latents=z,
                model_input=z_in,
                pred=pred,
                velocity=v_pred,
                dt=dt,
                mask_t_upper=mask_t_upper,
                mask=mask,
            )

            z = z + v_pred * dt[:, None, None, None, None]

            if mask is not None:
                z = torch.where(mask_t_upper[:, None, :, None, None], z, x0)
            z = _maybe_apply_prefix_step(z, mask=mask, step_index=i)

            _record_opensora_step(
                "end_step",
                scheduler=self,
                step_index=i,
                timestep=t,
                latents=z,
                pred=pred,
                velocity=v_pred,
                dt=dt,
                mask_t_upper=mask_t_upper,
                mask=mask,
            )

        return z

    def training_losses(
        self, model, x_start, model_kwargs=None, noise=None, mask=None, weights=None, t=None
    ):
        return self.scheduler.training_losses(
            model, x_start, model_kwargs, noise, mask, weights, t
        )
