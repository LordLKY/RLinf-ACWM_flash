from __future__ import annotations

from typing import Sequence

import torch
from torch.distributions import LogisticNormal


def mean_flat(tensor: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return tensor.mean(dim=list(range(1, len(tensor.shape))))

    if tensor.dim() != 5:
        raise ValueError(f"Expected tensor [B, C, T, H, W], got {tuple(tensor.shape)}.")
    if tensor.shape[2] != mask.shape[1]:
        raise ValueError(
            f"Mask temporal dim {mask.shape[1]} does not match tensor temporal dim {tensor.shape[2]}."
        )

    tensor = tensor.permute(0, 2, 1, 3, 4).reshape(tensor.shape[0], tensor.shape[2], -1)
    denom = mask.sum(dim=1) * tensor.shape[-1]
    return (tensor * mask.unsqueeze(2)).sum(dim=1).sum(dim=1) / denom


def _extract_into_tensor(
    arr: torch.Tensor,
    timesteps: torch.Tensor,
    broadcast_shape: Sequence[int],
) -> torch.Tensor:
    res = arr.to(timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)


def timestep_transform(
    t,
    model_kwargs,
    base_resolution=512 * 512,
    base_num_frames=1,
    scale=1.0,
    num_timesteps=1,
):
    for key in ["height", "width", "num_frames"]:
        if model_kwargs[key].dtype == torch.float16:
            model_kwargs[key] = model_kwargs[key].float()

    t = t / num_timesteps
    resolution = model_kwargs["height"] * model_kwargs["width"]
    ratio_space = (resolution / base_resolution).sqrt()
    if model_kwargs["num_frames"][0] == 1:
        num_frames = torch.ones_like(model_kwargs["num_frames"])
    else:
        num_frames = torch.ones_like(model_kwargs["num_frames"]) * 4
    ratio_time = (num_frames / base_num_frames).sqrt()

    ratio = ratio_space * ratio_time * scale
    new_t = ratio * t / (1 + (ratio - 1) * t)

    return new_t * num_timesteps


class RFlowScheduler:
    def __init__(
        self,
        num_timesteps=1000,
        num_sampling_steps=10,
        use_discrete_timesteps=False,
        sample_method="uniform",
        loc=0.0,
        scale=1.0,
        use_timestep_transform=False,
        transform_scale=1.0,
    ):
        self.num_timesteps = num_timesteps
        self.num_sampling_steps = num_sampling_steps
        self.use_discrete_timesteps = use_discrete_timesteps

        if sample_method not in ["uniform", "logit-normal"]:
            raise ValueError(f"Unsupported sample_method: {sample_method}")
        if sample_method != "uniform" and use_discrete_timesteps:
            raise ValueError("Only uniform sampling is supported for discrete timesteps.")
        self.sample_method = sample_method
        if sample_method == "logit-normal":
            self.distribution = LogisticNormal(torch.tensor([loc]), torch.tensor([scale]))
            self.sample_t = lambda x: self.distribution.sample((x.shape[0],))[:, 0].to(
                x.device
            )

        self.use_timestep_transform = use_timestep_transform
        self.transform_scale = transform_scale

    def training_losses(
        self,
        model,
        x_start,
        model_kwargs=None,
        noise=None,
        mask=None,
        weights=None,
        t=None,
    ):
        if t is None:
            if self.use_discrete_timesteps:
                t = torch.randint(
                    0, self.num_timesteps, (x_start.shape[0],), device=x_start.device
                )
            elif self.sample_method == "uniform":
                t = (
                    torch.rand((x_start.shape[0],), device=x_start.device)
                    * self.num_timesteps
                )
            elif self.sample_method == "logit-normal":
                t = self.sample_t(x_start) * self.num_timesteps

            if self.use_timestep_transform:
                t = timestep_transform(
                    t,
                    model_kwargs,
                    scale=self.transform_scale,
                    num_timesteps=self.num_timesteps,
                )

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        if noise.shape != x_start.shape:
            raise ValueError(
                f"Noise shape {tuple(noise.shape)} does not match x_start {tuple(x_start.shape)}."
            )

        x_t = self.add_noise(x_start, noise, t)
        if mask is not None:
            t0 = torch.ones_like(t) * 15
            x_t0 = self.add_noise(x_start, noise, t0)
            x_t = torch.where(mask[:, None, :, None, None], x_t, x_t0)

        model_output = model(x_t, t, **model_kwargs)
        velocity_pred = model_output.chunk(2, dim=1)[0]
        if weights is None:
            loss = mean_flat((velocity_pred - (x_start - noise)).pow(2), mask=mask)
        else:
            weight = _extract_into_tensor(weights, t, x_start.shape)
            loss = mean_flat(
                weight * (velocity_pred - (x_start - noise)).pow(2),
                mask=mask,
            )

        return {"loss": loss}

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        timepoints = timesteps.float() / self.num_timesteps
        timepoints = 1 - timepoints

        timepoints = timepoints.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        timepoints = timepoints.repeat(
            1, noise.shape[1], noise.shape[2], noise.shape[3], noise.shape[4]
        )

        return timepoints * original_samples + (1 - timepoints) * noise
