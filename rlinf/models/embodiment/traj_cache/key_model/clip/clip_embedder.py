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

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from transformers import CLIPModel, CLIPProcessor

DEFAULT_CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
ImageInput = str | os.PathLike[str] | Image.Image | np.ndarray | torch.Tensor


class ClipImageEmbedder(nn.Module):
    """Frozen CLIP image encoder for trajectory-cache keys.

    The class intentionally keeps a small surface area so rollout, env, and
    slice-model code can use the same module without depending on local
    preprocessing details.

    Args:
        model_name_or_path: Hugging Face model id or local checkpoint directory.
        device: Device used for inference. Defaults to CUDA when available.
        dtype: Model inference dtype. Defaults to fp16 on CUDA and fp32 on CPU.
        normalize: Whether to L2-normalize output embeddings.
        local_files_only: Forwarded to Hugging Face ``from_pretrained``.
        trust_remote_code: Forwarded to Hugging Face ``from_pretrained``.
    """

    def __init__(
        self,
        model_name_or_path: str | os.PathLike[str] = DEFAULT_CLIP_MODEL_NAME,
        device: str | torch.device | None = None,
        dtype: torch.dtype | str | None = None,
        normalize: bool = True,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__()
        self.model_name_or_path = str(model_name_or_path)
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device if device is not None else default_device)
        self.dtype = self._resolve_dtype(dtype, self.device)
        self.normalize = bool(normalize)

        self.processor = CLIPProcessor.from_pretrained(
            self.model_name_or_path,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        self.model = CLIPModel.from_pretrained(
            self.model_name_or_path,
            torch_dtype=self.dtype,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        ).eval()
        self.model.to(self.device)
        self.requires_grad_(False)

    @staticmethod
    def _resolve_dtype(
        dtype: torch.dtype | str | None,
        device: torch.device,
    ) -> torch.dtype:
        if dtype is None:
            return torch.float16 if device.type == "cuda" else torch.float32
        if isinstance(dtype, torch.dtype):
            return dtype
        normalized = str(dtype).lower()
        aliases = {
            "fp16": torch.float16,
            "float16": torch.float16,
            "half": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        if normalized not in aliases:
            raise ValueError(f"Unsupported CLIP dtype: {dtype}")
        return aliases[normalized]

    @property
    def embedding_dim(self) -> int:
        return int(self.model.config.projection_dim)

    @torch.inference_mode()
    def forward(
        self,
        images: ImageInput | Sequence[ImageInput],
        *,
        normalize: bool | None = None,
        output_device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """Encode images into CLIP image embeddings.

        Args:
            images: Single image or a batch of images. Supported forms are path,
                PIL image, numpy array, torch tensor, or a sequence of those.
            normalize: Per-call override for L2 normalization.
            output_device: Optional output device. Defaults to the model device.

        Returns:
            Tensor shaped ``[batch, embedding_dim]``.
        """

        image_batch = self._to_image_list(images)
        inputs = self.processor(images=image_batch, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=self.device, dtype=self.dtype)
        embeddings = self.model.get_image_features(pixel_values=pixel_values)
        should_normalize = self.normalize if normalize is None else normalize
        if should_normalize:
            embeddings = F.normalize(embeddings.float(), p=2, dim=-1).to(embeddings.dtype)
        if output_device is not None:
            embeddings = embeddings.to(output_device)
        return embeddings

    encode_image = forward
    embed_image = forward

    def _to_image_list(
        self,
        images: ImageInput | Sequence[ImageInput],
    ) -> list[Image.Image | np.ndarray]:
        if isinstance(images, (str, os.PathLike, Image.Image, np.ndarray, torch.Tensor)):
            images = [images]

        if not isinstance(images, Sequence):
            raise TypeError(f"Unsupported image input type: {type(images)}")

        image_list: list[Image.Image | np.ndarray] = []
        for image in images:
            image_list.extend(self._single_to_image_list(image))
        if not image_list:
            raise ValueError("CLIP image input is empty.")
        return image_list

    def _single_to_image_list(self, image: ImageInput) -> list[Image.Image | np.ndarray]:
        if isinstance(image, (str, os.PathLike)):
            with Image.open(Path(image)) as pil_image:
                return [pil_image.convert("RGB")]
        if isinstance(image, Image.Image):
            return [image.convert("RGB")]
        if isinstance(image, np.ndarray):
            return self._numpy_to_image_list(image)
        if isinstance(image, torch.Tensor):
            return self._numpy_to_image_list(image.detach().cpu().numpy())
        raise TypeError(f"Unsupported image item type: {type(image)}")

    def _numpy_to_image_list(self, array: np.ndarray) -> list[Image.Image | np.ndarray]:
        array = np.asarray(array)
        if array.ndim == 3:
            return [self._numpy_to_pil(array)]
        if array.ndim == 4:
            return [self._numpy_to_pil(item) for item in array]
        raise ValueError(
            "Expected image array/tensor with shape [H,W,C], [C,H,W], "
            "[B,H,W,C], or [B,C,H,W]; got "
            f"{tuple(array.shape)}."
        )

    def _numpy_to_pil(self, array: np.ndarray) -> Image.Image:
        if array.ndim != 3:
            raise ValueError(f"Expected a 3D image array, got {tuple(array.shape)}")

        # Accept channel-first tensors from model code and channel-last arrays
        # from image tooling. CLIP expects RGB images.
        if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = np.moveaxis(array, 0, -1)

        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        elif array.shape[-1] == 4:
            array = array[..., :3]
        elif array.shape[-1] != 3:
            raise ValueError(
                "Expected image channel dimension to be 1, 3, or 4; got "
                f"{tuple(array.shape)}."
            )

        if np.issubdtype(array.dtype, np.floating):
            max_value = float(np.nanmax(array)) if array.size else 0.0
            min_value = float(np.nanmin(array)) if array.size else 0.0
            if min_value >= -1.0 and max_value <= 1.0 and min_value < 0.0:
                array = (array + 1.0) * 127.5
            elif max_value <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0.0, 255.0).astype(np.uint8)
        elif array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)

        return Image.fromarray(np.ascontiguousarray(array), mode="RGB")


def build_clip_image_embedder(
    model_name_or_path: str | os.PathLike[str] = DEFAULT_CLIP_MODEL_NAME,
    **kwargs: Any,
) -> ClipImageEmbedder:
    """Factory used by downstream modules that prefer config-driven builders."""

    return ClipImageEmbedder(model_name_or_path=model_name_or_path, **kwargs)
