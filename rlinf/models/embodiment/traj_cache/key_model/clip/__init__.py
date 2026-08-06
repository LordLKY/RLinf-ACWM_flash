"""Plug-and-play CLIP image embedding module."""

from .clip_embedder import (
    DEFAULT_CLIP_MODEL_NAME,
    ClipImageEmbedder,
    build_clip_image_embedder,
)

__all__ = [
    "DEFAULT_CLIP_MODEL_NAME",
    "ClipImageEmbedder",
    "build_clip_image_embedder",
]

