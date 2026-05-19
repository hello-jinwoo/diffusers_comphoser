"""Shared RGB image loading and paired image transforms for ComPhoser."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch import Tensor


def ensure_rgb_image(image: Image.Image) -> Image.Image:
    normalized = exif_transpose(image)
    if normalized.mode != "RGB":
        normalized = normalized.convert("RGB")
    else:
        normalized = normalized.copy()
    return normalized


def load_rgb_image(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return ensure_rgb_image(image)


def save_rgb_image(path: str | Path, image: Image.Image) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_rgb_image(image).save(output_path)


def paired_image_transform(
    image: Image.Image,
    cond_image: Image.Image | None = None,
    *,
    size: tuple[int, int],
    center_crop: bool,
    random_flip: bool,
) -> tuple[Tensor, Tensor | None]:
    target_height, target_width = int(size[0]), int(size[1])
    resized_image = ensure_rgb_image(image).resize((target_width, target_height), resample=Image.BILINEAR)
    resized_cond_image = None
    if cond_image is not None:
        resized_cond_image = ensure_rgb_image(cond_image).resize(
            (target_width, target_height),
            resample=Image.BILINEAR,
        )

    if center_crop:
        resized_image = _center_crop(resized_image, (target_height, target_width))
        if resized_cond_image is not None:
            resized_cond_image = _center_crop(resized_cond_image, (target_height, target_width))

    if random_flip and random.random() < 0.5:
        resized_image = resized_image.transpose(Image.FLIP_LEFT_RIGHT)
        if resized_cond_image is not None:
            resized_cond_image = resized_cond_image.transpose(Image.FLIP_LEFT_RIGHT)

    return image_to_tensor(resized_image), None if resized_cond_image is None else image_to_tensor(resized_cond_image)


def image_to_tensor(image: Image.Image) -> Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor.sub(0.5).div(0.5)


def _center_crop(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_height, target_width = size
    width, height = image.size
    left = max((width - target_width) // 2, 0)
    top = max((height - target_height) // 2, 0)
    return image.crop((left, top, left + target_width, top + target_height))


__all__ = [
    "ensure_rgb_image",
    "image_to_tensor",
    "load_rgb_image",
    "paired_image_transform",
    "save_rgb_image",
]
