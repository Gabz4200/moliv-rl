from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torchvision import transforms

IMAGENET_MEAN: tuple[float, float, float] = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STD: tuple[float, float, float] = (
    0.229,
    0.224,
    0.225,
)


def _next_multiple(value: int, divisor: int) -> int:
    """Return the smallest multiple of ``divisor`` that is >= ``value``."""
    return int(math.ceil(value / divisor) * divisor)


def _compute_resize_size(image_size: int, resize_scale: float) -> int:
    """Return the resize dimension used before center-cropping back to ``image_size``."""
    return max(image_size, round(image_size * resize_scale))


def _build_pad_to_multiple(divisor: int = 8, fill: float = 0):
    """Build a padding transform that pads to the next multiple of ``divisor``.

    The padding is applied symmetrically when possible; any remainder from
    odd differences is added to the right/bottom side so the original
    content stays centered.
    """

    class _PadToMultiple:
        def __init__(self, divisor: int = 8, fill: float = 0) -> None:
            self.divisor = divisor
            self.fill = fill

        def __call__(self, img: torch.Tensor) -> torch.Tensor:
            if isinstance(img, torch.Tensor):
                _, h, w = img.shape
            else:
                w, h = img.size
            target_h = _next_multiple(h, self.divisor)
            target_w = _next_multiple(w, self.divisor)
            pad_h = max(target_h - h, 0)
            pad_w = max(target_w - w, 0)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            return transforms.functional.pad(
                img,
                [pad_left, pad_top, pad_right, pad_bottom],
                fill=self.fill,
            )

    return _PadToMultiple(divisor=divisor, fill=fill)


class ResizePreserveAspectRatio:
    """Resize an image so that its height equals ``height`` while preserving aspect ratio.

    The width is scaled proportionally; no cropping is performed, so the
    transformed image may have non-square dimensions.
    """

    def __init__(self, height: int, antialias: bool = True) -> None:
        self.height = height
        self.antialias = antialias

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if isinstance(img, torch.Tensor):
            _, h, w = img.shape
        else:
            w, h = img.size
        if h == self.height:
            return img
        new_w = max(1, round(w * (self.height / h)))
        return transforms.functional.resize(
            img,
            [self.height, new_w],
            antialias=self.antialias,
        )


def get_train_transforms(
    image_size: int = 256,
    *,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    horizontal_flip_probability: float = 0.5,
    color_jitter_probability: float = 0.4,
    rotation_degrees: float = 5.0,
    affine_degrees: float = 5.0,
    affine_translate: float = 0.05,
    affine_scale: tuple[float, float] = (0.95, 1.05),
    gaussian_blur_probability: float = 0.2,
    gaussian_blur_kernel_size: int = 3,
    random_erasing_probability: float = 0.15,
    random_erasing_scale: tuple[float, float] = (0.02, 0.08),
    random_erasing_ratio: tuple[float, float] = (0.3, 3.3),
    pad_multiple: int = 8,
    pad_fill: float = 0,
) -> transforms.Compose:
    r"""get_train_transforms(image_size=256, ...) -> transforms.Compose

    Build the standard data augmentation and normalization pipeline for training.

    The pipeline preserves the full image content by:
    1. Resizing so the **height** becomes ``image_size`` while the width scales
       proportionally.
    2. Padding both height and width to the next multiple of ``pad_multiple``
       (default: 8) without cropping.
    3. Applying lightweight color/spatial augmentations.
    4. Converting to tensor and normalizing.

    Args:
        image_size (int, optional): Target height after resizing. Default: ``256``
        mean (Sequence of float, optional): Per-channel normalization means. Default: :attr:`IMAGENET_MEAN`
        std (Sequence of float, optional): Per-channel normalization standard deviations. Default: :attr:`IMAGENET_STD`
        horizontal_flip_probability (float, optional): Probability of random horizontal flip. Default: ``0.5``
        color_jitter_probability (float, optional): Probability of applying color jitter. Default: ``0.4``
        rotation_degrees (float, optional): Maximum absolute rotation in degrees. Default: ``5.0``
        affine_degrees (float, optional): Maximum absolute rotation for the affine transform in degrees. Default: ``5.0``
        affine_translate (float, optional): Maximum absolute translation fraction for the affine transform. Default: ``0.05``
        affine_scale (tuple of float, optional): Scaling range for the affine transform. Default: ``(0.95, 1.05)``
        gaussian_blur_probability (float, optional): Probability of applying Gaussian blur. Default: ``0.2``
        gaussian_blur_kernel_size (int, optional): Kernel size for Gaussian blur. Default: ``3``
        random_erasing_probability (float, optional): Probability of applying random erasing. Default: ``0.15``
        random_erasing_scale (tuple of float, optional): Scale range for random erasing. Default: ``(0.02, 0.08)``
        random_erasing_ratio (tuple of float, optional): Aspect ratio range for random erasing. Default: ``(0.3, 3.3)``
        pad_multiple (int, optional): Pad height/width to the next multiple of this value. Default: ``8``
        pad_fill (int or float, optional): Fill value for padding. Default: ``0``

    Returns:
        transforms.Compose: Composed torchvision transformation pipeline.

    Examples::

        >>> transform = get_train_transforms(image_size=256)
    """
    augments: list[Any] = [
        transforms.RandomHorizontalFlip(p=horizontal_flip_probability),
    ]

    if color_jitter_probability > 0:
        augments.append(
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.1,
                    )
                ],
                p=color_jitter_probability,
            )
        )

    if rotation_degrees > 0:
        augments.append(
            transforms.RandomRotation(degrees=rotation_degrees, fill=pad_fill)
        )

    if affine_degrees > 0 or affine_translate > 0 or affine_scale != (1.0, 1.0):
        augments.append(
            transforms.RandomAffine(
                degrees=affine_degrees,
                translate=(affine_translate, affine_translate),
                scale=affine_scale,
                fill=pad_fill,
            )
        )

    if gaussian_blur_probability > 0 and gaussian_blur_kernel_size > 0:
        augments.append(
            transforms.RandomApply(
                [
                    transforms.GaussianBlur(
                        kernel_size=gaussian_blur_kernel_size,
                    )
                ],
                p=gaussian_blur_probability,
            )
        )

    return transforms.Compose(
        [
            ResizePreserveAspectRatio(height=image_size),
            _build_pad_to_multiple(divisor=pad_multiple, fill=pad_fill),
            *augments,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(
                p=random_erasing_probability,
                scale=random_erasing_scale,
                ratio=random_erasing_ratio,
                value=pad_fill,
            ),
        ]
    )


def get_val_transforms(
    image_size: int = 256,
    *,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    pad_multiple: int = 8,
    pad_fill: float = 0,
) -> transforms.Compose:
    r"""get_val_transforms(image_size=256, ...) -> transforms.Compose

    Build the deterministic evaluation transformation pipeline.

    The pipeline preserves the full image content by:
    1. Resizing so the **height** becomes ``image_size`` while the width scales
       proportionally.
    2. Padding both height and width to the next multiple of ``pad_multiple``
       (default: 8) without cropping.
    3. Converting to tensor and normalizing.

    Args:
        image_size (int, optional): Target height after resizing. Default: ``256``
        mean (Sequence of float, optional): Per-channel normalization means. Default: :attr:`IMAGENET_MEAN`
        std (Sequence of float, optional): Per-channel normalization standard deviations. Default: :attr:`IMAGENET_STD`
        pad_multiple (int, optional): Pad height/width to the next multiple of this value. Default: ``8``
        pad_fill (int or float, optional): Fill value for padding. Default: ``0``

    Returns:
        transforms.Compose: Composed torchvision transformation pipeline for evaluation.

    Examples::

        >>> transform = get_val_transforms(image_size=256)
    """
    return transforms.Compose(
        [
            ResizePreserveAspectRatio(height=image_size),
            _build_pad_to_multiple(divisor=pad_multiple, fill=pad_fill),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "get_train_transforms",
    "get_val_transforms",
]
