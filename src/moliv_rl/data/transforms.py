from __future__ import annotations

from collections.abc import Sequence

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


def _validate_image_size(image_size: int) -> None:
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")


def _validate_resize_scale(resize_scale: float) -> None:
    if resize_scale < 1.0:
        raise ValueError(f"resize_scale must be >= 1.0, got {resize_scale}")


def _validate_normalization(
    mean: Sequence[float],
    std: Sequence[float],
) -> None:
    if len(mean) != len(std):
        raise ValueError(
            "mean and std must have the same number of channels: "
            f"mean={len(mean)}, std={len(std)}"
        )

    if len(mean) == 0:
        raise ValueError("mean and std must not be empty")

    if any(value <= 0 for value in std):
        raise ValueError("all std values must be positive")


def get_train_transforms(
    image_size: int = 64,
    *,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    resize_scale: float = 1.15,
    horizontal_flip_probability: float = 0.5,
) -> transforms.Compose:
    """Build the training image transformation pipeline."""
    _validate_image_size(image_size)
    _validate_resize_scale(resize_scale)
    _validate_normalization(mean, std)

    if not 0.0 <= horizontal_flip_probability <= 1.0:
        raise ValueError(
            "horizontal_flip_probability must be in [0.0, 1.0], "
            f"got {horizontal_flip_probability}"
        )

    resize_size = max(
        image_size,
        round(image_size * resize_scale),
    )

    return transforms.Compose(
        [
            transforms.Resize(
                resize_size,
                antialias=True,
            ),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(
                p=horizontal_flip_probability,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std,
            ),
        ]
    )


def get_val_transforms(
    image_size: int = 64,
    *,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    resize_scale: float = 1.15,
) -> transforms.Compose:
    """Build the deterministic validation transformation pipeline."""
    _validate_image_size(image_size)
    _validate_resize_scale(resize_scale)
    _validate_normalization(mean, std)

    resize_size = max(
        image_size,
        round(image_size * resize_scale),
    )

    return transforms.Compose(
        [
            transforms.Resize(
                resize_size,
                antialias=True,
            ),
            transforms.CenterCrop(
                image_size
            ),  # <- I highly prefer non Crop regularizations instead, like padding or even unassimetric resizing. But since this is only the starter template, I will keep it like that TEMPORARILY.
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std,
            ),
        ]
    )
