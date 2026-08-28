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


def get_train_transforms(
    image_size: int = 64,
    *,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    resize_scale: float = 1.15,
    horizontal_flip_probability: float = 0.5,
) -> transforms.Compose:
    r"""get_train_transforms(image_size=64, *, mean=IMAGENET_MEAN, std=IMAGENET_STD, resize_scale=1.15, horizontal_flip_probability=0.5) -> transforms.Compose

    Build the standard data augmentation and normalization pipeline for training.

    Args:
        image_size (int, optional): Target output image spatial dimension :math:`(H = W)`. Default: ``64``
        mean (Sequence of float, optional): Per-channel normalization means. Default: :attr:`IMAGENET_MEAN`
        std (Sequence of float, optional): Per-channel normalization standard deviations. Default: :attr:`IMAGENET_STD`
        resize_scale (float, optional): Scale factor to enlarge image prior to random crop. Default: ``1.15``
        horizontal_flip_probability (float, optional): Probability of random horizontal flip. Default: ``0.5``

    Returns:
        transforms.Compose: Composed torchvision transformation pipeline.

    Examples::

        >>> transform = get_train_transforms(image_size=224)
    """
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
    r"""get_val_transforms(image_size=64, *, mean=IMAGENET_MEAN, std=IMAGENET_STD, resize_scale=1.15) -> transforms.Compose

    Build the deterministic evaluation transformation pipeline.

    Args:
        image_size (int, optional): Target output image spatial dimension :math:`(H = W)`. Default: ``64``
        mean (Sequence of float, optional): Per-channel normalization means. Default: :attr:`IMAGENET_MEAN`
        std (Sequence of float, optional): Per-channel normalization standard deviations. Default: :attr:`IMAGENET_STD`
        resize_scale (float, optional): Scale factor to resize image prior to center crop. Default: ``1.15``

    Returns:
        transforms.Compose: Composed torchvision transformation pipeline for evaluation.

    Examples::

        >>> transform = get_val_transforms(image_size=224)
    """
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


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "get_train_transforms",
    "get_val_transforms",
]
