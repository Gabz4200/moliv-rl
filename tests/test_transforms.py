from __future__ import annotations

from typing import cast

import numpy as np
import pytest
import torch
from PIL import Image

from moliv_rl.data.transforms import (
    IMAGENET_MEAN,
    get_train_transforms,
    get_val_transforms,
)


class TestTransforms:
    """Behavioral tests for train and val image transformations."""

    def test_train_transforms_output_shape_and_type(self) -> None:
        transform = get_train_transforms(image_size=64)
        arr = np.random.randint(0, 255, (100, 120, 3), dtype=np.uint8)
        pil_img = Image.fromarray(arr)

        tensor_out = cast(torch.Tensor, transform(pil_img))

        assert isinstance(tensor_out, torch.Tensor)
        assert tensor_out.shape == (3, 64, 64)
        assert tensor_out.dtype == torch.float32

    def test_val_transforms_output_shape_and_determinism(self) -> None:
        transform = get_val_transforms(image_size=64)
        arr = np.random.randint(0, 255, (100, 120, 3), dtype=np.uint8)
        pil_img = Image.fromarray(arr)

        tensor_1 = cast(torch.Tensor, transform(pil_img))
        tensor_2 = cast(torch.Tensor, transform(pil_img))

        assert isinstance(tensor_1, torch.Tensor)
        assert tensor_1.shape == (3, 64, 64)
        assert torch.equal(tensor_1, tensor_2)

    def test_custom_single_channel_transforms(self) -> None:
        transform = get_val_transforms(
            image_size=32,
            mean=(0.5,),
            std=(0.5,),
        )
        arr = np.random.randint(0, 255, (40, 40), dtype=np.uint8)
        pil_img = Image.fromarray(arr)

        tensor_out = cast(torch.Tensor, transform(pil_img))
        assert tensor_out.shape == (1, 32, 32)

    @pytest.mark.parametrize("image_size", [0, -16])
    def test_invalid_image_size_raises(self, image_size: int) -> None:
        with pytest.raises(ValueError, match="image_size must be positive"):
            get_train_transforms(image_size=image_size)
        with pytest.raises(ValueError, match="image_size must be positive"):
            get_val_transforms(image_size=image_size)

    @pytest.mark.parametrize("resize_scale", [0.5, 0.99])
    def test_invalid_resize_scale_raises(self, resize_scale: float) -> None:
        with pytest.raises(ValueError, match="resize_scale must be >= 1.0"):
            get_train_transforms(resize_scale=resize_scale)
        with pytest.raises(ValueError, match="resize_scale must be >= 1.0"):
            get_val_transforms(resize_scale=resize_scale)

    def test_mean_std_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="mean and std must have the same number of channels"):
            get_train_transforms(mean=(0.5, 0.5), std=(0.5,))
        with pytest.raises(ValueError, match="mean and std must have the same number of channels"):
            get_val_transforms(mean=(0.5,), std=(0.5, 0.5))

    def test_empty_mean_std_raises(self) -> None:
        with pytest.raises(ValueError, match="mean and std must not be empty"):
            get_train_transforms(mean=(), std=())
        with pytest.raises(ValueError, match="mean and std must not be empty"):
            get_val_transforms(mean=(), std=())

    def test_non_positive_std_raises(self) -> None:
        with pytest.raises(ValueError, match="all std values must be positive"):
            get_train_transforms(mean=IMAGENET_MEAN, std=(0.2, 0.0, 0.2))
        with pytest.raises(ValueError, match="all std values must be positive"):
            get_val_transforms(mean=IMAGENET_MEAN, std=(0.2, -0.1, 0.2))

    @pytest.mark.parametrize("prob", [-0.1, 1.5])
    def test_invalid_flip_probability_raises(self, prob: float) -> None:
        with pytest.raises(ValueError, match="horizontal_flip_probability must be in"):
            get_train_transforms(horizontal_flip_probability=prob)
