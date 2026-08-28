from __future__ import annotations

from typing import cast

import numpy as np
import torch
from PIL import Image

from moliv_rl.data.transforms import (
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
