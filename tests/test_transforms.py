from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from moliv_rl.data.transforms import (
    ResizePreserveAspectRatio,
    _build_pad_to_multiple,
    _next_multiple,
    get_train_transforms,
    get_val_transforms,
)


class TestNextMultiple:
    def test_exact_multiple(self) -> None:
        assert _next_multiple(16, 8) == 16

    def test_round_up(self) -> None:
        assert _next_multiple(17, 8) == 24

    def test_zero_value(self) -> None:
        assert _next_multiple(0, 8) == 0


class TestResizePreserveAspectRatio:
    def test_tensor_resize_height_only(self) -> None:
        transform = ResizePreserveAspectRatio(height=64)
        img = torch.rand(3, 100, 120)
        out = transform(img)
        assert out.shape[1] == 64
        assert out.shape[2] == 77  # round(120 * 64 / 100)

    def test_pil_resize_height_only(self) -> None:
        transform = ResizePreserveAspectRatio(height=64)
        img = Image.fromarray(np.random.randint(0, 255, (100, 120, 3), dtype=np.uint8))
        out = transform(img)
        assert out.size[1] == 64
        assert out.size[0] == 77

    def test_no_resize_when_height_matches(self) -> None:
        transform = ResizePreserveAspectRatio(height=64)
        img = torch.rand(3, 64, 120)
        out = transform(img)
        assert out.shape == img.shape


class TestPadToMultiple:
    def test_pad_tensor_to_multiple_of_8(self) -> None:
        transform = _build_pad_to_multiple(divisor=8, fill=0)
        img = torch.rand(3, 64, 77)
        out = transform(img)
        assert out.shape[1] == 64
        assert out.shape[2] == 80

    def test_pad_pil_to_multiple_of_8(self) -> None:
        transform = _build_pad_to_multiple(divisor=8, fill=0)
        img = Image.fromarray(np.random.randint(0, 255, (64, 77, 3), dtype=np.uint8))
        out = transform(img)
        assert out.size[1] == 64
        assert out.size[0] == 80

    def test_no_pad_when_already_multiple(self) -> None:
        transform = _build_pad_to_multiple(divisor=8, fill=0)
        img = torch.rand(3, 64, 80)
        out = transform(img)
        assert out.shape == img.shape


class TestTrainTransforms:
    def test_output_shape_and_type(self) -> None:
        transform = get_train_transforms(image_size=64)
        arr = np.random.randint(0, 255, (100, 120, 3), dtype=np.uint8)
        pil_img = Image.fromarray(arr)
        tensor_out = transform(pil_img)
        assert tensor_out.shape == (3, 64, 80)
        assert tensor_out.dtype == torch.float32


class TestValTransforms:
    def test_output_shape_and_determinism(self) -> None:
        transform = get_val_transforms(image_size=64)
        arr = np.random.randint(0, 255, (100, 120, 3), dtype=np.uint8)
        pil_img = Image.fromarray(arr)
        tensor_1 = transform(pil_img)
        tensor_2 = transform(pil_img)
        assert tensor_1.shape == (3, 64, 80)
        assert torch.equal(tensor_1, tensor_2)

    def test_custom_single_channel(self) -> None:
        transform = get_val_transforms(
            image_size=32,
            mean=(0.5,),
            std=(0.5,),
        )
        arr = np.random.randint(0, 255, (40, 40), dtype=np.uint8)
        pil_img = Image.fromarray(arr)
        tensor_out = transform(pil_img)
        assert tensor_out.shape == (1, 32, 32)
