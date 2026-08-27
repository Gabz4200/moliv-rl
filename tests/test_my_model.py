from __future__ import annotations

import pytest
import torch

from moliv_rl.models import (
    ClassificationModel,
    LiVConv2D,
    MLPConv2D,
    MyBlock,
    MyModel,
    SwiGluConv2D,
    get_model,
)


class TestConvolutionBlocks:
    """Behavioral tests for custom 2D convolution building blocks."""

    @pytest.mark.parametrize("block_cls", [LiVConv2D, MLPConv2D, SwiGluConv2D])
    @pytest.mark.parametrize("use_norm", [True, False])
    def test_block_same_channels_residual_and_gradient(
        self,
        block_cls: type[LiVConv2D | MLPConv2D | SwiGluConv2D],
        use_norm: bool,
    ) -> None:
        block = block_cls(
            in_channels=16,
            hidden_channels=32,
            out_channels=16,
            use_norm=use_norm,
            use_residual=True,
        )
        assert block.use_residual is True

        x = torch.randn(2, 16, 24, 24, requires_grad=True)
        out = block(x)

        assert out.shape == (2, 16, 24, 24)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    @pytest.mark.parametrize("block_cls", [LiVConv2D, MLPConv2D, SwiGluConv2D])
    def test_block_channel_change_no_residual(
        self,
        block_cls: type[LiVConv2D | MLPConv2D | SwiGluConv2D],
    ) -> None:
        block = block_cls(
            in_channels=16,
            hidden_channels=32,
            out_channels=32,
            use_residual=True,
        )
        # Residual must be disabled when in_channels != out_channels
        assert block.use_residual is False

        x = torch.randn(2, 16, 20, 20)
        out = block(x)
        assert out.shape == (2, 32, 20, 20)

    @pytest.mark.parametrize("block_cls", [LiVConv2D, MLPConv2D, SwiGluConv2D])
    def test_block_eval_determinism(
        self,
        block_cls: type[LiVConv2D | MLPConv2D | SwiGluConv2D],
    ) -> None:
        block = block_cls(
            in_channels=16,
            hidden_channels=32,
            out_channels=16,
            hidden_dropout=0.5,
        )
        block.eval()

        x = torch.randn(2, 16, 16, 16)
        with torch.no_grad():
            out1 = block(x)
            out2 = block(x)

        assert torch.equal(out1, out2)

    @pytest.mark.parametrize("block_cls", [LiVConv2D, MLPConv2D, SwiGluConv2D])
    def test_block_channels_last_compatibility(
        self,
        block_cls: type[LiVConv2D | MLPConv2D | SwiGluConv2D],
    ) -> None:
        block = block_cls(in_channels=16, hidden_channels=32, out_channels=16)
        block.to(memory_format=torch.channels_last)  # type: ignore[call-arg,no-matching-overload]

        x = torch.randn(2, 16, 16, 16).to(memory_format=torch.channels_last)
        out = block(x)
        assert out.shape == (2, 16, 16, 16)

    def test_swiglu_fused_depthwise_groups(self) -> None:
        block = SwiGluConv2D(in_channels=16, hidden_channels=32, out_channels=16)
        # Hidden channels is 32 -> input conv produces 64 channels
        # dw_conv operates on 64 channels with groups=64
        assert block.dw_conv.in_channels == 64
        assert block.dw_conv.out_channels == 64
        assert block.dw_conv.groups == 64

    @pytest.mark.parametrize("include_swiglu", [True, False])
    def test_my_block_contract(self, include_swiglu: bool) -> None:
        block = MyBlock(
            in_channels=16,
            hidden_channels=32,
            out_channels=16,
            include_swiglu=include_swiglu,
        )
        x = torch.randn(2, 16, 16, 16, requires_grad=True)
        out = block(x)

        assert out.shape == (2, 16, 16, 16)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestVisionModels:
    """Behavioral tests for MyModel and ClassificationModel."""

    @pytest.mark.parametrize("patch_size", [4, 8])
    def test_my_model_patch_downsampling_and_eval_determinism(
        self,
        patch_size: int,
    ) -> None:
        model = MyModel(
            block_dims=[16, 32],
            in_channels=3,
            out_channels=64,
            patch_size=patch_size,
            dropout=0.2,
        )
        model.eval()

        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        expected_spatial = 32 // patch_size
        assert out1.shape == (2, 64, expected_spatial, expected_spatial)
        assert torch.equal(out1, out2)

    def test_my_model_gradient_flow_all_params(self) -> None:
        model = MyModel(
            block_dims=[16, 32],
            in_channels=3,
            out_channels=64,
            patch_size=8,
        )
        model.train()

        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        out.sum().backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient missing for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    @pytest.mark.parametrize("num_classes", [5, 10])
    def test_classification_model_shape_and_eval_determinism(
        self,
        num_classes: int,
    ) -> None:
        model = ClassificationModel(
            block_dims=[16, 32],
            in_channels=3,
            out_channels=64,
            patch_size=8,
            dropout=0.2,
            num_classes=num_classes,
        )
        model.eval()

        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        assert out1.shape == (2, num_classes)
        assert torch.equal(out1, out2)

    def test_classification_model_gradient_flow_all_params(self) -> None:
        model = ClassificationModel(
            block_dims=[16, 32],
            in_channels=3,
            out_channels=64,
            patch_size=8,
            num_classes=10,
        )
        model.train()

        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        out.sum().backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient missing for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    def test_classification_model_channels_last(self) -> None:
        model = ClassificationModel(
            block_dims=[16, 32],
            in_channels=3,
            out_channels=64,
            patch_size=8,
            num_classes=10,
        )
        model.to(memory_format=torch.channels_last)  # type: ignore[call-arg,no-matching-overload]

        x = torch.randn(2, 3, 32, 32).to(memory_format=torch.channels_last)
        out = model(x)
        assert out.shape == (2, 10)

    def test_get_model_factory_dispatch(self) -> None:
        m1 = get_model(
            "classification_model",
            optimize=False,
            block_dims=[16, 32],
            in_channels=3,
            out_channels=64,
            patch_size=8,
            num_classes=10,
        )
        assert isinstance(m1, ClassificationModel)

        m2 = get_model(
            "my_model",
            optimize=False,
            block_dims=[16, 32],
            in_channels=3,
            out_channels=64,
            patch_size=8,
        )
        assert isinstance(m2, MyModel)

    def test_get_model_invalid_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown model name"):
            get_model("unsupported_model", optimize=False)

    @pytest.mark.parametrize("block_cls", [LiVConv2D, MLPConv2D, SwiGluConv2D])
    def test_conv_block_invalid_parameters_raise(
        self,
        block_cls: type[LiVConv2D | MLPConv2D | SwiGluConv2D],
    ) -> None:
        with pytest.raises(ValueError, match="in_channels must be positive"):
            block_cls(in_channels=0, hidden_channels=16, out_channels=16)

        with pytest.raises(ValueError, match="out_channels must be positive"):
            block_cls(in_channels=16, hidden_channels=16, out_channels=0)

        with pytest.raises(ValueError, match="kernel_size must be an odd positive"):
            block_cls(in_channels=16, hidden_channels=16, out_channels=16, kernel_size=2)

        with pytest.raises(ValueError, match="hidden_dropout must be in"):
            block_cls(in_channels=16, hidden_channels=16, out_channels=16, hidden_dropout=1.5)

    def test_my_model_and_classifier_invalid_parameters_raise(self) -> None:
        with pytest.raises(ValueError, match="block_dims must contain at least 2"):
            MyModel(block_dims=[16])

        with pytest.raises(ValueError, match="in_channels must be positive"):
            MyModel(block_dims=[16, 32], in_channels=0)

        with pytest.raises(ValueError, match="out_channels must be positive"):
            MyModel(block_dims=[16, 32], out_channels=0)

        with pytest.raises(ValueError, match="patch_size must be positive"):
            MyModel(block_dims=[16, 32], patch_size=0)

        with pytest.raises(ValueError, match="dropout must be in"):
            MyModel(block_dims=[16, 32], dropout=-0.1)

        with pytest.raises(ValueError, match="num_classes must be positive"):
            ClassificationModel(block_dims=[16, 32], num_classes=0)
