from __future__ import annotations

import pytest
import torch

from moliv_rl import (
    ClassificationModel,
    LiVConv2D,
    MLPConv2D,
    MyBlock,
    MyModel,
    MyVideoModel,
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

        m3 = get_model(
            "my_video_model",
            optimize=False,
            block_dims=[16, 32],
            in_channels=3,
            out_channels=64,
            patch_size=8,
        )
        assert isinstance(m3, MyVideoModel)

    def test_get_model_invalid_name_raises(self) -> None:
        with pytest.raises(KeyError):
            get_model("unsupported_model", optimize=False)

    def test_get_model_fullgraph_parameter(self) -> None:
        model = get_model(
            "classification_model",
            optimize=False,
            fullgraph=False,
            block_dims=[16, 32],
            in_channels=3,
            out_channels=64,
            patch_size=8,
            num_classes=10,
        )
        assert isinstance(model, ClassificationModel)


class TestVideoModels:
    """Behavioral tests for MyVideoModel."""

    def test_video_model_forward_shape_and_gradient(self) -> None:
        model = MyVideoModel(
            block_dims=[16, 32],
            in_channels=3,
            out_channels=32,
            patch_size=8,
            conv_kernel_size=3,
        )
        x = torch.randn(2, 4, 3, 16, 16, requires_grad=True)
        out = model(x)
        assert out.shape == (2, 4, 32, 2, 2)

        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_video_model_forward_step_matches_batched_forward(self) -> None:
        model = MyVideoModel(
            block_dims=[16, 32],
            in_channels=3,
            out_channels=32,
            patch_size=8,
            conv_kernel_size=3,
        )
        model.eval()

        torch.manual_seed(42069)
        x = torch.randn(2, 3, 3, 16, 16)
        with torch.no_grad():
            batched_out = model(x)

            step_outs: list[torch.Tensor] = []
            cache = None
            for t in range(x.shape[1]):
                y_t, cache = model.forward_step(x[:, t : t + 1], cache=cache)
                step_outs.append(y_t)

            sequential_out = torch.cat(step_outs, dim=1)
            assert torch.allclose(batched_out, sequential_out, atol=1e-4)

    def test_models_init_exports(self) -> None:
        import moliv_rl
        import moliv_rl.models

        assert hasattr(moliv_rl, "MyVideoModel")
        assert hasattr(moliv_rl, "MyBlock")
        assert hasattr(moliv_rl, "MODEL_REGISTRY")
        assert hasattr(moliv_rl.models, "MyVideoModel")
        assert hasattr(moliv_rl.models, "MyBlock")
        assert hasattr(moliv_rl.models, "MODEL_REGISTRY")
        assert "MyVideoModel" in moliv_rl.__all__
        assert "MyBlock" in moliv_rl.__all__
        assert "MODEL_REGISTRY" in moliv_rl.__all__
        assert "MyVideoModel" in moliv_rl.models.__all__
        assert "MyBlock" in moliv_rl.models.__all__
        assert "MODEL_REGISTRY" in moliv_rl.models.__all__
