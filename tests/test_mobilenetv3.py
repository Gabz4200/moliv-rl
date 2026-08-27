from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from moliv_rl.models.mobilenetv3 import (
    HardSigmoid,
    HardSwish,
    InvertedResidual,
    MobileNetV3,
    SEModule,
    make_act,
    mobilenetv3_large,
    mobilenetv3_small,
)


class TestMobileNetV3Components:
    """Behavioral tests for MobileNetV3 building blocks."""

    def test_hard_swish_contract_and_inplace(self) -> None:
        hs_out_of_place = HardSwish(inplace=False)
        hs_in_place = HardSwish(inplace=True)

        x = torch.tensor([-4.0, -3.0, 0.0, 3.0, 4.0])
        expected = x * F.relu6(x + 3.0) / 6.0

        out_oop = hs_out_of_place(x.clone())
        out_ip = hs_in_place(x.clone())

        assert torch.allclose(out_oop, expected, atol=1e-6)
        assert torch.allclose(out_ip, expected, atol=1e-6)

    def test_hard_sigmoid_contract_and_inplace(self) -> None:
        hsig_oop = HardSigmoid(inplace=False)
        hsig_ip = HardSigmoid(inplace=True)

        x = torch.tensor([-4.0, -3.0, 0.0, 3.0, 4.0])
        expected = F.relu6(x + 3.0) / 6.0

        out_oop = hsig_oop(x.clone())
        out_ip = hsig_ip(x.clone())

        assert torch.allclose(out_oop, expected, atol=1e-6)
        assert torch.allclose(out_ip, expected, atol=1e-6)
        assert (out_oop >= 0.0).all() and (out_oop <= 1.0).all()

    def test_make_act_factory(self) -> None:
        assert isinstance(make_act("RE"), nn.ReLU)
        assert isinstance(make_act("HS"), HardSwish)
        with pytest.raises(ValueError, match="Unknown activation name"):
            make_act("GELU")

    def test_se_module_contract_and_gradient(self) -> None:
        se = SEModule(channels=32, reduction=4)
        x = torch.randn(2, 32, 14, 14, requires_grad=True)

        out = se(x)
        assert out.shape == (2, 32, 14, 14)

        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("kernel", [3, 5])
    @pytest.mark.parametrize("use_se", [True, False])
    @pytest.mark.parametrize("nl", ["RE", "HS"])
    def test_inverted_residual_shape_and_gradient(
        self,
        stride: int,
        kernel: int,
        use_se: bool,
        nl: str,
    ) -> None:
        in_channels = 24
        out_channels = 24 if stride == 1 else 48
        exp_size = 72

        bneck = InvertedResidual(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=kernel,
            stride=stride,
            exp_size=exp_size,
            use_se=use_se,
            nl=nl,
        )

        x = torch.randn(2, in_channels, 16, 16, requires_grad=True)
        out = bneck(x)

        expected_spatial = 16 if stride == 1 else 8
        assert out.shape == (2, out_channels, expected_spatial, expected_spatial)

        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_inverted_residual_invalid_params_raise(self) -> None:
        with pytest.raises(ValueError, match="stride must be 1 or 2"):
            InvertedResidual(
                in_channels=16,
                out_channels=16,
                kernel=3,
                stride=3,
                exp_size=32,
                use_se=False,
                nl="RE",
            )

        with pytest.raises(ValueError, match="kernel must be 3 or 5"):
            InvertedResidual(
                in_channels=16,
                out_channels=16,
                kernel=7,
                stride=1,
                exp_size=32,
                use_se=False,
                nl="RE",
            )


class TestMobileNetV3Model:
    """Behavioral tests for MobileNetV3 architectures."""

    @pytest.mark.parametrize("num_classes", [10, 1000])
    def test_mobilenetv3_large_forward_and_eval_determinism(
        self,
        num_classes: int,
    ) -> None:
        model = mobilenetv3_large(num_classes=num_classes)
        model.eval()

        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        assert out1.shape == (2, num_classes)
        assert torch.equal(out1, out2)

    @pytest.mark.parametrize("num_classes", [10, 1000])
    def test_mobilenetv3_small_forward_and_eval_determinism(
        self,
        num_classes: int,
    ) -> None:
        model = mobilenetv3_small(num_classes=num_classes)
        model.eval()

        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        assert out1.shape == (2, num_classes)
        assert torch.equal(out1, out2)

    def test_mobilenetv3_backward_gradients_all_params(self) -> None:
        model = mobilenetv3_small(num_classes=10)
        model.train()

        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        out.sum().backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient missing for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    @pytest.mark.parametrize("width_mult", [0.75, 1.0, 1.25])
    def test_mobilenetv3_width_multiplier(self, width_mult: float) -> None:
        m_large = mobilenetv3_large(num_classes=50, width_mult=width_mult)
        m_small = mobilenetv3_small(num_classes=50, width_mult=width_mult)

        x = torch.randn(1, 3, 64, 64)
        out_l = m_large(x)
        out_s = m_small(x)

        assert out_l.shape == (1, 50)
        assert out_s.shape == (1, 50)

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="mode must be 'large' or 'small'"):
            MobileNetV3(mode="medium")
