import pytest
import torch

from moliv_rl.models.mobilenetv3 import (
    mobilenetv3_large,
    mobilenetv3_small,
)


def test_mobilenetv3_large_forward():
    model = mobilenetv3_large(num_classes=1000)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1000)


def test_mobilenetv3_small_forward():
    model = mobilenetv3_small(num_classes=1000)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1000)


def test_mobilenetv3_backward():
    model = mobilenetv3_large(num_classes=10)
    model.train()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    loss = out.sum()
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"


@pytest.mark.parametrize("width_mult", [0.75, 1.0, 1.25])
def test_mobilenetv3_width_mult(width_mult):
    m_large = mobilenetv3_large(num_classes=100, width_mult=width_mult)
    m_small = mobilenetv3_small(num_classes=100, width_mult=width_mult)

    x = torch.randn(1, 3, 224, 224)
    out_l = m_large(x)
    out_s = m_small(x)

    assert out_l.shape == (1, 100)
    assert out_s.shape == (1, 100)
