import pytest
import torch

from moliv_rl.models import LiVConv2D, MLPConv2D, SwiGluConv2D


def test_liv_conv_forward():
    block = LiVConv2D(in_channels=16, hidden_channels=32, out_channels=16)
    block.eval()
    x = torch.randn(2, 16, 32, 32)
    with torch.no_grad():
        out = block(x)
    assert out.shape == (2, 16, 32, 32)


def test_liv_conv_channel_change():
    block = LiVConv2D(in_channels=16, hidden_channels=32, out_channels=64, use_residual=True)
    block.eval()
    x = torch.randn(2, 16, 32, 32)
    with torch.no_grad():
        out = block(x)
    assert out.shape == (2, 64, 32, 32)


def test_mlp_conv2d_forward():
    block = MLPConv2D(in_channels=16, hidden_channels=32, out_channels=16)
    block.eval()
    x = torch.randn(2, 16, 32, 32)
    with torch.no_grad():
        out = block(x)
    assert out.shape == (2, 16, 32, 32)


def test_swiglu_conv2d_forward():
    block = SwiGluConv2D(in_channels=16, hidden_channels=32, out_channels=16)
    block.eval()
    x = torch.randn(2, 16, 32, 32)
    with torch.no_grad():
        out = block(x)
    assert out.shape == (2, 16, 32, 32)


def test_swiglu_conv2d_channel_change():
    block = SwiGluConv2D(in_channels=16, hidden_channels=32, out_channels=64, use_residual=True)
    block.eval()
    x = torch.randn(2, 16, 32, 32)
    with torch.no_grad():
        out = block(x)
    assert out.shape == (2, 64, 32, 32)


@pytest.mark.parametrize("use_norm", [True, False])
def test_all_conv_blocks_backward(use_norm):
    liv = LiVConv2D(in_channels=8, hidden_channels=16, out_channels=8, use_norm=use_norm)
    mlp = MLPConv2D(in_channels=8, hidden_channels=16, out_channels=8, use_norm=use_norm)
    swiglu = SwiGluConv2D(in_channels=8, hidden_channels=16, out_channels=8, use_norm=use_norm)

    x = torch.randn(2, 8, 16, 16)
    out_liv = liv(x)
    out_mlp = mlp(x)
    out_swiglu = swiglu(x)

    (out_liv.sum() + out_mlp.sum() + out_swiglu.sum()).backward()

    for p in liv.parameters():
        if p.requires_grad:
            assert p.grad is not None
    for p in mlp.parameters():
        if p.requires_grad:
            assert p.grad is not None
    for p in swiglu.parameters():
        if p.requires_grad:
            assert p.grad is not None
