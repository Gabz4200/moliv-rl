from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# Full MobileNetV3 implementation based on the paper "Searching for MobileNetV3" (arXiv:1905.02244).
# For comparison and inspiration.


class HardSwish(nn.Module):
    r"""HardSwish(inplace=False)

    Applies the hardswish function element-wise as defined in `Searching for MobileNetV3`_.

    .. math::
        \text{Hardswish}(x) = x \cdot \frac{\text{ReLU6}(x + 3)}{6}

    Args:
        inplace (bool, optional): Can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = HardSwish()
        >>> input = torch.randn(2)
        >>> output = m(input)

    .. _`Searching for MobileNetV3`:
        https://arxiv.org/abs/1905.02244
    """

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            x.mul_(F.relu6(x.add(3.0), inplace=True) / 6.0)
            if self.inplace
            else x * F.relu6(x + 3.0) / 6.0
        )


class HardSigmoid(nn.Module):
    r"""HardSigmoid(inplace=False)

    Applies the hardsigmoid function element-wise as defined in `Searching for MobileNetV3`_.

    .. math::
        \text{Hardsigmoid}(x) = \frac{\text{ReLU6}(x + 3)}{6}

    Args:
        inplace (bool, optional): Can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = HardSigmoid()
        >>> input = torch.randn(2)
        >>> output = m(input)

    .. _`Searching for MobileNetV3`:
        https://arxiv.org/abs/1905.02244
    """

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            F.relu6(x.add(3.0), inplace=True).div_(6.0)
            if self.inplace
            else F.relu6(x + 3.0) / 6.0
        )


ACT_MAP: dict[str, type[nn.Module]] = {
    "RE": nn.ReLU,
    "HS": HardSwish,
}


def make_act(name: str, inplace: bool = False) -> nn.Module:
    r"""make_act(name, inplace=False) -> nn.Module

    Instantiate activation layer by paper code name ('RE' for ReLU, 'HS' for HardSwish).

    Args:
        name (str): Activation identifier ('RE' or 'HS').
        inplace (bool, optional): In-place activation execution. Default: ``False``

    Returns:
        nn.Module: Instantiated activation layer.
    """
    return ACT_MAP[name](inplace=inplace)


class SEModule(nn.Module):
    r"""SEModule(channels, reduction=4)

    Squeeze-and-Excitation attention module for MobileNetV3 architectures.

    Applies global average pooling followed by a two-layer 1x1 convolution
    bottleneck with ReLU and HardSigmoid gating to recalibrate channel features.

    Args:
        channels (int): Number of input/output channels.
        reduction (int, optional): Squeeze reduction divisor. Default: ``4``

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)`

    Examples::

        >>> se = SEModule(channels=64, reduction=4)
        >>> x = torch.randn(2, 64, 16, 16)
        >>> out = se(x)
        >>> out.shape
        torch.Size([2, 64, 16, 16])
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = _make_divisible(channels / reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.expand = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.hsigmoid = HardSigmoid(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        # Global average pooling -> [B, C, 1, 1]
        y = self.pool(x)

        # Channel reduction (squeeze) -> [B, hidden, 1, 1]
        y = self.reduce(y)

        # Non-linearity after squeeze
        y = self.relu(y)

        # Channel expansion (excite) back to C -> [B, C, 1, 1]
        y = self.expand(y)

        # Gate in [0, 1] via hard-sigmoid
        y = self.hsigmoid(y)

        # Scale original features by gate
        return x * y


class InvertedResidual(nn.Module):
    r"""InvertedResidual(in_channels, out_channels, kernel, stride, exp_size, use_se, nl)

    MobileNetV3 inverted residual bottleneck block with depthwise convolution.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel (int): Depthwise convolution kernel size (3 or 5).
        stride (int): Stride for spatial downsampling (1 or 2).
        exp_size (int): Number of expanded intermediate channels.
        use_se (bool): If ``True``, applies Squeeze-and-Excitation attention.
        nl (str): Activation type identifier ('RE' or 'HS').

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        stride: int,
        exp_size: int,
        use_se: bool,
        nl: str,
    ):
        super().__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers: list[nn.Module] = []

        # Expand: 1x1 pointwise conv to increase channels (if needed)
        if exp_size != in_channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, exp_size, kernel_size=1, bias=False),
                    nn.BatchNorm2d(exp_size),
                    make_act(nl, inplace=True),
                ]
            )

        # Depthwise: kxk spatial filtering with groups=exp_size
        layers.extend(
            [
                nn.Conv2d(
                    exp_size,
                    exp_size,
                    kernel_size=kernel,
                    stride=stride,
                    padding=kernel // 2,
                    groups=exp_size,
                    bias=False,
                ),
                nn.BatchNorm2d(exp_size),
                make_act(nl, inplace=True),
            ]
        )

        self.conv = nn.Sequential(*layers)

        # SE: optional squeeze-and-excite attention on expanded features
        self.se = SEModule(exp_size, reduction=4) if use_se else nn.Identity()

        # Project: 1x1 pointwise conv to reduce back to out_channels (no activation)
        self.project = nn.Sequential(
            nn.Conv2d(exp_size, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expand + depthwise conv + activation
        out = self.conv(x)

        # Apply SE attention (or identity)
        out = self.se(out)

        # Project back to output channels
        out = self.project(out)

        # Add residual connection if shapes match and stride == 1
        return x + out if self.use_res_connect else out


# Configs extracted from Tables 1 & 2 of the paper MobileNetV3 [https://arxiv.org/pdf/1905.02244](https://arxiv.org/pdf/1905.02244)
# each tuple is (kernel, exp_size, out_channels, use_se, nl, stride)
CFG_LARGE: list[tuple[int, int, int, bool, str, int]] = [
    # (k,  exp, out, SE,  NL,  s)
    (3, 16, 16, False, "RE", 1),
    (3, 64, 24, False, "RE", 2),
    (3, 72, 24, False, "RE", 1),
    (5, 72, 40, True, "RE", 2),
    (5, 120, 40, True, "RE", 1),
    (5, 120, 40, True, "RE", 1),
    (3, 240, 80, False, "HS", 2),
    (3, 200, 80, False, "HS", 1),
    (3, 184, 80, False, "HS", 1),
    (3, 184, 80, False, "HS", 1),
    (3, 480, 112, True, "HS", 1),
    (3, 672, 112, True, "HS", 1),
    (5, 672, 160, True, "HS", 2),
    (5, 960, 160, True, "HS", 1),
    (5, 960, 160, True, "HS", 1),
]


CFG_SMALL: list[tuple[int, int, int, bool, str, int]] = [
    # (k,  exp, out, SE,  NL,  s)
    (3, 16, 16, True, "RE", 2),
    (3, 72, 24, False, "RE", 2),
    (3, 88, 24, False, "RE", 1),
    (5, 96, 40, True, "HS", 2),
    (5, 240, 40, True, "HS", 1),
    (5, 240, 40, True, "HS", 1),
    (5, 120, 48, True, "HS", 1),
    (5, 144, 48, True, "HS", 1),
    (5, 288, 96, True, "HS", 2),
    (5, 576, 96, True, "HS", 1),
    (5, 576, 96, True, "HS", 1),
]


def _make_divisible(v: float, divisor: int = 8, min_value: int | None = None) -> int:
    r"""_make_divisible(v, divisor=8, min_value=None) -> int

    Round channel counts to the nearest multiple of divisor while avoiding >10% down-rounding.

    Args:
        v (float): Raw channel count value.
        divisor (int, optional): Divisibility divisor. Default: ``8``
        min_value (int, optional): Minimum bound. Default: ``divisor``

    Returns:
        int: Divisible channel count.
    """
    min_val = divisor if min_value is None else min_value
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Ensure the round down does not go down by more than 10%.
    return new_v + divisor if new_v < 0.9 * v else new_v


class MobileNetV3(nn.Module):
    r"""MobileNetV3(mode='large', num_classes=1000, width_mult=1.0, input_size=224)

    MobileNetV3 architecture following `Searching for MobileNetV3`_.

    Args:
        mode (str, optional): Architecture configuration variant ('large' or 'small'). Default: ``'large'``
        num_classes (int, optional): Number of classification target classes. Default: ``1000``
        width_mult (float, optional): Channel width multiplier. Default: ``1.0``
        input_size (int, optional): Expected spatial dimension of input images. Default: ``224``

    Shape:
        - Input: :math:`(N, 3, H, W)`
        - Output: :math:`(N, \text{num\_classes})`

    Examples::

        >>> model = MobileNetV3(mode='small', num_classes=10)
        >>> x = torch.randn(2, 3, 224, 224)
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([2, 10])

    .. _`Searching for MobileNetV3`:
        https://arxiv.org/abs/1905.02244
    """

    def __init__(
        self,
        mode: str = "large",
        num_classes: int = 1000,
        width_mult: float = 1.0,
        input_size: int = 224,
    ):
        super().__init__()
        self.mode = mode
        self.input_size = input_size
        self.num_classes = num_classes

        cfg, last_exp, final_exp = {
            "large": (CFG_LARGE, 960, 1280),
            "small": (CFG_SMALL, 576, 1024),
        }[mode]

        # Stem: 3x3 conv, stride 2, 16 channels, h-swish (Section 5.1)
        stem_channels = _make_divisible(16 * width_mult, 8)
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            HardSwish(inplace=True),
        )

        # bneck blocks
        in_ch = stem_channels
        layers: list[nn.Module] = []
        for kernel, exp_size, out_channels, use_se, nl, stride in cfg:
            exp_size_scaled = _make_divisible(exp_size * width_mult, 8)
            out_channels_scaled = _make_divisible(out_channels * width_mult, 8)
            layers.append(
                InvertedResidual(
                    in_channels=in_ch,
                    out_channels=out_channels_scaled,
                    kernel=kernel,
                    stride=stride,
                    exp_size=exp_size_scaled,
                    use_se=use_se,
                    nl=nl,
                )
            )
            in_ch = out_channels_scaled

        # Expand channels before pooling (Table 1 row 16 / Table 2 row 12): 1x1 conv + BN + h-swish
        last_exp_scaled = _make_divisible(last_exp * width_mult, 8)
        layers.append(
            nn.Sequential(
                nn.Conv2d(in_ch, last_exp_scaled, kernel_size=1, bias=False),
                nn.BatchNorm2d(last_exp_scaled),
                HardSwish(inplace=True),
            )
        )
        self.features = nn.Sequential(*layers)

        # Efficient last stage (Section 5.1, Fig. 5):
        # - 1x1 conv (no BN) to final_exp after pooling
        # - linear classifier
        final_exp_scaled = _make_divisible(final_exp * width_mult, 8)

        self.pool = nn.AdaptiveAvgPool2d(1)

        # Final conv without BN, with h-swish (NBN = no batch norm in the paper)
        self.conv_head = nn.Sequential(
            nn.Conv2d(last_exp_scaled, final_exp_scaled, kernel_size=1, bias=True),
            HardSwish(inplace=True),
        )

        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(final_exp_scaled, num_classes)

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem: initial 3x3 conv + BN + h-swish, downsamples by 2
        x = self.stem(x)

        # Body: sequence of inverted residual blocks (bneck) + 1x1 conv expansion
        x = self.features(x)

        # Global average pooling: [B, C, H, W] -> [B, C, 1, 1]
        x = self.pool(x)

        # Head: 1x1 conv (no BN) + h-swish to expand channels before classifier
        x = self.conv_head(x)

        # Flatten spatial dims: [B, C, 1, 1] -> [B, C]
        x = torch.flatten(x, 1)

        # Dropout before final linear layer
        x = self.dropout(x)

        # Linear classifier to class logits
        return self.classifier(x)

    def _initialize_weights(self) -> None:
        # Kaiming normal initialization standard for MobileNetV3 architectures.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)


def mobilenetv3_large(
    num_classes: int = 1000,
    width_mult: float = 1.0,
    input_size: int = 224,
) -> MobileNetV3:
    r"""mobilenetv3_large(num_classes=1000, width_mult=1.0, input_size=224) -> MobileNetV3

    Instantiate MobileNetV3-Large architecture from Table 1 of `Searching for MobileNetV3`_.

    Args:
        num_classes (int, optional): Number of classification target classes. Default: ``1000``
        width_mult (float, optional): Width multiplier scaling all layer channels. Default: ``1.0``
        input_size (int, optional): Expected input image spatial resolution. Default: ``224``

    Returns:
        MobileNetV3: Configured MobileNetV3-Large model.

    .. _`Searching for MobileNetV3`:
        https://arxiv.org/abs/1905.02244
    """
    return MobileNetV3(
        mode="large",
        num_classes=num_classes,
        width_mult=width_mult,
        input_size=input_size,
    )


def mobilenetv3_small(
    num_classes: int = 1000,
    width_mult: float = 1.0,
    input_size: int = 224,
) -> MobileNetV3:
    r"""mobilenetv3_small(num_classes=1000, width_mult=1.0, input_size=224) -> MobileNetV3

    Instantiate MobileNetV3-Small architecture from Table 2 of `Searching for MobileNetV3`_.

    Args:
        num_classes (int, optional): Number of classification target classes. Default: ``1000``
        width_mult (float, optional): Width multiplier scaling all layer channels. Default: ``1.0``
        input_size (int, optional): Expected input image spatial resolution. Default: ``224``

    Returns:
        MobileNetV3: Configured MobileNetV3-Small model.

    .. _`Searching for MobileNetV3`:
        https://arxiv.org/abs/1905.02244
    """
    return MobileNetV3(
        mode="small",
        num_classes=num_classes,
        width_mult=width_mult,
        input_size=input_size,
    )


__all__ = [
    "MobileNetV3",
    "mobilenetv3_large",
    "mobilenetv3_small",
]
