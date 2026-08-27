from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

# Full MobileNetV3 implementation based on the paper "Searching for MobileNetV3" (arXiv:1905.02244).
# For comparison and inspiration.


class HardSwish(nn.Module):
    """h-swish[x] = x * ReLU6(x + 3) / 6"""

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply h-swish: x * ReLU6(x + 3) / 6
        if self.inplace:
            return x.mul_(F.relu6(x.add(3.0), inplace=True) / 6.0)
        return x * F.relu6(x + 3.0) / 6.0


class HardSigmoid(nn.Module):
    """h-sigmoid[x] = ReLU6(x + 3) / 6"""

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply h-sigmoid: ReLU6(x + 3) / 6
        if self.inplace:
            return F.relu6(x.add(3.0), inplace=True).div_(6.0)
        return F.relu6(x + 3.0) / 6.0


def make_act(name: str, inplace: bool = False) -> nn.Module:
    """
    Activation names as in the paper:
      'RE' -> ReLU
      'HS' -> HardSwish
    """
    if name == "RE":
        return nn.ReLU(inplace=inplace)
    if name == "HS":
        return HardSwish(inplace=inplace)
    raise ValueError(f"Unknown activation name: {name}")


class SEModule(nn.Module):
    """
    Squeeze-and-Excite as used in MobileNetV3:
    - Global average pool
    - 1x1 conv (reduce) -> ReLU
    - 1x1 conv (expand) -> hard-sigmoid
    - scale = x * gate
    Reduction is fixed to 1/4 of the expansion channels (Section 5.3).
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
    """
    MobileNetV3 bottleneck block (bneck) as in Tables 1 & 2 and Fig. 4.


    Structure:
      - 1x1 expand -> BN -> NL
      - kxk depthwise -> BN -> NL
      - SE (optional)
      - 1x1 project -> BN (no NL)
      - residual if stride==1 and in_channels == out_channels
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
        if stride not in (1, 2):
            raise ValueError(f"stride must be 1 or 2, got {stride}")
        if kernel not in (3, 5):
            raise ValueError(f"kernel must be 3 or 5 for MobileNetV3, got {kernel}")

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
    min_val = divisor if min_value is None else min_value
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Ensure the round down does not go down by more than 10%.
    return new_v + divisor if new_v < 0.9 * v else new_v


class MobileNetV3(nn.Module):
    """
    MobileNetV3 Large / Small following arXiv:1905.02244.


    """

    def __init__(
        self,
        mode: str = "large",
        num_classes: int = 1000,
        width_mult: float = 1.0,
        input_size: int = 224,
    ):
        super().__init__()
        if mode not in ("large", "small"):
            raise ValueError("mode must be 'large' or 'small'")

        self.mode = mode
        self.input_size = input_size
        self.num_classes = num_classes

        if mode == "large":
            cfg = CFG_LARGE
            last_exp = 960  # channels before pooling (see Table 1)
            final_exp = 1280  # final expansion after pooling (Fig. 5)
        else:
            cfg = CFG_SMALL
            last_exp = 576  # channels before pooling (Table 2)
            final_exp = 1024  # final expansion after pooling

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

    def _initialize_weights(self):
        # Default initialization similar to torchvision / common practice.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / n))
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
    """MobileNetV3-Large as in Table 1 of the paper."""
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
    """MobileNetV3-Small as in Table 2 of the paper."""
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
