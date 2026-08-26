import torch
from torch import nn

# Placeholder code from my local template for now. Soon to change.


class ConvBlock(nn.Module):
    """Reusable 2D convolutional block with BatchNorm and ReLU activation."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"Expected 4D input tensor (B, C, H, W), got shape {x.shape}"
            )
        return self.block(x)


class MyModel(nn.Module):
    """Modular CNN backbone with classification/representation head."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        self.backbone = nn.Sequential(
            ConvBlock(in_channels, 64),
            ConvBlock(64, 128),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"Expected 4D input tensor (B, C, H, W), got shape {x.shape}"
            )
        features = self.backbone(x)
        pooled = features.mean(dim=[2, 3])
        return self.head(pooled)
