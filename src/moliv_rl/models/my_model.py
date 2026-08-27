from __future__ import annotations

import logging
from typing import Any, cast

import torch
from torch import nn

logger = logging.getLogger(__name__)


def _validate_conv_params(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    hidden_dropout: float,
) -> None:
    """Validate common 2D convolution block hyperparameters."""
    if in_channels <= 0:
        raise ValueError(f"in_channels must be positive, got {in_channels}")
    if out_channels <= 0:
        raise ValueError(f"out_channels must be positive, got {out_channels}")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size must be an odd positive integer, got {kernel_size}"
        )
    if not (0.0 <= hidden_dropout < 1.0):
        raise ValueError(
            f"hidden_dropout must be in [0.0, 1.0), got {hidden_dropout}"
        )


class LiVConv2D(nn.Module):
    """Gated depthwise convolution block for 2D feature maps."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int | None,
        out_channels: int,
        kernel_size: int = 3,
        hidden_dropout: float = 0.1,
        use_norm: bool = True,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        _validate_conv_params(in_channels, out_channels, kernel_size, hidden_dropout)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = (
            hidden_channels if hidden_channels is not None else in_channels * 2
        )
        self.kernel_size = kernel_size
        self.use_residual = use_residual and (in_channels == out_channels)

        self.input_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.hidden_channels * 3,
            kernel_size=1,
            bias=not use_norm,
        )
        self.conv = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=self.hidden_channels,
            bias=not use_norm,
        )
        self.output_conv = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            bias=not use_norm,
        )
        self.norm_output = nn.BatchNorm2d(out_channels) if use_norm else nn.Identity()
        self.dropout = nn.Dropout2d(hidden_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.input_conv(x)
        b_gate, c_gate, x_proj = x.chunk(3, dim=1)
        x = b_gate * x_proj
        x = self.conv(x)
        x = c_gate * x
        x = self.output_conv(x)
        x = self.norm_output(x)
        x = self.dropout(x)
        return x + identity if self.use_residual else x


class MLPConv2D(nn.Module):
    """Lightweight MLP-like 2D convolution block."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int | None,
        out_channels: int,
        kernel_size: int = 3,
        hidden_dropout: float = 0.1,
        use_norm: bool = True,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        _validate_conv_params(in_channels, out_channels, kernel_size, hidden_dropout)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.hidden_dropout = hidden_dropout
        self.use_residual = use_residual and (in_channels == out_channels)
        self.hidden_channels = (
            hidden_channels if hidden_channels is not None else in_channels * 4
        )

        self.input_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.hidden_channels,
            kernel_size=1,
            bias=not use_norm,
        )
        self.intermediate_conv = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=self.hidden_channels,
            bias=not use_norm,
        )
        self.output_conv = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            bias=not use_norm,
        )
        self.norm1 = nn.BatchNorm2d(self.hidden_channels) if use_norm else nn.Identity()
        self.norm2 = nn.BatchNorm2d(self.hidden_channels) if use_norm else nn.Identity()
        self.norm3 = nn.BatchNorm2d(self.out_channels) if use_norm else nn.Identity()
        self.act = nn.Hardswish()
        self.dropout = nn.Dropout2d(hidden_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.input_conv(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.intermediate_conv(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.output_conv(x)
        x = self.norm3(x)
        x = self.dropout(x)
        return x + identity if self.use_residual else x


class SwiGluConv2D(nn.Module):
    """SwiGLU-style lightweight 2D convolution block."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int | None,
        out_channels: int,
        kernel_size: int = 3,
        hidden_dropout: float = 0.1,
        use_norm: bool = True,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        _validate_conv_params(in_channels, out_channels, kernel_size, hidden_dropout)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.hidden_dropout = hidden_dropout
        self.use_residual = use_residual and (in_channels == out_channels)
        self.hidden_channels = (
            hidden_channels if hidden_channels is not None else in_channels * 4
        )

        self.input_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.hidden_channels * 2,
            kernel_size=1,
            bias=not use_norm,
        )
        self.dw_conv = nn.Conv2d(
            in_channels=self.hidden_channels * 2,
            out_channels=self.hidden_channels * 2,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            groups=self.hidden_channels * 2,
            bias=not use_norm,
        )
        self.output_conv = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            bias=not use_norm,
        )
        self.norm1 = (
            nn.BatchNorm2d(self.hidden_channels * 2) if use_norm else nn.Identity()
        )
        self.norm_dw = (
            nn.BatchNorm2d(self.hidden_channels * 2) if use_norm else nn.Identity()
        )
        self.norm3 = nn.BatchNorm2d(self.out_channels) if use_norm else nn.Identity()
        self.act = nn.Hardswish()
        self.dropout = nn.Dropout2d(hidden_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.input_conv(x)
        x = self.norm1(x)
        x = self.dw_conv(x)
        x = self.norm_dw(x)
        feat, gate = x.chunk(2, dim=1)
        gate = self.act(gate)
        x = gate * feat
        x = self.output_conv(x)
        x = self.norm3(x)
        x = self.dropout(x)
        return x + identity if self.use_residual else x


class MyBlock(nn.Module):
    """A custom block that combines LiVConv2D, MLPConv2D, and SwiGluConv2D."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int | None,
        out_channels: int,
        kernel_size: int = 3,
        hidden_dropout: float = 0.1,
        include_swiglu: bool = True,
        use_norm: bool = True,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.liv_conv = LiVConv2D(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            hidden_dropout=hidden_dropout,
            use_norm=use_norm,
            use_residual=use_residual,
        )
        self.mlp_conv = MLPConv2D(
            in_channels=out_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            hidden_dropout=hidden_dropout,
            use_norm=use_norm,
            use_residual=use_residual,
        )
        self.swiglu_conv = (
            SwiGluConv2D(
                in_channels=out_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                hidden_dropout=hidden_dropout,
                use_norm=use_norm,
                use_residual=use_residual,
            )
            if include_swiglu
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.liv_conv(x)
        x = self.mlp_conv(x)
        x = self.swiglu_conv(x)
        return x


# still not implemented
class MyModel(nn.Module):
    """Vision foundation model with optimized gated convolutions."""

    def __init__(
        self,
        block_dims: list[int],
        in_channels: int = 3,
        out_channels: int = 512,
        patch_size: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if not block_dims or len(block_dims) < 2:
            raise ValueError(
                f"block_dims must contain at least 2 dimensions, got {block_dims}"
            )
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {dropout}")

        self.block_dims = block_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.block_dims[0],
                kernel_size=3,
                padding=1,
            ),
            nn.Conv2d(
                in_channels=self.block_dims[0],
                out_channels=self.block_dims[0],
                kernel_size=patch_size,
                stride=patch_size,
            ),
            nn.BatchNorm2d(self.block_dims[0]),
            nn.Hardswish(),
        )

        self.output_conv = nn.Conv2d(
            in_channels=self.block_dims[-1],
            out_channels=self.out_channels,
            kernel_size=1,
        )

        self.model = nn.Sequential(
            *[
                MyBlock(
                    in_channels=self.block_dims[i],
                    hidden_channels=None,
                    out_channels=self.block_dims[i + 1],
                    kernel_size=3,
                    hidden_dropout=self.dropout,
                    include_swiglu=i % 2 == 0,
                    use_norm=True,
                    use_residual=True,
                )
                for i in range(len(self.block_dims) - 1)
            ]
        )

        self.to(memory_format=torch.channels_last)  # type: ignore[call-arg,no-matching-overload]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.model(x)
        x = self.output_conv(x)
        return x


class ClassificationModel(nn.Module):
    """A classification model that uses MyModel as a feature extractor."""

    def __init__(
        self,
        block_dims: list[int],
        in_channels: int = 3,
        out_channels: int = 512,
        patch_size: int = 8,
        dropout: float = 0.2,
        num_classes: int = 1000,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")

        self.feature_extractor = MyModel(
            block_dims=block_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=patch_size,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(out_channels, num_classes),
        )

        self.to(memory_format=torch.channels_last)  # type: ignore[call-arg,no-matching-overload]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "my_model": MyModel,
    "classification_model": ClassificationModel,
}


def get_model(
    model_name: str,
    optimize: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """Instantiate and optionally optimize a model by name."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {model_name}")

    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(**kwargs)

    if optimize:
        device = "CUDA" if torch.cuda.is_available() else "CPU"
        logger.info("Optimizing for %s: Compiling macro model...", device)

        compile_mode = "reduce-overhead" if torch.cuda.is_available() else "default"
        model = cast(
            nn.Module,
            torch.compile(model, fullgraph=True, mode=compile_mode),
        )

    return model


__all__ = [
    "ClassificationModel",
    "LiVConv2D",
    "MLPConv2D",
    "MyBlock",
    "MyModel",
    "SwiGluConv2D",
    "get_model",
]
