import torch
from torch import nn


class LiVConv2D(nn.Module):
    """
    Gated depthwise convolution block for 2D feature maps.

    Structure:
        - 1x1 conv expands to 3 * hidden_channels
        - Split into (b_gate, c_gate, x_proj)
        - x = b_gate * x_proj
        - depthwise conv
        - x = c_gate * conv(x)
        - dropout
        - 1x1 conv to out_channels

    Optionally adds:
        - BatchNorm after the final projection
        - Residual connection when in_channels == out_channels
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int | None,
        out_channels: int,
        kernel_size: int = 3,
        hidden_dropout: float = 0.1,
        use_norm: bool = True,
        use_residual: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = (
            hidden_channels if hidden_channels is not None else in_channels * 2
        )
        self.kernel_size = kernel_size
        self.use_residual = use_residual and (in_channels == out_channels)

        # Input projection: (B, C_in, H, W) -> (B, 3*C_mid, H, W)
        self.input_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.hidden_channels * 3,
            kernel_size=1,
            bias=not use_norm,
        )

        # Depthwise spatial convolution
        self.conv = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=self.hidden_channels,  # depthwise
            bias=not use_norm,
        )

        # Output projection
        self.output_conv = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            bias=not use_norm,
        )

        # Optional normalization layers
        self.norm_output = nn.BatchNorm2d(out_channels) if use_norm else nn.Identity()

        self.dropout = nn.Dropout2d(hidden_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input torch.Tensor with shape (B, C_in, H, W).

        Returns:
            Output torch.Tensor with shape (B, C_out, H, W).
        """
        identity = x

        # Input projection
        x = self.input_conv(x)

        # Split into gates and projection
        b_gate, c_gate, x_proj = x.chunk(3, dim=1)  # each: (B, C_mid, H, W)

        # First multiplicative gate
        x = b_gate * x_proj

        # Depthwise local spatial mixing
        x = self.conv(x)

        # Second multiplicative gate
        x = c_gate * x

        # Regularization
        x = self.dropout(x)

        # Output projection
        x = self.output_conv(x)
        x = self.norm_output(x)

        # Optional residual connection
        x = x + identity if self.use_residual else x

        return x


class MLPConv2D(nn.Module):
    """
    Lightweight MLP-like 2D convolution block:

        - 1x1 conv expands channels
        - Hardswish
        - depthwise conv (spatial mixing)
        - Hardswish
        - dropout
        - 1x1 conv to out_channels (linear projection)

    Optionally adds:
        - BatchNorm after each conv
        - Residual connection when in_channels == out_channels
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int | None,
        out_channels: int,
        kernel_size: int = 3,
        hidden_dropout: float = 0.1,
        use_norm: bool = True,
        use_residual: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.hidden_dropout = hidden_dropout
        self.use_residual = use_residual and (in_channels == out_channels)

        self.hidden_channels = (
            hidden_channels if hidden_channels is not None else in_channels * 4
        )

        # Expansion 1x1 conv
        self.input_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.hidden_channels,
            kernel_size=1,
            bias=not use_norm,
        )

        # Depthwise conv
        self.intermediate_conv = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=self.hidden_channels,  # depthwise
            bias=not use_norm,
        )

        # Projection 1x1 conv
        self.output_conv = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            bias=not use_norm,
        )

        # Optional normalization
        self.norm1 = nn.BatchNorm2d(self.hidden_channels) if use_norm else nn.Identity()
        self.norm2 = nn.BatchNorm2d(self.hidden_channels) if use_norm else nn.Identity()
        self.norm3 = nn.BatchNorm2d(self.out_channels) if use_norm else nn.Identity()

        self.act = nn.Hardswish()
        self.dropout = nn.Dropout2d(hidden_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input torch.Tensor with shape (B, C_in, H, W).

        Returns:
            Output torch.Tensor with shape (B, C_out, H, W).
        """
        identity = x

        # Expand + activate
        x = self.input_conv(x)
        x = self.norm1(x)
        x = self.act(x)

        # Depthwise spatial mixing + activate
        x = self.intermediate_conv(x)
        x = self.norm2(x)
        x = self.act(x)

        # Regularization
        x = self.dropout(x)

        # Project to output channels
        x = self.output_conv(x)
        x = self.norm3(x)

        # Optional residual connection
        x = x + identity if self.use_residual else x

        return x


class SwiGluConv2D(nn.Module):
    """
    SwiGLU-style lightweight 2D convolution block:

        - 1x1 conv expands channels to 2 * hidden_channels
        - Split into two independent channel projections (feat and gate)
        - Dual depthwise conv branches (intermediate and gating paths)
        - Hardswish activation on the gating branch
        - Gating: x = act(gate) * feat
        - dropout
        - 1x1 conv to out_channels (linear projection)

    Optionally adds:
        - BatchNorm after convs
        - Residual connection when in_channels == out_channels
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int | None,
        out_channels: int,
        kernel_size: int = 3,
        hidden_dropout: float = 0.1,
        use_norm: bool = True,
        use_residual: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.hidden_dropout = hidden_dropout
        self.use_residual = use_residual and (in_channels == out_channels)

        self.hidden_channels = (
            hidden_channels if hidden_channels is not None else in_channels * 4
        )

        # Expansion 1x1 conv (projects to 2x hidden channels for independent gate/feat)
        self.input_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.hidden_channels * 2,
            kernel_size=1,
            bias=not use_norm,
        )

        # Depthwise conv for feature branch
        self.intermediate_conv = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            groups=self.hidden_channels,  # depthwise
            bias=not use_norm,
        )

        # Depthwise conv for gate branch
        self.gating_conv = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            groups=self.hidden_channels,  # depthwise
            bias=not use_norm,
        )

        # Projection 1x1 conv
        self.output_conv = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            bias=not use_norm,
        )

        # Optional normalization
        self.norm1 = (
            nn.BatchNorm2d(self.hidden_channels * 2) if use_norm else nn.Identity()
        )
        self.norm2 = nn.BatchNorm2d(self.hidden_channels) if use_norm else nn.Identity()
        self.norm3 = nn.BatchNorm2d(self.out_channels) if use_norm else nn.Identity()

        self.act = nn.Hardswish()
        self.dropout = nn.Dropout2d(hidden_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input torch.Tensor with shape (B, C_in, H, W).

        Returns:
            Output torch.Tensor with shape (B, C_out, H, W).
        """
        identity = x

        # Expand to 2x channels
        x = self.input_conv(x)
        x = self.norm1(x)

        # Split into independent feature and gate channel projections
        feat, gate = x.chunk(2, dim=1)

        # Depthwise spatial mixing on independent branches
        feat = self.intermediate_conv(feat)
        gate = self.gating_conv(gate)

        # Gating
        gate = self.act(gate)
        x = gate * feat
        x = self.norm2(x)

        # Regularization
        x = self.dropout(x)

        # Project to output channels
        x = self.output_conv(x)
        x = self.norm3(x)

        # Optional residual connection
        x = x + identity if self.use_residual else x

        return x
