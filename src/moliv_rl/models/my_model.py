from __future__ import annotations

import logging
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


class LiVConv2D(nn.Module):
    r"""LiVConv2D(in_channels, hidden_channels, out_channels, kernel_size=3, hidden_dropout=0.1, use_norm=True, use_residual=True)

    Gated depthwise convolution block for 2D feature maps.

    Applies a 1x1 input convolution expanding to :math:`3 \times \text{hidden\_channels}`,
    chunks into bilinear gates and projections, performs depthwise spatial convolution,
    and applies an output 1x1 convolution with optional batch normalization and dropout.

    Args:
        in_channels (int): Number of input channels.
        hidden_channels (int, optional): Intermediate expanded channels.
            Default: if ``None``, ``in_channels * 2``.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Depthwise convolution kernel size. Default: ``3``
        hidden_dropout (float, optional): Dropout probability for 2D spatial features. Default: ``0.1``
        use_norm (bool, optional): Whether to apply batch normalization. Default: ``True``
        use_residual (bool, optional): Whether to add residual connection when channel dimensions match. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, H, W)`
        - Output: :math:`(N, C_{out}, H, W)`

    Examples::

        >>> block = LiVConv2D(in_channels=32, hidden_channels=64, out_channels=32)
        >>> x = torch.randn(2, 32, 16, 16)
        >>> out = block(x)
        >>> out.shape
        torch.Size([2, 32, 16, 16])
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
    ) -> None:
        super().__init__()
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
            bias=True,
        )
        self.conv = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=self.hidden_channels,
            bias=True,
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
    r"""MLPConv2D(in_channels, hidden_channels, out_channels, kernel_size=3, hidden_dropout=0.1, use_norm=True, use_residual=True)

    Lightweight MLP-like 2D convolution block.

    Features depthwise convolution flanked by pointwise projections,
    batch normalization, HardSwish non-linearities, and spatial dropout.

    Args:
        in_channels (int): Number of input channels.
        hidden_channels (int, optional): Intermediate channel dimension.
            Default: if ``None``, ``in_channels * 4``.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Spatial convolution kernel size. Default: ``3``
        hidden_dropout (float, optional): Dropout probability. Default: ``0.1``
        use_norm (bool, optional): Whether to apply batch normalization layers. Default: ``True``
        use_residual (bool, optional): Whether to use residual skip connection. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, H, W)`
        - Output: :math:`(N, C_{out}, H, W)`

    Examples::

        >>> block = MLPConv2D(in_channels=32, hidden_channels=128, out_channels=32)
        >>> x = torch.randn(2, 32, 16, 16)
        >>> out = block(x)
        >>> out.shape
        torch.Size([2, 32, 16, 16])
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
    ) -> None:
        super().__init__()
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


class GatedConv2D(nn.Module):
    r"""GatedConv2D(in_channels, hidden_channels, out_channels, kernel_size=3, hidden_dropout=0.1, use_norm=True, use_residual=True)

    SwiGLU-style lightweight 2D convolution block with dual-branch depthwise gating.

    Args:
        in_channels (int): Number of input channels.
        hidden_channels (int, optional): Intermediate channel dimension per branch.
            Default: if ``None``, ``in_channels * 4``.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Spatial convolution kernel size. Default: ``3``
        hidden_dropout (float, optional): Dropout probability. Default: ``0.1``
        use_norm (bool, optional): Whether to apply batch normalization layers. Default: ``True``
        use_residual (bool, optional): Whether to use residual connection when shapes match. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, H, W)`
        - Output: :math:`(N, C_{out}, H, W)`

    Examples::

        >>> block = GatedConv2D(in_channels=32, hidden_channels=128, out_channels=32)
        >>> x = torch.randn(2, 32, 16, 16)
        >>> out = block(x)
        >>> out.shape
        torch.Size([2, 32, 16, 16])
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
    ) -> None:
        super().__init__()
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
    r"""MyBlock(in_channels, hidden_channels, out_channels, kernel_size=3, hidden_dropout=0.1, include_swiglu=True, use_norm=True, use_residual=True)

    Sequential composite block combining LiVConv2D, MLPConv2D, and optional GatedConv2D.

    Args:
        in_channels (int): Number of input channels.
        hidden_channels (int, optional): Intermediate channel count for sub-blocks. Default: ``None``
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Spatial kernel size. Default: ``3``
        hidden_dropout (float, optional): Dropout probability. Default: ``0.1``
        include_swiglu (bool, optional): If ``True``, appends GatedConv2D to the block. Default: ``True``
        use_norm (bool, optional): Whether to apply batch normalization. Default: ``True``
        use_residual (bool, optional): Whether to enable residual skip connections. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, H, W)`
        - Output: :math:`(N, C_{out}, H, W)`
    """

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
            GatedConv2D(
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


class MyModel(nn.Module):
    r"""MyModel(block_dims, in_channels=3, out_channels=512, patch_size=8, dropout=0.2)

    Vision foundation model with non-overlapping patch stem and gated convolution stages.

    Args:
        block_dims (list of int): Sequence of channel dimensions across successive stages.
        in_channels (int, optional): Number of input image channels. Default: ``3``
        out_channels (int, optional): Output embedding dimension of the backbone. Default: ``512``
        patch_size (int, optional): Spatial downsampling stride and patch kernel size. Default: ``8``
        dropout (float, optional): Dropout rate applied across stages. Default: ``0.2``

    Shape:
        - Input: :math:`(N, C_{in}, H, W)`
        - Output: :math:`(N, C_{out}, H / \text{patch\_size}, W / \text{patch\_size})`

    Examples::

        >>> model = MyModel(block_dims=[32, 64, 128], patch_size=8, out_channels=256)
        >>> x = torch.randn(2, 3, 64, 64)
        >>> feats = model(x)
        >>> feats.shape
        torch.Size([2, 256, 8, 8])
    """

    def __init__(
        self,
        block_dims: list[int],
        in_channels: int = 3,
        out_channels: int = 512,
        patch_size: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
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
                bias=False,
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

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        r"""Return a flat embedding vector for the input tensor.

        Applies the backbone feature extractor and global average pooling
        to produce a :math:`(N, D)` embedding suitable for contrastive and
        regularization losses such as SIGReg.

        Args:
            x (torch.Tensor): Input image batch of shape :math:`(N, C, H, W)`.

        Returns:
            torch.Tensor: Flat embedding tensor of shape :math:`(N, D)`.
        """
        x = self.stem(x)
        x = self.model(x)
        x = self.output_conv(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        return x.flatten(1)


class MyVideoModel(nn.Module):
    r"""MyVideoModel(block_dims, in_channels=3, out_channels=512, patch_size=8, dropout=0.2, conv_kernel_size=3)

    Causal video sequence model with temporal feature delta representations and state caching.

    Extracts spatial representations per frame via :class:`MyModel`, computes consecutive
    frame feature differences :math:`\Delta_t = \text{feats}_t - \text{feats}_{t-1}`, normalizes
    the concatenated representation with :class:`~torch.nn.GroupNorm`, and applies causal 1D temporal
    convolution. Supports both batched sequence forward passes and cached single-step streaming inference.

    Args:
        block_dims (list of int): Channel dimensions across backbone stages.
        in_channels (int, optional): Number of input video channels. Default: ``3``
        out_channels (int, optional): Output channel dimension. Default: ``512``
        patch_size (int, optional): Spatial stem downsampling patch size. Default: ``8``
        dropout (float, optional): Dropout probability. Default: ``0.2``
        conv_kernel_size (int, optional): Temporal convolution kernel size. Default: ``3``

    Shape:
        - Batched Forward: :math:`(B, T, C_{in}, H, W) \to (B, T, C_{out}, H', W')`
        - Streaming Forward Step: :math:`(B, 1, C_{in}, H, W) \to (B, 1, C_{out}, H', W')`
    """

    def __init__(
        self,
        block_dims: list[int],
        in_channels: int = 3,
        out_channels: int = 512,
        patch_size: int = 8,
        dropout: float = 0.2,
        conv_kernel_size: int = 3,
    ) -> None:
        super().__init__()

        self.out_channels = out_channels
        self.conv_kernel_size = conv_kernel_size

        self.feature_extractor = MyModel(
            block_dims=block_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=patch_size,
            dropout=dropout,
        )

        # Input channels: [feature, delta] -> 2 * out_channels
        self.temporal_conv = nn.Conv1d(
            in_channels=out_channels * 2,
            out_channels=out_channels,
            kernel_size=conv_kernel_size,
            padding=0,  # no padding; we handle causality via explicit window
            bias=True,
        )

        num_groups = next(
            (
                g
                for g in range(min(32, out_channels * 2), 0, -1)
                if (out_channels * 2) % g == 0
            ),
            1,
        )
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            (B, T, out_channels, H', W')
        """
        B, T, C, H, W = x.shape

        # Frame-wise feature extraction
        x = x.view(B * T, C, H, W)
        feats = self.feature_extractor(x)
        _, C_feat, Hf, Wf = feats.shape
        feats = feats.view(B, T, C_feat, Hf, Wf)

        # delta_t = feats_t - feats_{t-1}, with feats_{-1} = 0
        feats_prev = torch.cat(
            [
                torch.zeros_like(feats[:, :1]),
                feats[:, :-1],
            ],
            dim=1,
        )
        delta = feats - feats_prev

        return self._apply_temporal_conv(feats, delta)

    def _apply_temporal_conv(
        self,
        feats: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        B, T, C_feat, Hf, Wf = feats.shape

        # Concat [feature, delta]
        x = torch.cat([feats, delta], dim=2)  # (B, T, 2*C_feat, Hf, Wf)
        x = self.norm(x.view(B * T, 2 * C_feat, Hf, Wf)).view(B, T, 2 * C_feat, Hf, Wf)

        B, T, C2, Hf, Wf = x.shape

        # Rearrange to (B, Hf, Wf, C2, T) to match forward_step spatial layout
        x = x.permute(0, 3, 4, 2, 1).contiguous()
        x = x.view(B * Hf * Wf, C2, T)

        # Causal conv over T: we need each output at t to depend on t, t-1, ..., t-k+1
        # We'll implement this via explicit padding on the left.
        k = self.conv_kernel_size
        # Left-pad with k-1 zeros along T
        x = F.pad(x, (k - 1, 0))  # (..., T + k - 1)

        # Now standard conv with no padding
        x = self.temporal_conv(x)  # output length = (T + k - 1) - k + 1 = T
        # x: (B*Hf*Wf, out_channels, T)

        # Reshape back to (B, T, out_channels, Hf, Wf)
        x = x.view(B, Hf, Wf, self.out_channels, T)
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        return x

    def forward_step(
        self,
        x_t: torch.Tensor,
        cache: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Single-frame (or chunk) forward with caching for temporal conv.

        Args:
            x_t: (B, 1, C, H, W) or (B, C, H, W)
            cache: dict with:
                - "prev_feats": (B, C_feat, Hf, Wf) or None for t=0
                - "conv_buffer": (B*Hf*Wf, 2*C_feat, k-1) or None
        Returns:
            y_t: (B, 1, out_channels, Hf, Wf)
            new_cache: updated cache dict
        """
        if x_t.dim() == 4:
            x_t = x_t.unsqueeze(1)  # (B, 1, C, H, W)

        B, T, C, H, W = x_t.shape
        assert T == 1

        # Feature extraction for this frame
        x_flat = x_t.view(B, C, H, W)
        feats_t = self.feature_extractor(x_flat)  # (B, C_feat, Hf, Wf)
        _, C_feat, Hf, Wf = feats_t.shape

        # Compute delta_t = feats_t - feats_{t-1}
        feats_prev = (
            cache["prev_feats"]
            if cache is not None and "prev_feats" in cache
            else torch.zeros_like(feats_t)
        )
        delta_t = feats_t - feats_prev

        # Concat [feature, delta]
        x_t = torch.cat([feats_t, delta_t], dim=1)  # (B, 2*C_feat, Hf, Wf)
        x_t = self.norm(x_t)

        # Prepare for temporal conv over T
        # We maintain a buffer of last (k-1) steps per spatial location
        k = self.conv_kernel_size

        # Reshape to (B*Hf*Wf, 2*C_feat, 1)
        x_t = x_t.view(B, 2 * C_feat, Hf, Wf)
        x_t = x_t.permute(0, 2, 3, 1).contiguous()  # (B, Hf, Wf, 2*C_feat)
        x_t = x_t.view(B * Hf * Wf, 2 * C_feat, 1)

        expected_buffer_shape = (B * Hf * Wf, 2 * C_feat, k - 1)
        cached_buffer = (
            cache["conv_buffer"]
            if cache is not None and "conv_buffer" in cache
            else None
        )

        if cached_buffer is not None and cached_buffer.shape != expected_buffer_shape:
            raise RuntimeError(
                "Streaming cache shape mismatch: expected "
                f"{expected_buffer_shape}, got {tuple(cached_buffer.shape)}. "
                "All frames in a streaming sequence must have identical "
                "batch size, channels, and spatial dimensions."
            )

        buffer = (
            cached_buffer
            if cached_buffer is not None
            else torch.zeros(
                expected_buffer_shape,
                device=x_t.device,
                dtype=x_t.dtype,
            )
        )

        # Concat buffer + current step -> (..., k)
        x_with_hist = torch.cat([buffer, x_t], dim=-1)  # (..., k)

        # Apply conv (no padding)
        y_t = self.temporal_conv(x_with_hist)  # (..., 1)

        # Update buffer: drop oldest (leftmost) time step, append current
        new_buffer = x_with_hist[..., 1:]  # keep last (k-1) steps

        # Reshape output back to (B, 1, out_channels, Hf, Wf)
        y_t = y_t.view(B, Hf, Wf, self.out_channels, 1)
        y_t = y_t.permute(0, 4, 3, 1, 2).contiguous()

        new_cache: dict[str, Any] = {
            "prev_feats": feats_t,  # (B, C_feat, Hf, Wf)
            "conv_buffer": new_buffer,  # (B*Hf*Wf, 2*C_feat, k-1)
        }

        return y_t, new_cache


class PretrainedMobileNetV3(nn.Module):
    r"""PretrainedMobileNetV3(num_classes=10, mode='large', freeze_backbone=False)

    Transfer-learning wrapper around torchvision's pretrained MobileNetV3.

    Loads ImageNet-pretrained weights and replaces the final classifier head
    with a new linear layer for the target number of classes.

    Args:
        num_classes (int, optional): Number of target classification classes. Default: ``10``
        mode (str, optional): Architecture variant ('large' or 'small'). Default: ``'large'``
        freeze_backbone (bool, optional): If ``True``, freezes all backbone weights. Default: ``False``

    Shape:
        - Input: :math:`(N, 3, H, W)`
        - Output: :math:`(N, \text{num\_classes})`

    Examples::

        >>> model = PretrainedMobileNetV3(num_classes=10)
        >>> x = torch.randn(2, 3, 224, 224)
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([2, 10])
    """

    def __init__(
        self,
        num_classes: int = 10,
        mode: str = "large",
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

        weights = MobileNet_V3_Large_Weights.DEFAULT if mode == "large" else None
        if mode == "large":
            backbone = mobilenet_v3_large(weights=weights)
        else:
            raise ValueError(
                f"Unsupported mode '{mode}'. Only 'large' is currently supported for pretrained weights."
            )

        self.backbone = backbone.features
        self.pool = backbone.avgpool

        backbone_out = cast(int, backbone.classifier[0].in_features)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_out, num_classes),
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.pool.parameters():
                param.requires_grad = False

        self.to(memory_format=torch.channels_last)  # type: ignore[call-arg]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        r"""Return pre-pooling feature maps."""
        return self.backbone(x)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        r"""Return flat embedding vector for the input tensor."""
        x = self.backbone(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


class ClassificationModel(nn.Module):
    r"""ClassificationModel(block_dims, in_channels=3, out_channels=512, patch_size=8, dropout=0.2, num_classes=1000)

    Image classification model combining MyModel backbone with global average pooling and linear classifier.

    Args:
        block_dims (list of int): Channel dimensions across backbone stages.
        in_channels (int, optional): Number of input image channels. Default: ``3``
        out_channels (int, optional): Intermediate backbone feature dimension. Default: ``512``
        patch_size (int, optional): Patch stem kernel and stride size. Default: ``8``
        dropout (float, optional): Dropout probability. Default: ``0.2``
        num_classes (int, optional): Number of target classification classes. Default: ``1000``

    Shape:
        - Input: :math:`(N, C_{in}, H, W)`
        - Output: :math:`(N, \text{num\_classes})`

    Examples::

        >>> model = ClassificationModel(block_dims=[32, 64], num_classes=10)
        >>> x = torch.randn(2, 3, 64, 64)
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([2, 10])
    """

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

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        r"""Return pre-classifier spatial feature maps.

        Args:
            x (torch.Tensor): Input image batch of shape :math:`(N, C, H, W)`.

        Returns:
            torch.Tensor: Spatial feature maps of shape :math:`(N, C_{out}, H', W')`.
        """
        return self.feature_extractor(x)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        r"""Return a flat embedding vector for regularization losses.

        Applies global average pooling over the spatial feature maps to
        produce a :math:`(N, D)` tensor suitable for SIGReg and similar
        isotropic-Gaussian regularizers.

        Args:
            x (torch.Tensor): Input image batch of shape :math:`(N, C, H, W)`.

        Returns:
            torch.Tensor: Flat embedding tensor of shape :math:`(N, D)`.
        """
        features = self.get_features(x)
        pooled = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        return pooled.flatten(1)


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "my_model": MyModel,
    "classification_model": ClassificationModel,
    "my_video_model": MyVideoModel,
    "mobilenetv3_pretrained": PretrainedMobileNetV3,
}


def get_model(
    model_name: str,
    optimize: bool = True,
    fullgraph: bool = True,
    **kwargs: Any,
) -> nn.Module:
    r"""get_model(model_name, optimize=True, fullgraph=True, **kwargs) -> nn.Module

    Instantiate and optionally compile a registered neural network model by name.

    Args:
        model_name (str): Model architecture name registered in :attr:`MODEL_REGISTRY`.
        optimize (bool, optional): If ``True``, compiles the model using :func:`torch.compile`. Default: ``True``
        fullgraph (bool, optional): Whether to enforce fullgraph compilation with :func:`torch.compile`. Default: ``True``
        **kwargs: Additional keyword arguments forwarded to the model constructor.

    Returns:
        nn.Module: Instantiated and optionally compiled PyTorch model.
    """
    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(**kwargs)

    if optimize:
        device = "CUDA" if torch.cuda.is_available() else "CPU"
        logger.info("Optimizing for %s: Compiling macro model...", device)

        compile_mode = "reduce-overhead" if torch.cuda.is_available() else "default"
        model = cast(
            nn.Module,
            torch.compile(model, fullgraph=fullgraph, mode=compile_mode),
        )

    return model


__all__ = [
    "MODEL_REGISTRY",
    "ClassificationModel",
    "GatedConv2D",
    "LiVConv2D",
    "MLPConv2D",
    "MyBlock",
    "MyModel",
    "MyVideoModel",
    "PretrainedMobileNetV3",
    "get_model",
]
