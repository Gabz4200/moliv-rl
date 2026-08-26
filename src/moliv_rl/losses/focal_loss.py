from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

Reduction = Literal["none", "mean", "sum"]


class FocalLoss(nn.Module):
    """Multiclass focal loss for logits and integer class targets.

    Supports standard classification inputs shaped ``[N, C]`` and dense
    classification inputs shaped ``[N, C, d1, ..., dK]``.
    """

    def __init__(
        self,
        alpha: float | torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: Reduction = "mean",
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()

        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")

        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(
                f"reduction must be one of 'none', 'mean', or 'sum'; got {reduction!r}"
            )

        if not 0.0 <= label_smoothing <= 1.0:
            raise ValueError(
                f"label_smoothing must be in [0.0, 1.0], got {label_smoothing}"
            )

        if isinstance(alpha, (float, int)) and alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")

        if alpha is not None and not isinstance(
            alpha,
            (float, int, torch.Tensor),
        ):
            raise TypeError(
                "alpha must be None, a non-negative scalar, "
                "or a tensor of class weights"
            )

        if isinstance(alpha, torch.Tensor):
            if alpha.ndim != 1:
                raise ValueError(
                    "tensor alpha must have shape [num_classes], "
                    f"got {tuple(alpha.shape)}"
                )

            if not torch.is_floating_point(alpha):
                alpha = alpha.float()

            if torch.any(alpha < 0):
                raise ValueError("all tensor alpha values must be non-negative")

            self.register_buffer(
                "alpha",
                alpha.detach().clone(),
            )
            self.alpha_scalar: float | None = None
        else:
            self.alpha = None
            self.alpha_scalar = float(alpha) if alpha is not None else None

        self.gamma = float(gamma)
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss from logits and integer class targets."""
        self._validate_inputs(inputs, targets)

        if inputs.ndim == 2:
            expected_target_shape = (inputs.size(0),)
        else:
            expected_target_shape = (
                inputs.size(0),
                *inputs.shape[2:],
            )

        if tuple(targets.shape) != expected_target_shape:
            raise ValueError(
                "targets must have shape "
                f"{expected_target_shape} for inputs shaped "
                f"{tuple(inputs.shape)}, got {tuple(targets.shape)}"
            )

        if self.alpha is not None and self.alpha.numel() != inputs.size(1):
            raise ValueError(
                "alpha must contain one value per class: "
                f"expected {inputs.size(1)}, "
                f"got {self.alpha.numel()}"
            )

        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.alpha,
            reduction="none",
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )

        log_pt = -ce_loss
        pt = log_pt.exp()

        focal_loss = (1.0 - pt).pow(self.gamma) * ce_loss

        ignored = targets == self.ignore_index
        focal_loss = focal_loss.masked_fill(ignored, 0.0)

        if self.alpha_scalar is not None:
            focal_loss = focal_loss * self.alpha_scalar

        if self.reduction == "none":
            return focal_loss

        if self.reduction == "sum":
            return focal_loss.sum()

        valid_count = (~ignored).sum()

        if valid_count == 0:
            return focal_loss.sum()

        return focal_loss.sum() / valid_count

    @staticmethod
    def _validate_inputs(
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        if not isinstance(inputs, torch.Tensor):
            raise TypeError(
                f"inputs must be a torch.Tensor, got {type(inputs).__name__}"
            )

        if not isinstance(targets, torch.Tensor):
            raise TypeError(
                f"targets must be a torch.Tensor, got {type(targets).__name__}"
            )

        if inputs.ndim < 2:
            raise ValueError(
                f"inputs must have shape [N, C, ...], got {tuple(inputs.shape)}"
            )

        if targets.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise TypeError(
                f"targets must contain integer class indices, got dtype={targets.dtype}"
            )

        if inputs.size(0) != targets.size(0):
            raise ValueError(
                "Batch size mismatch: "
                f"inputs={inputs.size(0)}, "
                f"targets={targets.size(0)}"
            )
