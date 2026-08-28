from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

Reduction = Literal["none", "mean", "sum"]


class FocalLoss(nn.Module):
    r"""FocalLoss(alpha=None, gamma=2.0, reduction='mean', ignore_index=-100, label_smoothing=0.0)

    Multiclass focal loss criterion for logits and integer class targets based on
    `Focal Loss for Dense Object Detection`_.

    The focal loss is defined as:

    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)

    where :math:`p_t` is the model's estimated probability for the ground-truth class,
    :math:`\gamma` is the focusing parameter, and :math:`\alpha_t` is an optional balancing factor.

    Supports standard classification inputs shaped :math:`(N, C)` and dense
    classification inputs shaped :math:`(N, C, d_1, \dots, d_K)`.

    Args:
        alpha (float or Tensor, optional): Weighting factor. Can be a scalar float or a 1D Tensor of shape
            :math:`(C)` assigning per-class weights. Default: ``None``
        gamma (float, optional): Focusing parameter :math:`\gamma \ge 0` modulating easy examples. Default: ``2.0``
        reduction (str, optional): Reduction mode (``'none'``, ``'mean'``, or ``'sum'``). Default: ``'mean'``
        ignore_index (int, optional): Target class value that is ignored and contributes zero gradient. Default: ``-100``
        label_smoothing (float, optional): Label smoothing epsilon in :math:`[0.0, 1.0]`. Default: ``0.0``

    Shape:
        - Input: :math:`(N, C)` where :math:`C` is the number of classes, or :math:`(N, C, d_1, \dots, d_K)`.
        - Target: :math:`(N)` where each value is :math:`0 \le \text{targets}[i] < C`, or :math:`(N, d_1, \dots, d_K)`.
        - Output: Scalar if :attr:`reduction` is ``'mean'`` or ``'sum'``; otherwise same shape as target.

    Examples::

        >>> criterion = FocalLoss(gamma=2.0)
        >>> inputs = torch.randn(4, 5, requires_grad=True)
        >>> targets = torch.tensor([1, 0, 4, 2])
        >>> loss = criterion(inputs, targets)
        >>> loss.backward()

    .. _`Focal Loss for Dense Object Detection`:
        https://arxiv.org/abs/1708.02002
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

        if isinstance(alpha, torch.Tensor):
            alpha_tensor = (
                alpha.float() if not torch.is_floating_point(alpha) else alpha
            )
            self.register_buffer(
                "alpha",
                alpha_tensor.detach().clone(),
            )
            self.alpha_scalar: float | None = None
        else:
            self.alpha = None
            self.alpha_scalar = float(alpha) if alpha is not None else None

        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss from logits and integer class targets.

        Args:
            inputs: Unnormalized model logits shaped [N, C] or [N, C, d1, ..., dK].
            targets: Ground-truth class indices shaped [N] or [N, d1, ..., dK].

        Returns:
            Computed focal loss (scalar or Tensor matching reduction mode).
        """
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

        return focal_loss.sum() / valid_count if valid_count > 0 else focal_loss.sum()


__all__ = [
    "FocalLoss",
    "Reduction",
]
