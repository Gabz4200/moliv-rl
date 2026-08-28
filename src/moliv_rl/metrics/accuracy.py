from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AverageMeter:
    r"""AverageMeter()

    Computes and stores the running average and current value of a metric.

    Useful for tracking metrics such as loss, accuracy, throughput, and learning rate.

    Examples::

        >>> meter = AverageMeter()
        >>> meter.update(10.0, n=2)
        >>> meter.update(20.0, n=3)
        >>> meter.avg
        16.0
    """

    val: float = 0.0
    avg: float = 0.0
    sum: float = 0.0
    count: int = 0

    def reset(self) -> None:
        """Reset all accumulated values."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        """Add ``n`` observations with the specified value."""
        if n == 0:
            return

        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_accuracy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    r"""calculate_accuracy(outputs, targets) -> float

    Calculate top-1 classification accuracy from model logits and integer targets.

    .. math::
        \text{Accuracy} = \frac{1}{N} \sum_{i=1}^N \mathbf{1}(\text{argmax}(\text{outputs}_i) = \text{targets}_i)

    Args:
        outputs (Tensor): Model predicted logits or scores of shape :math:`(N, C)`.
        targets (Tensor): Ground-truth class labels of shape :math:`(N)`.

    Returns:
        float: Top-1 accuracy in :math:`[0.0, 1.0]`. If batch size :math:`N = 0`, returns ``0.0``.

    Examples::

        >>> outputs = torch.tensor([[2.0, 0.5], [0.1, 3.0]])
        >>> targets = torch.tensor([0, 1])
        >>> calculate_accuracy(outputs, targets)
        1.0
    """
    batch_size = outputs.size(0)
    if batch_size == 0:
        return 0.0

    predictions = outputs.argmax(dim=1)
    correct = (predictions == targets).sum().item()

    return correct / batch_size


__all__ = [
    "AverageMeter",
    "calculate_accuracy",
]
