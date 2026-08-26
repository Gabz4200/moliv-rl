from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class AverageMeter:
    """Track a weighted running average.

    Useful for metrics such as loss, accuracy, throughput, or learning rate.
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
        if not math.isfinite(value):
            raise ValueError(f"value must be finite, got {value}")

        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")

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
    """Calculate top-1 accuracy for single-label classification."""
    if outputs.ndim != 2:
        raise ValueError(
            "outputs must have shape [batch_size, num_classes], "
            f"got {tuple(outputs.shape)}"
        )

    if targets.ndim != 1:
        raise ValueError(
            f"targets must have shape [batch_size], got {tuple(targets.shape)}"
        )

    batch_size, num_classes = outputs.shape

    if targets.size(0) != batch_size:
        raise ValueError(
            f"Batch size mismatch: outputs={batch_size}, targets={targets.size(0)}"
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

    if batch_size == 0:
        return 0.0

    min_target = targets.min().item()
    max_target = targets.max().item()

    if min_target < 0 or max_target >= num_classes:
        raise ValueError(
            "targets contain an invalid class index: "
            f"valid range is [0, {num_classes - 1}], "
            f"got range [{min_target}, {max_target}]"
        )

    predictions = outputs.argmax(dim=1)
    correct = (predictions == targets).sum().item()

    return correct / batch_size
