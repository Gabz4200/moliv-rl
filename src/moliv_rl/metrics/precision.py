from __future__ import annotations

from typing import Literal

import torch

PrecisionAverage = Literal["micro", "macro", "weighted"] | None


def calculate_precision(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    average: PrecisionAverage = "macro",
    num_classes: int | None = None,
    zero_division: float = 0.0,
) -> float | torch.Tensor:
    """Calculate multiclass precision from model outputs and targets."""
    _validate_inputs(outputs, targets)

    if average not in {"micro", "macro", "weighted", None}:
        raise ValueError(
            "average must be one of 'micro', 'macro', 'weighted', or None; "
            f"got {average!r}"
        )

    if zero_division not in {0.0, 1.0}:
        raise ValueError(f"zero_division must be 0.0 or 1.0, got {zero_division}")

    inferred_classes = outputs.size(1)

    if num_classes is None:
        num_classes = inferred_classes
    elif num_classes != inferred_classes:
        raise ValueError(
            "num_classes must match outputs.size(1): "
            f"num_classes={num_classes}, outputs.size(1)={inferred_classes}"
        )

    if targets.numel() == 0:
        if average is None:
            return torch.full(
                (num_classes,),
                fill_value=zero_division,
                dtype=torch.float32,
                device=outputs.device,
            )
        return float(zero_division)

    if targets.min().item() < 0 or targets.max().item() >= num_classes:
        raise ValueError(
            "targets contain an invalid class index: "
            f"valid range is [0, {num_classes - 1}], "
            f"got range [{targets.min().item()}, {targets.max().item()}]"
        )

    predictions = outputs.argmax(dim=1)

    true_positives = torch.zeros(
        num_classes,
        dtype=torch.float64,
        device=outputs.device,
    )
    predicted_positives = torch.zeros_like(true_positives)
    support = torch.zeros_like(true_positives)

    true_positives.scatter_add_(
        dim=0,
        index=targets,
        src=(predictions == targets).to(torch.float64),
    )
    predicted_positives.scatter_add_(
        dim=0,
        index=predictions,
        src=torch.ones_like(predictions, dtype=torch.float64),
    )
    support.scatter_add_(
        dim=0,
        index=targets,
        src=torch.ones_like(targets, dtype=torch.float64),
    )

    per_class_precision = torch.full_like(
        true_positives,
        fill_value=zero_division,
    )

    valid_classes = predicted_positives > 0
    per_class_precision[valid_classes] = (
        true_positives[valid_classes] / predicted_positives[valid_classes]
    )

    if average is None:
        return per_class_precision.float()

    if average == "micro":
        total_true_positives = true_positives.sum()
        total_predicted_positives = predicted_positives.sum()

        if total_predicted_positives == 0:
            return float(zero_division)

        return float((total_true_positives / total_predicted_positives).item())

    if average == "macro":
        return float(per_class_precision.mean().item())

    total_support = support.sum()
    if total_support == 0:
        return float(zero_division)

    weighted_precision = (per_class_precision * support).sum() / total_support

    return float(weighted_precision.item())


def _validate_inputs(
    outputs: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    if not isinstance(outputs, torch.Tensor):
        raise TypeError(f"outputs must be a torch.Tensor, got {type(outputs).__name__}")

    if not isinstance(targets, torch.Tensor):
        raise TypeError(f"targets must be a torch.Tensor, got {type(targets).__name__}")

    if outputs.ndim != 2:
        raise ValueError(
            "outputs must have shape [batch_size, num_classes], "
            f"got {tuple(outputs.shape)}"
        )

    if targets.ndim != 1:
        raise ValueError(
            f"targets must have shape [batch_size], got {tuple(targets.shape)}"
        )

    if outputs.size(0) != targets.size(0):
        raise ValueError(
            f"Batch size mismatch: outputs={outputs.size(0)}, targets={targets.size(0)}"
        )

    integer_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }

    if targets.dtype not in integer_dtypes:
        raise TypeError(
            f"targets must contain integer class indices, got dtype={targets.dtype}"
        )

    if outputs.size(1) < 1:
        raise ValueError("outputs must contain at least one class")
