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
    r"""calculate_precision(outputs, targets, *, average='macro', num_classes=None, zero_division=0.0) -> Union[float, Tensor]

    Calculate multiclass precision from model prediction logits and integer ground-truth targets.

    Precision is computed as:

    .. math::
        \text{Precision}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c}

    Args:
        outputs (Tensor): Model predicted logits or class probabilities of shape :math:`(N, C)`.
        targets (Tensor): Ground-truth integer class labels of shape :math:`(N)`.
        average (str or None, optional): Averaging reduction strategy across classes:
            - ``'macro'``: Calculate precision for each class and calculate unweighted mean.
            - ``'micro'``: Calculate precision globally across all classes.
            - ``'weighted'``: Calculate precision for each class and average weighted by support.
            - ``None``: Return a 1D Tensor of shape :math:`(C)` with per-class precisions.
            Default: ``'macro'``
        num_classes (int, optional): Total number of target classes :math:`C`. If ``None``, inferred as :math:`\text{outputs.size}(1)`. Default: ``None``
        zero_division (float, optional): Value returned for classes with zero predicted positives. Default: ``0.0``

    Returns:
        float or Tensor: Scalar float when :attr:`average` is ``'macro'``, ``'micro'``, or ``'weighted'``;
        or a 1D Tensor of shape :math:`(C)` containing per-class precisions when :attr:`average` is ``None``.

    Examples::

        >>> outputs = torch.tensor([[10.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
        >>> targets = torch.tensor([0, 1, 1])
        >>> calculate_precision(outputs, targets, average='macro')
        0.75
    """
    num_classes = num_classes if num_classes is not None else outputs.size(1)

    if targets.numel() == 0:
        return (
            torch.full(
                (num_classes,),
                fill_value=zero_division,
                dtype=torch.float32,
                device=outputs.device,
            )
            if average is None
            else zero_division
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

        return (
            float((total_true_positives / total_predicted_positives).item())
            if total_predicted_positives > 0
            else zero_division
        )

    if average == "macro":
        return float(per_class_precision.mean().item())

    total_support = support.sum()
    return (
        float(((per_class_precision * support).sum() / total_support).item())
        if total_support > 0
        else zero_division
    )


__all__ = [
    "PrecisionAverage",
    "calculate_precision",
]
