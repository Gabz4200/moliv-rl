from __future__ import annotations

from typing import Any, Literal

import torch
from torch import nn

OptimizerName = Literal[
    "adam",
    "adamw",
    "sgd",
    "adagrad",
    "lamb",
    "lars",
    "lion",
    "rmsprop",
    "adam8bit",
    "adam32bit",
    "pagedadam",
    "pagedadam8bit",
    "pagedadam32bit",
    "adamw8bit",
    "adamw32bit",
    "pagedadamw",
    "pagedadamw8bit",
    "pagedadamw32bit",
    "adagrad8bit",
    "adagrad32bit",
    "lamb8bit",
    "lamb32bit",
    "lars8bit",
    "lars32bit",
    "pytorchlars",
    "lion8bit",
    "lion32bit",
    "pagedlion",
    "pagedlion8bit",
    "pagedlion32bit",
    "rmsprop8bit",
    "rmsprop32bit",
    "sgd8bit",
    "sgd32bit",
]

SchedulerName = Literal[
    "cosine_annealing",
    "cosine",
    "none",
]

LossName = Literal[
    "cross_entropy",
    "focal_loss",
    "focal",
    "invariance",
]


def build_optimizer(
    model: nn.Module | list[nn.Parameter] | Any,
    name: OptimizerName | str,
    learning_rate: float,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    r"""build_optimizer(model, name, learning_rate, weight_decay=0.0, momentum=0.9) -> Optimizer

    Create an optimizer instance by name, supporting both native PyTorch and
    8-bit/paged optimizers from ``bitsandbytes``.

    Supported names:
        - Native PyTorch: ``adam``, ``adamw``, ``sgd``, ``adagrad``,
          ``lamb``, ``lars``, ``lion``, ``rmsprop``
        - 8-bit/paged: ``adam8bit``, ``adam32bit``, ``pagedadam``,
          ``pagedadam8bit``, ``pagedadam32bit``, ``adamw8bit``,
          ``adamw32bit``, ``pagedadamw``, ``pagedadamw8bit``,
          ``pagedadamw32bit``, ``adagrad8bit``, ``adagrad32bit``,
          ``lamb8bit``, ``lamb32bit``, ``lars8bit``, ``lars32bit``,
          ``pytorchlars``, ``lion8bit``, ``lion32bit``, ``pagedlion``,
          ``pagedlion8bit``, ``pagedlion32bit``, ``rmsprop8bit``,
          ``rmsprop32bit``, ``sgd8bit``, ``sgd32bit``

    Args:
        model (nn.Module, list[nn.Parameter], or iterable): Model, parameter
            list, or parameter iterable whose parameters will be optimized.
        name (str): Optimizer algorithm name.
        learning_rate (float): Initial learning rate.
        weight_decay (float, optional): Weight decay factor. Default: ``0.0``
        momentum (float, optional): Momentum factor for SGD/LARS/LION-style
            optimizers. Default: ``0.9``

    Returns:
        torch.optim.Optimizer: Configured optimizer instance.

    Raises:
        ImportError: If a bitsandbytes optimizer is requested but
            ``bitsandbytes`` is not installed.
        ValueError: If the optimizer name is not recognized.
    """
    normalized_name = name.lower().replace("-", "_").replace(" ", "_")

    if isinstance(model, list):
        parameters = model
    elif hasattr(model, "parameters"):
        parameters = list(model.parameters())
    else:
        parameters = list(model)  # type: ignore[arg-type]

    native_optimizers: dict[str, Any] = {}
    if hasattr(torch.optim, "Adam"):
        native_optimizers["adam"] = lambda: torch.optim.Adam(
            parameters, lr=learning_rate, weight_decay=weight_decay
        )
    if hasattr(torch.optim, "AdamW"):
        native_optimizers["adamw"] = lambda: torch.optim.AdamW(
            parameters, lr=learning_rate, weight_decay=weight_decay
        )
    if hasattr(torch.optim, "SGD"):
        native_optimizers["sgd"] = lambda: torch.optim.SGD(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    if hasattr(torch.optim, "Adagrad"):
        native_optimizers["adagrad"] = lambda: torch.optim.Adagrad(
            parameters, lr=learning_rate, weight_decay=weight_decay
        )
    if hasattr(torch.optim, "LAMB"):
        native_optimizers["lamb"] = lambda: torch.optim.LAMB(
            parameters, lr=learning_rate, weight_decay=weight_decay
        )
    if hasattr(torch.optim, "LARS"):
        native_optimizers["lars"] = lambda: torch.optim.LARS(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    if hasattr(torch.optim, "Lion"):
        native_optimizers["lion"] = lambda: torch.optim.Lion(
            parameters, lr=learning_rate, weight_decay=weight_decay
        )
    if hasattr(torch.optim, "RMSprop"):
        native_optimizers["rmsprop"] = lambda: torch.optim.RMSprop(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
        )

    if normalized_name in native_optimizers:
        return native_optimizers[normalized_name]()

    bnb_optimizers = {
        "adam8bit": ("bitsandbytes.optim", "Adam8bit"),
        "adam32bit": ("bitsandbytes.optim", "Adam32bit"),
        "pagedadam": ("bitsandbytes.optim", "PagedAdam"),
        "pagedadam8bit": ("bitsandbytes.optim", "PagedAdam8bit"),
        "pagedadam32bit": ("bitsandbytes.optim", "PagedAdam32bit"),
        "adamw8bit": ("bitsandbytes.optim", "AdamW8bit"),
        "adamw32bit": ("bitsandbytes.optim", "AdamW32bit"),
        "pagedadamw": ("bitsandbytes.optim", "PagedAdamW"),
        "pagedadamw8bit": ("bitsandbytes.optim", "PagedAdamW8bit"),
        "pagedadamw32bit": ("bitsandbytes.optim", "PagedAdamW32bit"),
        "adagrad8bit": ("bitsandbytes.optim", "Adagrad8bit"),
        "adagrad32bit": ("bitsandbytes.optim", "Adagrad32bit"),
        "lamb8bit": ("bitsandbytes.optim", "LAMB8bit"),
        "lamb32bit": ("bitsandbytes.optim", "LAMB32bit"),
        "lars8bit": ("bitsandbytes.optim", "LARS8bit"),
        "lars32bit": ("bitsandbytes.optim", "LARS32bit"),
        "pytorchlars": ("bitsandbytes.optim", "PytorchLARS"),
        "lion8bit": ("bitsandbytes.optim", "Lion8bit"),
        "lion32bit": ("bitsandbytes.optim", "Lion32bit"),
        "pagedlion": ("bitsandbytes.optim", "PagedLion"),
        "pagedlion8bit": ("bitsandbytes.optim", "PagedLion8bit"),
        "pagedlion32bit": ("bitsandbytes.optim", "PagedLion32bit"),
        "rmsprop8bit": ("bitsandbytes.optim", "RMSprop8bit"),
        "rmsprop32bit": ("bitsandbytes.optim", "RMSprop32bit"),
        "sgd8bit": ("bitsandbytes.optim", "SGD8bit"),
        "sgd32bit": ("bitsandbytes.optim", "SGD32bit"),
    }

    if normalized_name not in bnb_optimizers:
        supported = sorted(set(native_optimizers) | set(bnb_optimizers))
        raise ValueError(
            f"Unsupported optimizer: {name!r}. "
            f"Supported: {supported}"
        )

    module_name, class_name = bnb_optimizers[normalized_name]
    try:
        import importlib

        module = importlib.import_module(module_name)
        optimizer_cls = getattr(module, class_name)
    except ImportError as exc:
        raise ImportError(
            f"Optimizer {name!r} requires bitsandbytes. "
            "Install it with: pip install bitsandbytes"
        ) from exc

    try:
        return optimizer_cls(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    except TypeError:
        return optimizer_cls(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    name: SchedulerName | str,
    scheduler_interval: Literal["epoch", "step"] = "epoch",
    epochs: int = 10,
    steps_per_epoch: int = 1,
    eta_min: float = 0.0,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    r"""build_scheduler(optimizer, name, scheduler_interval='epoch', epochs=10, steps_per_epoch=1, eta_min=0.0) -> LRScheduler | None

    Create a learning rate scheduler by name.

    Args:
        optimizer (Optimizer): Optimizer instance to wrap.
        name (str): Scheduler algorithm name.
        scheduler_interval (str, optional): Stepping frequency.
            Default: ``'epoch'``
        epochs (int, optional): Total number of epochs. Default: ``10``
        steps_per_epoch (int, optional): Number of optimizer steps per epoch.
            Default: ``1``
        eta_min (float, optional): Minimum learning rate for cosine annealing.
            Default: ``0.0``

    Returns:
        LRScheduler | None: Configured scheduler, or ``None`` if ``name`` is
            ``'none'``.
    """
    normalized_name = name.lower().replace("-", "_").replace(" ", "_")

    if normalized_name in ("none",):
        return None

    if normalized_name in ("cosine_annealing", "cosine"):
        if scheduler_interval == "step":
            t_max = epochs * max(steps_per_epoch, 1)
        else:
            t_max = epochs

        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=eta_min,
        )

    raise ValueError(
        f"Unsupported scheduler: {name!r}. "
        "Supported: cosine_annealing, cosine, none"
    )


__all__ = [
    "LossName",
    "OptimizerName",
    "SchedulerName",
    "build_optimizer",
    "build_scheduler",
]
