from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..metrics import PrecisionAverage, calculate_accuracy, calculate_precision

# I am thinking about moving all that to Pytorch Lightning so it handles the training, but I am not sure yet, maybe later.


class ClassificationTrainer:
    r"""ClassificationTrainer(model, optimizer=None, criterion=None, device=None, scheduler=None, scheduler_interval='epoch', logger=None, use_amp=False)

    Trainer for multiclass image classification models.

    Handles mixed precision via :class:`~torch.amp.GradScaler`, gradient accumulation,
    DDP-aware gradient synchronization, learning rate schedule stepping, validation metric evaluation,
    and safe checkpoint persistence.

    Args:
        model (nn.Module): Neural network model for training and evaluation.
        optimizer (Optimizer, optional): Optimizer instance. If ``None``, trainer operates in evaluation-only mode. Default: ``None``
        criterion (nn.Module, optional): Loss function module. Default: :class:`~torch.nn.CrossEntropyLoss`
        device (torch.device or str, optional): Target execution device. Default: ``'cuda'`` if available else ``'cpu'``
        scheduler (LRScheduler, optional): Learning rate scheduler instance. Default: ``None``
        scheduler_interval (str, optional): Scheduler stepping frequency (``'epoch'`` or ``'step'``). Default: ``'epoch'``
        logger (Logger, optional): Custom logger instance. Default: root logger for module.
        use_amp (bool, optional): Enable automatic mixed precision on CUDA. Default: ``False``

    Examples::

        >>> model = nn.Linear(10, 2)
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        >>> trainer = ClassificationTrainer(model=model, optimizer=optimizer)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        criterion: nn.Module | None = None,
        device: torch.device | str | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        scheduler_interval: Literal["epoch", "step"] = "epoch",
        logger: logging.Logger | None = None,
        use_amp: bool = False,
        amp_dtype: torch.dtype | None = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.to(memory_format=torch.channels_last)  # type: ignore[call-arg,no-matching-overload]
        self.optimizer = optimizer
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.scheduler = scheduler
        self.scheduler_interval = scheduler_interval
        self.logger = logger or logging.getLogger(__name__)

        self.use_amp = use_amp and self.device.type == "cuda"
        self.amp_dtype = amp_dtype or torch.float16
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp,
        )

    def _autocast(self) -> Any:
        return (
            torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
            )
            if self.use_amp
            else nullcontext()
        )

    def _model_for_checkpoint(self) -> nn.Module:
        """Return the underlying model when wrapped by DDP."""
        return getattr(self.model, "module", self.model)

    def _prepare_batch(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transfer a ``(inputs, targets)`` batch to the trainer device."""
        inputs, targets = batch
        inputs = inputs.to(
            self.device,
            memory_format=torch.channels_last,
            non_blocking=True,
        )
        targets = targets.to(
            self.device,
            non_blocking=True,
        )
        return inputs, targets

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        grad_accum_steps: int = 1,
    ) -> float:
        r"""train_epoch(dataloader, epoch, grad_accum_steps=1) -> float

        Execute a single training epoch across all batches in the dataloader.

        Args:
            dataloader (DataLoader): Training DataLoader yielding ``(inputs, targets)`` batches.
            epoch (int): Current 1-based epoch counter for progress display and logging.
            grad_accum_steps (int, optional): Number of micro-batches to accumulate before optimizer step. Default: ``1``

        Returns:
            float: Average unscaled training loss across all batches in the epoch.
        """
        if self.optimizer is None:
            raise ValueError(
                "ClassificationTrainer.train_epoch requires an optimizer, but self.optimizer is None."
            )

        try:
            num_loader_batches = len(dataloader)
        except TypeError:
            num_loader_batches = None

        if num_loader_batches == 0:
            return 0.0

        optimizer = self.optimizer
        self.model.train()
        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        num_batches = 0
        accumulation_count = 0

        progress = tqdm(
            dataloader,
            desc=f"Train Epoch {epoch}",
            leave=False,
        )

        for batch_idx, batch in enumerate(progress):
            inputs, targets = self._prepare_batch(batch)

            is_last_batch = batch_idx + 1 == num_loader_batches
            should_step = accumulation_count + 1 >= grad_accum_steps or is_last_batch

            no_sync = getattr(self.model, "no_sync", None)
            sync_context: Any = (
                no_sync()
                if (not should_step and callable(no_sync))
                else nullcontext()
            )

            with sync_context:
                with self._autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    scaled_loss = loss / grad_accum_steps

                scaled = self.scaler.scale(scaled_loss)
                cast(torch.Tensor, scaled).backward()

            accumulation_count += 1
            num_batches += 1
            total_loss += loss.detach().item()

            progress.set_postfix(loss=f"{loss.detach().item():.4f}")

            if should_step:
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accumulation_count = 0

                if self.scheduler is not None and self.scheduler_interval == "step":
                    self.scheduler.step()

        if self.scheduler is not None and self.scheduler_interval == "epoch":
            self.scheduler.step()

        mean_loss = total_loss / num_batches
        learning_rate = optimizer.param_groups[0]["lr"]

        self.logger.info(
            "epoch=%d train_loss=%.6f lr=%.6g",
            epoch,
            mean_loss,
            learning_rate,
        )

        return mean_loss

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        *,
        precision_average: PrecisionAverage = "macro",
    ) -> dict[str, float]:
        r"""evaluate(dataloader, *, precision_average='macro') -> dict

        Evaluate the classifier over the given evaluation dataset.

        Args:
            dataloader (DataLoader): Validation or test DataLoader yielding ``(inputs, targets)`` batches.
            precision_average (str, optional): Averaging reduction mode for precision (``'macro'``, ``'micro'``, ``'weighted'``). Default: ``'macro'``

        Returns:
            dict: Evaluation metrics dictionary containing:
                - ``'val_loss'``: Cross-entropy loss averaged per sample.
                - ``'val_acc'``: Top-1 classification accuracy in :math:`[0.0, 1.0]`.
                - ``'val_precision'``: Precision metric matching :attr:`precision_average`.
        """
        self.model.eval()

        total_loss = 0.0
        total_samples = 0
        all_outputs: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        for batch in dataloader:
            inputs, targets = self._prepare_batch(batch)

            with self._autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            batch_size = targets.size(0)
            total_loss += loss.detach().item() * batch_size
            total_samples += batch_size

            all_outputs.append(outputs.detach())
            all_targets.append(targets.detach())

        outputs = torch.cat(all_outputs, dim=0)
        targets = torch.cat(all_targets, dim=0)

        val_precision = calculate_precision(
            outputs,
            targets,
            average=precision_average,
        )
        # calculate_precision returns float for string average values;
        # runtime cast keeps the annotation accurate.
        val_precision_val = (
            float(val_precision)
            if isinstance(val_precision, (int, float))
            else float(val_precision.item())
        )

        metrics: dict[str, float] = {
            "val_loss": total_loss / total_samples,
            "val_acc": calculate_accuracy(outputs, targets),
            "val_precision": val_precision_val,
        }

        self.logger.info(
            "val_loss=%.6f val_acc=%.4f val_precision_%s=%.4f",
            metrics["val_loss"],
            metrics["val_acc"],
            precision_average,
            metrics["val_precision"],
        )

        return metrics

    def save_checkpoint(
        self,
        path: Path | str,
        epoch: int,
        extra: dict[str, Any] | None = None,
    ) -> None:
        r"""save_checkpoint(path, epoch, extra=None) -> None

        Persist current model weights, optimizer state, scheduler state, scaler state, and extra metadata.

        Args:
            path (Path or str): Target file path for the saved ``.pth`` checkpoint.
            epoch (int): Current epoch number to record in the checkpoint payload.
            extra (dict, optional): Extra metadata dictionary to merge into the checkpoint payload. Default: ``None``
        """
        payload: dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": (self._model_for_checkpoint().state_dict()),
            "scaler_state_dict": self.scaler.state_dict(),
        }

        if self.optimizer is not None:
            payload["optimizer_state_dict"] = self.optimizer.state_dict()

        if self.scheduler is not None:
            payload["scheduler_state_dict"] = self.scheduler.state_dict()

        if extra:
            reserved_keys = set(payload)
            collisions = reserved_keys.intersection(extra)

            if collisions:
                raise KeyError(
                    f"extra contains reserved checkpoint keys: {sorted(collisions)}"
                )

            payload.update(extra)

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        torch.save(payload, destination)

        self.logger.info(
            "saved checkpoint: %s",
            destination,
        )

    def load_checkpoint(
        self,
        path: Path | str,
        *,
        safe_load: bool = True,
    ) -> dict[str, Any]:
        r"""load_checkpoint(path, *, safe_load=True) -> dict

        Restore model, optimizer, scheduler, and scaler states from a saved checkpoint file.

        Args:
            path (Path or str): Path to the checkpoint file to load.
            safe_load (bool, optional): If ``True``, restricts unpickling to primitive types using PyTorch's weights_only loader. Default: ``True``

        Returns:
            dict: The deserialized checkpoint dictionary containing restored metadata and state dicts.
        """
        source = Path(path)
        checkpoint = torch.load(
            source,
            map_location=self.device,
            weights_only=safe_load,
        )

        self._model_for_checkpoint().load_state_dict(checkpoint["model_state_dict"])

        if self.optimizer is not None and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.logger.info(
            "loaded checkpoint: %s epoch=%s",
            source,
            checkpoint.get("epoch", "unknown"),
        )

        return checkpoint


__all__ = [
    "ClassificationTrainer",
]
