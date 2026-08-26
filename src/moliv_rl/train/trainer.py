from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..metrics import PrecisionAverage, calculate_accuracy, calculate_precision

# I am thinking about moving all that to Pytorch Lightning so it handles the training, but I am not sure yet, maybe later.


class ClassificationTrainer:
    """Simple multiclass classification trainer.

    Supports device placement, gradient accumulation, optional automatic
    mixed precision, learning-rate scheduling, DDP-aware accumulation,
    validation metrics, and checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device | str | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        scheduler_interval: Literal["epoch", "step"] = "epoch",
        logger: logging.Logger | None = None,
        use_amp: bool = False,
    ) -> None:
        if scheduler_interval not in {"epoch", "step"}:
            raise ValueError("scheduler_interval must be 'epoch' or 'step'")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.scheduler_interval = scheduler_interval
        self.logger = logger or logging.getLogger(__name__)

        self.use_amp = use_amp and self.device.type == "cuda"
        self.amp_dtype = torch.float16
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp,
        )

    def _autocast(self):
        if not self.use_amp:
            return nullcontext()

        return torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
        )

    def _model_for_checkpoint(self) -> nn.Module:
        """Return the underlying model when wrapped by DDP."""
        return getattr(self.model, "module", self.model)

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        grad_accum_steps: int = 1,
    ) -> float:
        """Run one training epoch and return the mean batch loss."""
        if epoch < 1:
            raise ValueError(f"epoch must be >= 1, got {epoch}")

        if grad_accum_steps < 1:
            raise ValueError(f"grad_accum_steps must be >= 1, got {grad_accum_steps}")

        try:
            num_loader_batches = len(dataloader)
        except TypeError as exc:
            raise TypeError(
                "train_epoch requires a dataloader with a defined length"
            ) from exc

        if num_loader_batches == 0:
            raise ValueError("dataloader must not be empty")

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        num_batches = 0
        accumulation_count = 0

        progress = tqdm(
            dataloader,
            desc=f"Train Epoch {epoch}",
            leave=False,
        )

        for batch_idx, batch in enumerate(progress):
            inputs, targets = batch

            inputs = inputs.to(
                self.device,
                non_blocking=True,
            )
            targets = targets.to(
                self.device,
                non_blocking=True,
            )

            is_last_batch = batch_idx + 1 == num_loader_batches
            should_step = accumulation_count + 1 >= grad_accum_steps or is_last_batch

            sync_context = nullcontext()
            if not should_step and hasattr(self.model, "no_sync"):
                sync_context = self.model.no_sync()

            with sync_context:
                with self._autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    scaled_loss = loss / grad_accum_steps

                self.scaler.scale(scaled_loss).backward()

            accumulation_count += 1
            num_batches += 1
            total_loss += loss.detach().item()

            progress.set_postfix(loss=f"{loss.detach().item():.4f}")

            if should_step:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                accumulation_count = 0

                if self.scheduler is not None and self.scheduler_interval == "step":
                    self.scheduler.step()

        if self.scheduler is not None and self.scheduler_interval == "epoch":
            self.scheduler.step()

        mean_loss = total_loss / num_batches
        learning_rate = self.optimizer.param_groups[0]["lr"]

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
        """Evaluate a multiclass classifier.

        The model must return logits with shape ``[N, C]`` and targets must
        have shape ``[N]``.
        """
        if precision_average not in {
            "micro",
            "macro",
            "weighted",
        }:
            raise ValueError(
                "precision_average must be one of "
                "'micro', 'macro', or 'weighted'; "
                f"got {precision_average!r}"
            )

        self.model.eval()

        total_loss = 0.0
        total_samples = 0
        all_outputs: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        for batch in dataloader:
            inputs, targets = batch

            inputs = inputs.to(
                self.device,
                non_blocking=True,
            )
            targets = targets.to(
                self.device,
                non_blocking=True,
            )

            with self._autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            if outputs.ndim != 2:
                raise ValueError(
                    "outputs must have shape [batch_size, num_classes], "
                    f"got {tuple(outputs.shape)}"
                )

            if targets.ndim != 1:
                raise ValueError(
                    f"targets must have shape [batch_size], got {tuple(targets.shape)}"
                )

            batch_size = targets.size(0)

            if outputs.size(0) != batch_size:
                raise ValueError(
                    "Batch size mismatch: "
                    f"outputs={outputs.size(0)}, "
                    f"targets={batch_size}"
                )

            total_loss += loss.detach().item() * batch_size
            total_samples += batch_size

            all_outputs.append(outputs.detach())
            all_targets.append(targets.detach())

        if total_samples == 0:
            raise ValueError("dataloader yielded zero samples")

        outputs = torch.cat(all_outputs, dim=0)
        targets = torch.cat(all_targets, dim=0)

        metrics = {
            "val_loss": total_loss / total_samples,
            "val_acc": calculate_accuracy(outputs, targets),
            "val_precision": calculate_precision(
                outputs,
                targets,
                average=precision_average,
            ),
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
        """Save model, optimizer, scheduler, AMP, and metadata state."""
        if epoch < 1:
            raise ValueError(f"epoch must be >= 1, got {epoch}")

        payload: dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": (self._model_for_checkpoint().state_dict()),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
        }

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
        """Load a checkpoint and return its stored metadata.

        Set ``safe_load=False`` only for trusted checkpoints requiring
        unrestricted pickle deserialization.
        """
        source = Path(path)

        if not source.is_file():
            raise FileNotFoundError(f"checkpoint not found at {source}")

        checkpoint = torch.load(
            source,
            map_location=self.device,
            weights_only=safe_load,
        )

        if not isinstance(checkpoint, dict):
            raise TypeError("checkpoint must contain a dictionary")

        if "model_state_dict" not in checkpoint:
            raise KeyError("checkpoint is missing 'model_state_dict'")

        self._model_for_checkpoint().load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
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
