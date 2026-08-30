from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..metrics import PrecisionAverage, calculate_accuracy, calculate_precision

# I am thinking about moving all that to Pytorch Lightning so it handles the training, but I am not sure yet, maybe later.


class _BaseTrainer:
    r"""_BaseTrainer(model, optimizer, scheduler, scheduler_interval, logger, use_amp, amp_dtype, device)

    Shared base for :class:`ClassificationTrainer` and :class:`LeJepaTrainer`.

    Handles common device placement, mixed precision setup, checkpoint I/O,
    and DDP-aware model access.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        scheduler_interval: Literal["epoch", "step"],
        logger: logging.Logger,
        use_amp: bool,
        amp_dtype: torch.dtype | None,
        device: torch.device | str | None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.to(memory_format=torch.channels_last)  # type: ignore[call-arg,no-matching-overload]
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_interval = scheduler_interval
        self.logger = logger
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

    def save_checkpoint(
        self,
        path: Path | str,
        epoch: int,
        extra: dict[str, Any] | None = None,
        extra_modules: dict[str, nn.Module | None] | None = None,
    ) -> None:
        r"""save_checkpoint(path, epoch, extra=None, extra_modules=None) -> None

        Persist current model weights, optimizer state, scheduler state,
        scaler state, and extra metadata.

        Args:
            path (Path or str): Target file path for the saved ``.pth`` checkpoint.
            epoch (int): Current epoch number to record in the checkpoint payload.
            extra (dict, optional): Extra metadata dictionary to merge into the checkpoint payload. Default: ``None``
            extra_modules (dict, optional): Mapping of module names to modules whose state dicts should be saved. Default: ``None``
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

        if extra_modules:
            for module_name, module in extra_modules.items():
                if module is not None:
                    payload[f"{module_name}_state_dict"] = module.state_dict()

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
        extra_modules: dict[str, nn.Module | None] | None = None,
    ) -> dict[str, Any]:
        r"""load_checkpoint(path, *, safe_load=True, extra_modules=None) -> dict

        Restore model, optimizer, scheduler, and scaler states from a saved checkpoint file.

        Args:
            path (Path or str): Path to the checkpoint file to load.
            safe_load (bool, optional): If ``True``, restricts unpickling to primitive types using PyTorch's weights_only loader. Default: ``True``
            extra_modules (dict, optional): Mapping of module names to modules whose state dicts should be loaded. Default: ``None``

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

        if extra_modules:
            for module_name, module in extra_modules.items():
                if module is not None and f"{module_name}_state_dict" in checkpoint:
                    module.load_state_dict(checkpoint[f"{module_name}_state_dict"])

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


class ClassificationTrainer(_BaseTrainer):
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
        sigreg_loss_fn: nn.Module | None = None,
        sigreg_weight: float = 0.0,
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_interval=scheduler_interval,
            logger=logger or logging.getLogger(__name__),
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            device=device,
        )
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.sigreg_loss_fn = sigreg_loss_fn
        self.sigreg_weight = float(sigreg_weight)

    def _prepare_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        r"""_prepare_batch(batch) -> Tuple[Tensor, Tensor]

        Unpack a single training batch into inputs and targets on ``device``.

        Args:
            batch (tuple): Batch tuple containing inputs and targets.

        Returns:
            tuple: ``(inputs, targets)`` both moved to ``device``.
        """
        inputs, targets = batch
        return inputs.to(self.device), targets.to(self.device)

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        grad_accum_steps: int = 1,
    ) -> float:
        r"""train_epoch(dataloader, epoch, grad_accum_steps=1) -> float

        Execute one training epoch with optional gradient accumulation and mixed precision.

        Args:
            dataloader (DataLoader): Training data loader for the epoch.
            epoch (int): Current epoch index for logging purposes.
            grad_accum_steps (int, optional): Number of gradient accumulation steps before optimizer step. Default: ``1``

        Returns:
            float: Average training loss across all batches in the epoch.
        """
        if self.optimizer is None:
            raise ValueError(
                "ClassificationTrainer.train_epoch requires an optimizer. "
                "Instantiate with optimizer=... or use evaluate() for inference."
            )

        try:
            total_steps = len(dataloader)
        except (TypeError, RuntimeError):
            total_steps = 0

        if total_steps == 0:
            return 0.0

        self.model.train()
        total_loss = 0.0
        num_batches = 0
        accumulation_count = 0
        sync_context = cast(
            Callable[[], Any],
            self.model.no_sync if hasattr(self.model, "no_sync") else lambda: nullcontext(),
        )

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch:03d} [train]",
            leave=False,
        )

        for batch_idx, batch in enumerate(progress_bar, start=1):
            inputs, targets = self._prepare_batch(batch)

            with sync_context(), self._autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                if self.sigreg_loss_fn is not None:
                    sigreg_loss = self.sigreg_loss_fn(outputs)
                    loss = loss + self.sigreg_weight * sigreg_loss

            scaled_loss = self.scaler.scale(loss / grad_accum_steps)
            assert isinstance(scaled_loss, torch.Tensor)
            scaled_loss.backward()  # type: ignore[no-any-return, call-arg]

            accumulation_count += 1
            if accumulation_count >= grad_accum_steps or batch_idx == total_steps:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                accumulation_count = 0

                if (
                    self.scheduler is not None
                    and self.scheduler_interval == "step"
                ):
                    self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            current_loss = total_loss / num_batches
            progress_bar.set_postfix({"loss": f"{current_loss:.4f}"})

        mean_loss = total_loss / num_batches
        learning_rate = (
            self.scheduler.get_last_lr()[0]
            if self.scheduler is not None
            else float(self.optimizer.param_groups[0]["lr"])
        )

        if self.scheduler is not None and self.scheduler_interval == "epoch":
            self.scheduler.step()

        self.logger.info(
            "Epoch %03d | train_loss=%.6f | lr=%s",
            epoch,
            mean_loss,
            learning_rate,
        )

        return mean_loss

    def evaluate(
        self,
        dataloader: DataLoader,
        *,
        precision_average: PrecisionAverage = "macro",
    ) -> dict[str, float]:
        r"""evaluate(dataloader, *, precision_average='macro') -> dict

        Run evaluation over ``dataloader`` and return loss, accuracy, and precision.

        Args:
            dataloader (DataLoader): Validation or test data loader.
            precision_average (str, optional): Averaging strategy for precision metric. Default: ``'macro'``

        Returns:
            dict: Evaluation metrics with keys ``val_loss``, ``val_acc``, and ``val_precision``.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_outputs: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        progress_bar = tqdm(
            dataloader,
            desc="Evaluating",
            leave=False,
        )

        with torch.no_grad():
            for batch in progress_bar:
                inputs, targets = self._prepare_batch(batch)

                with self._autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                num_batches += 1

                all_outputs.append(outputs.detach().cpu())
                all_targets.append(targets.detach().cpu())

                progress_bar.set_postfix({"val_loss": f"{loss.item():.4f}"})

        if num_batches == 0:
            return {
                "val_loss": 0.0,
                "val_acc": 0.0,
                "val_precision": 0.0,
            }

        mean_loss = total_loss / num_batches
        outputs_cat = torch.cat(all_outputs, dim=0)
        targets_cat = torch.cat(all_targets, dim=0)

        accuracy = calculate_accuracy(outputs_cat, targets_cat)
        precision = float(calculate_precision(
            outputs_cat,
            targets_cat,
            average=cast(PrecisionAverage, precision_average),
        ))

        self.logger.info(
            "val_loss=%.6f val_acc=%.4f val_precision=%.4f",
            mean_loss,
            accuracy,
            precision,
        )

        return {
            "val_loss": mean_loss,
            "val_acc": accuracy,
            "val_precision": precision,
        }

    def save_checkpoint(
        self,
        path: Path | str,
        epoch: int,
        extra: dict[str, Any] | None = None,
        extra_modules: dict[str, nn.Module | None] | None = None,
    ) -> None:
        r"""save_checkpoint(path, epoch, extra=None) -> None

        Persist current model weights, optimizer state, scheduler state,
        scaler state, and extra metadata.

        Args:
            path (Path or str): Target file path for the saved ``.pth`` checkpoint.
            epoch (int): Current epoch number to record in the checkpoint payload.
            extra (dict, optional): Extra metadata dictionary to merge into the checkpoint payload. Default: ``None``
            extra_modules (dict, optional): Mapping of module names to modules whose state dicts should be saved. Default: ``None``
        """
        super().save_checkpoint(path, epoch, extra=extra, extra_modules=extra_modules)

    def load_checkpoint(
        self,
        path: Path | str,
        *,
        safe_load: bool = True,
        extra_modules: dict[str, nn.Module | None] | None = None,
    ) -> dict[str, Any]:
        r"""load_checkpoint(path, *, safe_load=True) -> dict

        Restore model, optimizer, scheduler, and scaler states from a saved checkpoint file.

        Args:
            path (Path or str): Path to the checkpoint file to load.
            safe_load (bool, optional): If ``True``, restricts unpickling to primitive types using PyTorch's weights_only loader. Default: ``True``
            extra_modules (dict, optional): Mapping of module names to modules whose state dicts should be loaded. Default: ``None``

        Returns:
            dict: The deserialized checkpoint dictionary containing restored metadata and state dicts.
        """
        return super().load_checkpoint(path, safe_load=safe_load, extra_modules=extra_modules)


class LeJepaTrainer(_BaseTrainer):
    r"""LeJepaTrainer(model, projector=None, optimizer=None, lejepa_criterion=None, probe_criterion=None, device=None, scheduler=None, scheduler_interval='epoch', logger=None, use_amp=False)

    Trainer for self-supervised LeJEPA learning with optional linear probe evaluation.

    Handles mixed precision via :class:`~torch.amp.GradScaler`, gradient accumulation,
    DDP-aware gradient synchronization, learning rate schedule stepping, validation metric evaluation,
    and safe checkpoint persistence for both backbone and projector weights.

    Args:
        model (nn.Module): Backbone neural network model for training and evaluation.
        projector (nn.Module, optional): Optional projector head for LeJEPA loss. Default: ``None``
        optimizer (Optimizer, optional): Optimizer instance. Default: ``None``
        lejepa_criterion (nn.Module, optional): Self-supervised loss function combining invariance and SIGReg. Default: ``None``
        probe_criterion (nn.Module, optional): Supervised linear probe loss for validation evaluation. Default: ``None``
        device (torch.device or str, optional): Target execution device. Default: ``'cuda'`` if available else ``'cpu'``
        scheduler (LRScheduler, optional): Learning rate scheduler instance. Default: ``None``
        scheduler_interval (str, optional): Scheduler stepping frequency (``'epoch'`` or ``'step'``). Default: ``'epoch'``
        logger (Logger, optional): Custom logger instance. Default: root logger for module.
        use_amp (bool, optional): Enable automatic mixed precision on CUDA. Default: ``False``
    """

    def __init__(
        self,
        model: nn.Module,
        projector: nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        lejepa_criterion: nn.Module | None = None,
        probe_criterion: nn.Module | None = None,
        device: torch.device | str | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        scheduler_interval: Literal["epoch", "step"] = "epoch",
        logger: logging.Logger | None = None,
        use_amp: bool = False,
        amp_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_interval=scheduler_interval,
            logger=logger or logging.getLogger(__name__),
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            device=device,
        )
        self.projector = projector.to(self.device) if projector is not None else None
        self.lejepa_criterion = lejepa_criterion
        self.probe_criterion = probe_criterion

    def _prepare_views(
        self, views: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""_prepare_views(views) -> Tuple[Tensor, Tensor]

        Unpack a multi-view batch into the view tensor and dummy targets on ``device``.

        Args:
            views (Tensor): Batch of augmented views shaped :math:`(B, V, C, H, W)`.

        Returns:
            tuple: ``(views, targets)`` with views on ``device`` and targets as zeros on ``device``.
        """
        batch_size = views.size(0)
        targets = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        return views.to(self.device), targets

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        grad_accum_steps: int = 1,
    ) -> dict[str, float]:
        r"""train_epoch(dataloader, epoch, grad_accum_steps=1) -> dict

        Execute one self-supervised training epoch with optional gradient accumulation and mixed precision.

        Args:
            dataloader (DataLoader): Training data loader for the epoch.
            epoch (int): Current epoch index for logging purposes.
            grad_accum_steps (int, optional): Number of gradient accumulation steps before optimizer step. Default: ``1``

        Returns:
            dict: Average training losses with keys ``train_loss``, ``invariance_loss``, ``sigreg_loss``, and ``probe_loss``.
        """
        if self.optimizer is None:
            raise ValueError(
                "LeJepaTrainer.train_epoch requires an optimizer. "
                "Instantiate with optimizer=... or use evaluate_probe() for inference."
            )

        try:
            total_steps = len(dataloader)
        except (TypeError, RuntimeError):
            total_steps = 0

        if total_steps == 0:
            return {
                "train_loss": 0.0,
                "invariance_loss": 0.0,
                "sigreg_loss": 0.0,
                "probe_loss": 0.0,
            }

        self.model.train()
        if self.projector is not None:
            self.projector.train()

        total_loss = 0.0
        total_invariance = 0.0
        total_sigreg = 0.0
        total_probe = 0.0
        num_batches = 0
        accumulation_count = 0
        sync_context = cast(
            Callable[[], Any],
            self.model.no_sync if hasattr(self.model, "no_sync") else lambda: nullcontext(),
        )

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch:03d} [train]",
            leave=False,
        )

        for batch_idx, batch in enumerate(progress_bar, start=1):
            views, _ = self._prepare_views(batch)

            with sync_context(), self._autocast():
                features = self.model(views)
                if self.projector is not None:
                    projections = self.projector(features)
                else:
                    projections = features

                lejepa_loss = torch.tensor(0.0, device=self.device)
                invariance_loss = torch.tensor(0.0, device=self.device)
                sigreg_loss = torch.tensor(0.0, device=self.device)
                if self.lejepa_criterion is not None:
                    lejepa_loss, invariance_loss, sigreg_loss = (
                        self.lejepa_criterion(projections)
                    )

                probe_loss = torch.tensor(0.0, device=self.device)
                if self.probe_criterion is not None:
                    probe_out = self.model(views[:, 0])
                    probe_loss = self.probe_criterion(probe_out, _)

                loss = lejepa_loss + probe_loss

            scaled_loss = self.scaler.scale(loss / grad_accum_steps)
            assert isinstance(scaled_loss, torch.Tensor)
            scaled_loss.backward()

            accumulation_count += 1
            if accumulation_count >= grad_accum_steps or batch_idx == total_steps:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                accumulation_count = 0

                if (
                    self.scheduler is not None
                    and self.scheduler_interval == "step"
                ):
                    self.scheduler.step()

            total_loss += loss.item()
            total_invariance += invariance_loss.item()
            total_sigreg += sigreg_loss.item()
            total_probe += probe_loss.item()
            num_batches += 1

            current_loss = total_loss / num_batches
            progress_bar.set_postfix(
                {
                    "loss": f"{current_loss:.4f}",
                    "inv": f"{total_invariance / num_batches:.4f}",
                    "sigreg": f"{total_sigreg / num_batches:.4f}",
                    "probe": f"{total_probe / num_batches:.4f}",
                }
            )

        mean_loss = total_loss / num_batches
        mean_invariance = total_invariance / num_batches
        mean_sigreg = total_sigreg / num_batches
        mean_probe = total_probe / num_batches
        learning_rate = (
            self.scheduler.get_last_lr()[0]
            if self.scheduler is not None
            else float(self.optimizer.param_groups[0]["lr"])
        )

        if self.scheduler is not None and self.scheduler_interval == "epoch":
            self.scheduler.step()

        self.logger.info(
            "Epoch %03d | train_loss=%.6f | inv=%.6f | sigreg=%.6f | probe=%.6f | lr=%s",
            epoch,
            mean_loss,
            mean_invariance,
            mean_sigreg,
            mean_probe,
            learning_rate,
        )

        return {
            "train_loss": mean_loss,
            "invariance_loss": mean_invariance,
            "sigreg_loss": mean_sigreg,
            "probe_loss": mean_probe,
        }

    def evaluate_probe(
        self,
        dataloader: DataLoader,
        *,
        precision_average: PrecisionAverage = "macro",
    ) -> dict[str, float]:
        r"""evaluate_probe(dataloader, *, precision_average='macro') -> dict

        Run supervised linear-probe evaluation over ``dataloader`` and return loss, accuracy, and precision.

        Args:
            dataloader (DataLoader): Validation or test data loader.
            precision_average (str, optional): Averaging strategy for precision metric. Default: ``'macro'``

        Returns:
            dict: Evaluation metrics with keys ``val_loss``, ``val_acc``, and ``val_precision``.
        """
        self.model.eval()
        if self.projector is not None:
            self.projector.eval()

        total_loss = 0.0
        num_batches = 0
        all_outputs: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        progress_bar = tqdm(
            dataloader,
            desc="Evaluating probe",
            leave=False,
        )

        with torch.no_grad():
            for batch in progress_bar:
                views, targets = self._prepare_views(batch)

                with self._autocast():
                    features = self.model(views)
                    if self.projector is not None:
                        features = self.projector(features)
                    outputs = (
                        self.probe_criterion(features)
                        if self.probe_criterion is not None
                        else features
                    )
                    loss = (
                        self.probe_criterion(features, targets)
                        if self.probe_criterion is not None
                        else torch.tensor(0.0, device=self.device)
                    )

                total_loss += loss.item()
                num_batches += 1

                all_outputs.append(outputs.detach().cpu())
                all_targets.append(targets.detach().cpu())

                progress_bar.set_postfix({"val_loss": f"{loss.item():.4f}"})

        if num_batches == 0:
            return {
                "val_loss": 0.0,
                "val_acc": 0.0,
                "val_precision": 0.0,
            }

        mean_loss = total_loss / num_batches
        outputs_cat = torch.cat(all_outputs, dim=0)
        targets_cat = torch.cat(all_targets, dim=0)

        accuracy = calculate_accuracy(outputs_cat, targets_cat)
        precision = float(calculate_precision(
            outputs_cat,
            targets_cat,
            average=cast(PrecisionAverage, precision_average),
        ))

        self.logger.info(
            "val_loss=%.6f val_acc=%.4f val_precision=%.4f",
            mean_loss,
            accuracy,
            precision,
        )

        return {
            "val_loss": mean_loss,
            "val_acc": accuracy,
            "val_precision": precision,
        }

    def save_checkpoint(
        self,
        path: Path | str,
        epoch: int,
        extra: dict[str, Any] | None = None,
        extra_modules: dict[str, nn.Module | None] | None = None,
    ) -> None:
        r"""save_checkpoint(path, epoch, extra=None) -> None

        Persist current backbone, projector, optimizer state, scheduler state,
        scaler state, and extra metadata.

        Args:
            path (Path or str): Target file path for the saved ``.pth`` checkpoint.
            epoch (int): Current epoch number to record in the checkpoint payload.
            extra (dict, optional): Extra metadata dictionary to merge into the checkpoint payload. Default: ``None``
            extra_modules (dict, optional): Mapping of module names to modules whose state dicts should be saved. Default: ``None``
        """
        super().save_checkpoint(
            path,
            epoch,
            extra=extra,
            extra_modules={"projector": self.projector},
        )

    def load_checkpoint(
        self,
        path: Path | str,
        *,
        safe_load: bool = True,
        extra_modules: dict[str, nn.Module | None] | None = None,
    ) -> dict[str, Any]:
        r"""load_checkpoint(path, *, safe_load=True) -> dict

        Restore backbone, projector, optimizer, scheduler, and scaler states from a saved checkpoint file.

        Args:
            path (Path or str): Path to the checkpoint file to load.
            safe_load (bool, optional): If ``True``, restricts unpickling to primitive types using PyTorch's weights_only loader. Default: ``True``
            extra_modules (dict, optional): Mapping of module names to modules whose state dicts should be loaded. Default: ``None``

        Returns:
            dict: The deserialized checkpoint dictionary containing restored metadata and state dicts.
        """
        return super().load_checkpoint(
            path,
            safe_load=safe_load,
            extra_modules={"projector": self.projector},
        )


__all__ = [
    "ClassificationTrainer",
    "LeJepaTrainer",
]