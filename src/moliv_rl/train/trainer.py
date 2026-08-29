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
        sigreg_loss_fn: nn.Module | None = None,
        sigreg_weight: float = 0.0,
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

        self.sigreg_loss_fn = sigreg_loss_fn
        self.sigreg_weight = float(sigreg_weight)

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

                    if self.sigreg_loss_fn is not None and self.sigreg_weight > 0.0:
                        embedding_fn = getattr(self.model, "get_embedding", None)
                        if embedding_fn is not None:
                            features = embedding_fn(inputs)
                            sigreg = self.sigreg_loss_fn(features)
                            loss = loss + self.sigreg_weight * sigreg

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

        total_sigreg_loss = 0.0

        for batch in dataloader:
            inputs, targets = self._prepare_batch(batch)

            with self._autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                if self.sigreg_loss_fn is not None and self.sigreg_weight > 0.0:
                    embedding_fn = getattr(self.model, "get_embedding", None)
                    if embedding_fn is not None:
                        features = embedding_fn(inputs)
                        sigreg = self.sigreg_loss_fn(features)
                        loss = loss + self.sigreg_weight * sigreg
                        total_sigreg_loss += sigreg.detach().item() * targets.size(0)

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

        if self.sigreg_loss_fn is not None and self.sigreg_weight > 0.0:
            metrics["val_sigreg_loss"] = total_sigreg_loss / total_samples

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


class LeJepaTrainer:
    r"""LeJepaTrainer(model, projector, optimizer=None, lejepa_criterion=None, probe_criterion=None, device=None, scheduler=None, scheduler_interval='epoch', logger=None, use_amp=False)

    Trainer for LeJEPA self-supervised learning with optional online linear probing.

    Handles mixed precision via :class:`~torch.amp.GradScaler`, gradient accumulation,
    and learning rate schedule stepping for the LeJEPA objective.

    Args:
        model (nn.Module): Backbone encoder network.
        projector (nn.Module, optional): Projection head mapping encoder
            outputs to the LeJEPA embedding space. Default: ``None``
        optimizer (Optimizer, optional): Optimizer instance. If ``None``, trainer
            operates in evaluation-only mode. Default: ``None``
        lejepa_criterion (nn.Module, optional): LeJEPA loss module combining
            invariance and SIGReg terms. Default: ``None``
        probe_criterion (nn.Module, optional): Optional online linear probe
            loss for supervised evaluation. Default: ``None``
        device (torch.device or str, optional): Target execution device.
            Default: ``'cuda'`` if available else ``'cpu'``
        scheduler (LRScheduler, optional): Learning rate scheduler instance.
            Default: ``None``
        scheduler_interval (str, optional): Scheduler stepping frequency
            (``'epoch'`` or ``'step'``). Default: ``'epoch'``
        logger (Logger, optional): Custom logger instance. Default: root logger
            for module.
        use_amp (bool, optional): Enable automatic mixed precision on CUDA.
            Default: ``False``

    Examples::

        >>> model = MyModel(...)
        >>> projector = nn.Sequential(nn.Linear(512, 2048), nn.ReLU(), nn.Linear(2048, 128))
        >>> optimizer = torch.optim.AdamW(list(model.parameters()) + list(projector.parameters()))
        >>> trainer = LeJepaTrainer(model=model, projector=projector, optimizer=optimizer)
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
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.projector = projector.to(self.device) if projector is not None else None
        self.optimizer = optimizer
        self.lejepa_criterion = lejepa_criterion
        self.probe_criterion = probe_criterion
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

    def _prepare_views(self, views: torch.Tensor) -> torch.Tensor:
        """Transfer a batch of views to the trainer device."""
        return views.to(
            self.device,
            memory_format=torch.channels_last,
            non_blocking=True,
        )

    @torch.no_grad()
    def evaluate_probe(
        self,
        dataloader: DataLoader,
        *,
        precision_average: PrecisionAverage = "macro",
    ) -> dict[str, float]:
        r"""evaluate_probe(dataloader, *, precision_average='macro') -> dict

        Evaluate the online linear probe over the given dataset.

        Args:
            dataloader (DataLoader): DataLoader yielding ``(views, targets)``
                batches where ``views`` has shape ``(V, N, C, H, W)``.
            precision_average (str, optional): Averaging reduction mode for
                precision. Default: ``'macro'``

        Returns:
            dict: Evaluation metrics dictionary containing:
                - ``'val_loss'``: Probe loss averaged per sample.
                - ``'val_acc'``: Top-1 classification accuracy.
                - ``'val_precision'``: Precision metric matching
                  :attr:`precision_average`.
        """
        if self.projector is None or self.probe_criterion is None:
            raise ValueError(
                "evaluate_probe requires both projector and probe_criterion."
            )

        self.model.eval()
        self.projector.eval()

        total_loss = 0.0
        total_samples = 0
        all_outputs: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        for batch in dataloader:
            views, targets = batch
            views = self._prepare_views(views)
            targets = targets.to(self.device, non_blocking=True)

            with self._autocast():
                # Use first view for evaluation
                embeddings = self.model(views[0])
                projections = self.projector(embeddings)
                logits = self.probe_criterion(projections)
                loss = nn.functional.cross_entropy(logits, targets)

            batch_size = targets.size(0)
            total_loss += loss.detach().item() * batch_size
            total_samples += batch_size

            all_outputs.append(logits.detach())
            all_targets.append(targets.detach())

        outputs = torch.cat(all_outputs, dim=0)
        targets = torch.cat(all_targets, dim=0)

        val_precision = calculate_precision(
            outputs,
            targets,
            average=precision_average,
        )
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

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        grad_accum_steps: int = 1,
    ) -> dict[str, float]:
        r"""train_epoch(dataloader, epoch, grad_accum_steps=1) -> dict

        Execute a single LeJEPA training epoch.

        Args:
            dataloader (DataLoader): Training DataLoader yielding
                ``(views, targets)`` batches where ``views`` has shape
                ``(V, N, C, H, W)``.
            epoch (int): Current 1-based epoch counter.
            grad_accum_steps (int, optional): Number of micro-batches to
                accumulate before optimizer step. Default: ``1``

        Returns:
            dict: Average losses for the epoch with keys:
                - ``'train_loss'``: Combined LeJEPA loss
                - ``'invariance_loss'``: Invariance component
                - ``'sigreg_loss'``: SIGReg component
                - ``'probe_loss'``: Probe loss (if probe is active)
        """
        if self.optimizer is None:
            raise ValueError(
                "LeJepaTrainer.train_epoch requires an optimizer, but self.optimizer is None."
            )

        try:
            num_loader_batches = len(dataloader)
        except TypeError:
            num_loader_batches = None

        if num_loader_batches == 0:
            return {
                "train_loss": 0.0,
                "invariance_loss": 0.0,
                "sigreg_loss": 0.0,
                "probe_loss": 0.0,
            }

        model = self.model
        projector = self.projector
        optimizer = self.optimizer
        lejepa_criterion = self.lejepa_criterion

        model.train()
        if projector is not None:
            projector.train()

        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        total_inv_loss = 0.0
        total_sigreg_loss = 0.0
        total_probe_loss = 0.0
        num_batches = 0
        accumulation_count = 0

        progress = tqdm(
            dataloader,
            desc=f"Train Epoch {epoch}",
            leave=False,
        )

        for batch_idx, batch in enumerate(progress):
            views, targets = batch
            views = self._prepare_views(views)
            targets = targets.to(self.device, non_blocking=True)

            is_last_batch = batch_idx + 1 == num_loader_batches
            should_step = (
                accumulation_count + 1 >= grad_accum_steps or is_last_batch
            )

            no_sync = getattr(model, "no_sync", None)
            sync_context: Any = (
                no_sync()
                if (not should_step and callable(no_sync))
                else nullcontext()
            )

            with sync_context:
                with self._autocast():
                    # Encode all views
                    V = views.size(0)
                    embeddings = model(views.flatten(0, 1))
                    embeddings = embeddings.view(V, -1, embeddings.size(-1))

                    # Project embeddings
                    if projector is not None:
                        projections = projector(embeddings)
                    else:
                        projections = embeddings

                    # LeJEPA loss
                    if lejepa_criterion is not None:
                        lejepa_total, inv_loss, sigreg_loss = lejepa_criterion(
                            projections
                        )
                    else:
                        lejepa_total = torch.tensor(0.0, device=self.device)
                        inv_loss = torch.tensor(0.0, device=self.device)
                        sigreg_loss = torch.tensor(0.0, device=self.device)

                    # Online probe loss
                    probe_loss = torch.tensor(0.0, device=self.device)
                    if self.probe_criterion is not None:
                        # Use first view's embeddings for probing
                        probe_logits = self.probe_criterion(
                            projections[0].detach()
                            if projector is not None
                            else embeddings[0].detach()
                        )
                        probe_loss = nn.functional.cross_entropy(
                            probe_logits, targets
                        )

                    loss = lejepa_total + probe_loss
                    scaled_loss = loss / grad_accum_steps

                scaled = self.scaler.scale(scaled_loss)
                cast(torch.Tensor, scaled).backward()

            accumulation_count += 1
            num_batches += 1
            total_loss += loss.detach().item()
            total_inv_loss += inv_loss.detach().item()
            total_sigreg_loss += sigreg_loss.detach().item()
            total_probe_loss += probe_loss.detach().item()

            progress.set_postfix(
                loss=f"{loss.detach().item():.4f}",
                inv=f"{inv_loss.detach().item():.4f}",
                sigreg=f"{sigreg_loss.detach().item():.4f}",
                probe=f"{probe_loss.detach().item():.4f}",
            )

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
            "epoch=%d train_loss=%.6f inv=%.6f sigreg=%.6f probe=%.6f lr=%.6g",
            epoch,
            mean_loss,
            total_inv_loss / num_batches,
            total_sigreg_loss / num_batches,
            total_probe_loss / num_batches,
            learning_rate,
        )

        return {
            "train_loss": mean_loss,
            "invariance_loss": total_inv_loss / num_batches,
            "sigreg_loss": total_sigreg_loss / num_batches,
            "probe_loss": total_probe_loss / num_batches,
        }

    def save_checkpoint(
        self,
        path: Path | str,
        epoch: int,
        extra: dict[str, Any] | None = None,
    ) -> None:
        r"""save_checkpoint(path, epoch, extra=None) -> None

        Persist current model weights, optimizer state, scheduler state,
        scaler state, and extra metadata.

        Args:
            path (Path or str): Target file path for the saved ``.pth``
                checkpoint.
            epoch (int): Current epoch number to record.
            extra (dict, optional): Extra metadata dictionary to merge into
                the checkpoint payload. Default: ``None``
        """
        payload: dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
        }

        if self.projector is not None:
            payload["projector_state_dict"] = self.projector.state_dict()

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

        Restore model, projector, optimizer, scheduler, and scaler states
        from a saved checkpoint file.

        Args:
            path (Path or str): Path to the checkpoint file to load.
            safe_load (bool, optional): If ``True``, restricts unpickling to
                primitive types using PyTorch's weights_only loader.
                Default: ``True``

        Returns:
            dict: The deserialized checkpoint dictionary.
        """
        source = Path(path)
        checkpoint = torch.load(
            source,
            map_location=self.device,
            weights_only=safe_load,
        )

        model = self._model_for_checkpoint()
        model.load_state_dict(checkpoint["model_state_dict"])

        if self.projector is not None and "projector_state_dict" in checkpoint:
            self.projector.load_state_dict(checkpoint["projector_state_dict"])

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

    def _model_for_checkpoint(self) -> nn.Module:
        """Return the underlying model when wrapped by DDP."""
        return getattr(self.model, "module", self.model)


__all__ = [
    "ClassificationTrainer",
    "LeJepaTrainer",
]
