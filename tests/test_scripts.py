from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from scripts import evaluate as eval_script
from scripts import train as train_script


@pytest.fixture
def dummy_dataset_dir(tmp_path: Path) -> Path:
    """Create a temporary dataset with train, val, and test splits."""
    for split in ["train", "val", "test"]:
        for cls_name in ["class_0", "class_1"]:
            cls_dir = tmp_path / split / cls_name
            cls_dir.mkdir(parents=True, exist_ok=True)
            for idx in range(2):
                img_arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
                img = Image.fromarray(img_arr)
                img.save(cls_dir / f"img_{idx}.png")
    return tmp_path


class TestScriptHelpers:
    """Behavioral tests for script helpers: config loading, device resolution, and validations."""

    def test_load_config(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "custom_config.yaml"
        cfg_file.write_text("project:\n  name: test_run\n", encoding="utf-8")

        data = train_script.load_config(cfg_file)
        assert data["project"]["name"] == "test_run"

        assert train_script.load_config(None) == {}

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            train_script.load_config(tmp_path / "non_existent.yaml")

    def test_resolve_device(self) -> None:
        dev_cpu = train_script.resolve_device("cpu")
        assert dev_cpu == torch.device("cpu")

        dev_auto = train_script.resolve_device("auto")
        assert isinstance(dev_auto, torch.device)

        if not torch.cuda.is_available():
            with pytest.raises(RuntimeError, match="CUDA was requested but is not available"):
                train_script.resolve_device("cuda")

    def test_model_name_choices_include_all_registered_models(self) -> None:
        from moliv_rl.models import MODEL_REGISTRY

        assert set(train_script.MODEL_REGISTRY.keys()) == set(MODEL_REGISTRY.keys())
        assert set(eval_script.MODEL_REGISTRY.keys()) == set(MODEL_REGISTRY.keys())


@pytest.fixture
def trained_checkpoint(
    dummy_dataset_dir: Path,
    tmp_path: Path,
) -> Path:
    """Train one epoch and return the saved best checkpoint path."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_args = argparse.Namespace(
        config=None,
        data_dir=dummy_dataset_dir,
        checkpoint_dir=checkpoint_dir,
        train_split="train",
        val_split="val",
        image_size=32,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        dataset_type="local",
        dataset_id="local",
        max_train_samples=None,
        max_val_samples=None,
        results_dir=tmp_path / "results",
        metrics_file=None,
        plots_dir=None,
        epochs=1,
        lr=0.01,
        weight_decay=0.0,
        optimizer="sgd",
        loss="cross_entropy",
        label_smoothing=0.0,
        scheduler="none",
        scheduler_interval="epoch",
        eta_min=0.0,
        model_name="classification_model",
        block_dims=[16, 32],
        in_channels=3,
        out_channels=32,
        patch_size=4,
        dropout=0.1,
        num_classes=2,
        optimize_model=False,
        seed=42069,
        grad_accum_steps=1,
        precision_average="macro",
        device="cpu",
        use_amp=False,
        save_best=True,
        save_last=True,
    )

    train_script.set_seeds(train_args.seed)
    device = train_script.resolve_device(train_args.device)
    logger = train_script.get_logger("test_train_pipeline")

    train_ds, val_ds = train_script.build_datasets(train_args)
    train_loader, val_loader = train_script.build_dataloaders(
        train_dataset=train_ds,
        val_dataset=val_ds,
        args=train_args,
        device=device,
    )

    trainer = train_script.build_trainer(
        args=train_args,
        device=device,
        logger=logger,
        steps_per_epoch=len(train_loader),
    )

    train_loss = trainer.train_epoch(train_loader, epoch=1)
    assert isinstance(train_loss, float)

    val_metrics = trainer.evaluate(val_loader)
    assert "val_acc" in val_metrics

    checkpoint_path = checkpoint_dir / "model_best.pth"
    trainer.save_checkpoint(
        checkpoint_path,
        epoch=1,
        extra={"best_val_acc": val_metrics["val_acc"]},
    )
    assert checkpoint_path.is_file()
    return checkpoint_path


class TestScriptsIntegration:
    """End-to-end integration tests for train.py and evaluate.py workflows."""

    def test_train_pipeline(
        self,
        dummy_dataset_dir: Path,
        tmp_path: Path,
    ) -> None:
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        train_args = argparse.Namespace(
            config=None,
            data_dir=dummy_dataset_dir,
            checkpoint_dir=checkpoint_dir,
            train_split="train",
            val_split="val",
            image_size=32,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            dataset_type="local",
            dataset_id="local",
            max_train_samples=None,
            max_val_samples=None,
            results_dir=tmp_path / "results",
            metrics_file=None,
            plots_dir=None,
            epochs=1,
            lr=0.01,
            weight_decay=0.0,
            optimizer="sgd",
            loss="cross_entropy",
            label_smoothing=0.0,
            scheduler="none",
            scheduler_interval="epoch",
            eta_min=0.0,
            model_name="classification_model",
            block_dims=[16, 32],
            in_channels=3,
            out_channels=32,
            patch_size=4,
            dropout=0.1,
            num_classes=2,
            optimize_model=False,
            seed=42069,
            grad_accum_steps=1,
            precision_average="macro",
            device="cpu",
            use_amp=False,
            save_best=True,
            save_last=True,
        )

        train_script.set_seeds(train_args.seed)

        device = train_script.resolve_device(train_args.device)
        logger = train_script.get_logger("test_train_pipeline")

        train_ds, val_ds = train_script.build_datasets(train_args)
        train_loader, val_loader = train_script.build_dataloaders(
            train_dataset=train_ds,
            val_dataset=val_ds,
            args=train_args,
            device=device,
        )

        trainer = train_script.build_trainer(
            args=train_args,
            device=device,
            logger=logger,
            steps_per_epoch=len(train_loader),
        )

        train_loss = trainer.train_epoch(train_loader, epoch=1)
        assert isinstance(train_loss, float)

        val_metrics = trainer.evaluate(val_loader)
        assert "val_acc" in val_metrics

        checkpoint_path = checkpoint_dir / "model_best.pth"
        trainer.save_checkpoint(
            checkpoint_path,
            epoch=1,
            extra={"best_val_acc": val_metrics["val_acc"]},
        )
        assert checkpoint_path.is_file()

    def test_evaluate_pipeline(
        self,
        dummy_dataset_dir: Path,
        trained_checkpoint: Path,
    ) -> None:
        checkpoint_path = trained_checkpoint
        device = torch.device("cpu")

        eval_args = argparse.Namespace(
            config=None,
            checkpoint=checkpoint_path,
            data_dir=dummy_dataset_dir,
            split="test",
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            model_name="classification_model",
            block_dims=[16, 32],
            in_channels=3,
            out_channels=32,
            patch_size=4,
            dropout=0.1,
            num_classes=2,
            optimize_model=False,
            image_size=32,
            precision_average="macro",
            device="cpu",
            use_amp=False,
            safe_load=True,
        )

        eval_logger = eval_script.get_logger("test_eval_pipeline")

        eval_trainer = eval_script.create_trainer(
            args=eval_args,
            device=device,
            logger=eval_logger,
        )
        eval_trainer.load_checkpoint(checkpoint_path, safe_load=True)

        eval_ds = eval_script.create_dataset(eval_args)
        eval_loader = eval_script.create_dataloader(
            dataset=eval_ds,
            args=eval_args,
            device=device,
        )

        test_metrics = eval_trainer.evaluate(eval_loader)
        assert "val_loss" in test_metrics
        assert "val_acc" in test_metrics
        assert "val_precision" in test_metrics
