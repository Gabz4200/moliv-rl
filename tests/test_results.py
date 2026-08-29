from __future__ import annotations

import argparse
from pathlib import Path

from scripts import train as train_script


class TestBuildResultPaths:
    def test_default_paths_use_timestamp(self, tmp_path: Path) -> None:
        paths = train_script._build_result_paths(
            results_dir=tmp_path,
            metrics_file=None,
            plots_dir=None,
        )
        assert paths.metrics_file.parent == tmp_path
        assert paths.metrics_file.name.startswith("metrics_")
        assert paths.plots_dir.parent == tmp_path
        assert paths.plots_dir.name.startswith("plots_")

    def test_explicit_paths_are_respected(self, tmp_path: Path) -> None:
        metrics = tmp_path / "custom" / "metrics.txt"
        plots = tmp_path / "custom" / "plots"
        paths = train_script._build_result_paths(
            results_dir=tmp_path,
            metrics_file=metrics,
            plots_dir=plots,
        )
        assert paths.metrics_file == metrics
        assert paths.plots_dir == plots
        assert paths.plots_dir.exists()


class TestSaveMetrics:
    def test_metrics_file_content(self, tmp_path: Path) -> None:
        metrics_file = tmp_path / "metrics.txt"
        args = argparse.Namespace(
            config=Path("configs/cpu_test.yaml"),
            dataset_type="streaming",
            dataset_id="test/dataset",
            train_split="train",
            val_split="val",
            image_size=256,
            batch_size=1,
            device="cpu",
            epochs=3,
            grad_accum_steps=1,
            use_amp=False,
        )
        history = [
            {
                "epoch": 1,
                "train_loss": 1.0,
                "val_loss": 1.1,
                "val_acc": 0.5,
                "val_precision": 0.5,
                "lr": 0.001,
            },
            {
                "epoch": 2,
                "train_loss": 0.8,
                "val_loss": 0.9,
                "val_acc": 0.6,
                "val_precision": 0.6,
                "lr": 0.001,
            },
        ]
        train_script._save_metrics(
            metrics_file=metrics_file,
            args=args,
            history=history,
            best_val_acc=0.6,
            best_epoch=2,
        )
        text = metrics_file.read_text(encoding="utf-8")
        assert "dataset_id: test/dataset" in text
        assert "best_val_acc: 0.600000" in text
        assert "best_epoch: 2" in text
        assert "epoch | train_loss |   val_loss |    val_acc | val_precision |           lr" in text
        assert "Raw history (JSON)" in text


class TestSavePlots:
    def test_plot_files_created(self, tmp_path: Path) -> None:
        plots_dir = tmp_path / "plots"
        history = [
            {
                "epoch": 1,
                "train_loss": 1.0,
                "val_loss": 1.1,
                "val_acc": 0.5,
                "val_precision": 0.5,
                "lr": 0.001,
            },
            {
                "epoch": 2,
                "train_loss": 0.8,
                "val_loss": 0.9,
                "val_acc": 0.6,
                "val_precision": 0.6,
                "lr": 0.001,
            },
        ]
        train_script._save_plots(plots_dir=plots_dir, history=history)
        expected_files = [
            "loss.png",
            "accuracy.png",
            "precision.png",
            "combined_metrics.png",
        ]
        for filename in expected_files:
            assert (plots_dir / filename).exists()
