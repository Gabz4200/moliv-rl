from __future__ import annotations

from pathlib import Path

import torch
import yaml

from moliv_rl.models import get_model


class TestConfigs:
    """Behavioral tests for configs/default.yaml and configuration structure."""

    def test_default_yaml_exists_and_parses(self) -> None:
        config_path = Path("configs/default.yaml")
        assert config_path.is_file()

        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        assert isinstance(cfg, dict)

        # Verify required root sections
        expected_sections = {
            "project",
            "paths",
            "data",
            "model",
            "optimizer",
            "scheduler",
            "loss",
            "training",
            "runtime",
        }
        assert expected_sections.issubset(cfg.keys())

    def test_default_yaml_values_sanity(self) -> None:
        with Path("configs/default.yaml").open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        assert cfg["project"]["seed"] == 42069
        assert cfg["data"]["image_size"] > 0
        assert cfg["data"]["batch_size"] > 0
        assert cfg["data"]["resize_scale"] >= 1.0

        model_cfg = cfg["model"]
        assert model_cfg["name"] in {"classification_model", "my_model"}
        assert len(model_cfg["block_dims"]) >= 2
        assert all(dim > 0 for dim in model_cfg["block_dims"])
        assert model_cfg["in_channels"] > 0
        assert model_cfg["out_channels"] > 0
        assert model_cfg["patch_size"] > 0
        assert 0.0 <= model_cfg["dropout"] < 1.0
        assert model_cfg["num_classes"] > 0

        assert cfg["optimizer"]["name"] in {"adamw", "adam", "sgd"}
        assert cfg["optimizer"]["learning_rate"] > 0.0
        assert cfg["optimizer"]["weight_decay"] >= 0.0

        assert cfg["scheduler"]["name"] in {"cosine_annealing", "cosine", "none"}
        assert cfg["scheduler"]["interval"] in {"epoch", "step"}

        assert cfg["loss"]["name"] in {"cross_entropy", "focal"}

    def test_instantiate_model_from_default_yaml(self) -> None:
        with Path("configs/default.yaml").open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        model_cfg = cfg["model"]
        model = get_model(
            model_name=model_cfg["name"],
            optimize=False,
            block_dims=model_cfg["block_dims"],
            in_channels=model_cfg["in_channels"],
            out_channels=model_cfg["out_channels"],
            patch_size=model_cfg["patch_size"],
            dropout=model_cfg["dropout"],
            num_classes=model_cfg["num_classes"],
        )

        x = torch.randn(2, model_cfg["in_channels"], 64, 64)
        out = model(x)
        assert out.shape == (2, model_cfg["num_classes"])
