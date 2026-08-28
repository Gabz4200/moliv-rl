from __future__ import annotations

from pathlib import Path

import torch
import yaml

from moliv_rl.models import get_model


class TestConfigs:
    """Behavioral tests for configs/default.yaml and configuration structure."""

    def test_default_yaml_structure_and_types(self) -> None:
        config_path = Path("configs/default.yaml")
        assert config_path.is_file()

        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        assert isinstance(cfg, dict)
        assert cfg["project"]["seed"] == 42069
        assert cfg["data"]["batch_size"] > 0
        assert cfg["optimizer"]["learning_rate"] > 0.0

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
