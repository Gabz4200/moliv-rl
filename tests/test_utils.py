from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from moliv_rl.utils import get_logger, seed_worker, set_seeds


class TestGetLogger:
    """Behavioral tests for get_logger."""

    def test_logger_creation_console_only(self) -> None:
        logger_name = "test_logger_console_unique"
        logger = get_logger(logger_name, level=logging.DEBUG)

        assert logger.name == logger_name
        assert logger.level == logging.DEBUG
        assert not logger.propagate
        assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)

    def test_logger_file_handler_creation(self, tmp_path: Path) -> None:
        logger_name = "test_logger_file_unique"
        log_dir = tmp_path / "logs"
        logger = get_logger(logger_name, log_dir=log_dir)

        assert log_dir.is_dir()
        log_file = log_dir / f"{logger_name}.log"
        assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)

        logger.info("Test message for file output")
        # Flush handlers to ensure file write
        for h in logger.handlers:
            h.flush()

        assert log_file.is_file()
        content = log_file.read_text(encoding="utf-8")
        assert "Test message for file output" in content

    def test_logger_handler_reuse(self, tmp_path: Path) -> None:
        logger_name = "test_logger_idempotent"
        logger1 = get_logger(logger_name, log_dir=tmp_path)
        initial_handler_count = len(logger1.handlers)

        logger2 = get_logger(logger_name, log_dir=tmp_path)
        assert len(logger2.handlers) == initial_handler_count
        assert logger1 is logger2


class TestReproducibility:
    """Behavioral tests for set_seeds and seed_worker."""

    def test_set_seeds_determinism(self) -> None:
        set_seeds(42069)
        val_random_1 = random.random()
        val_numpy_1 = np.random.rand(5)
        val_torch_1 = torch.randn(5)

        set_seeds(42069)
        val_random_2 = random.random()
        val_numpy_2 = np.random.rand(5)
        val_torch_2 = torch.randn(5)

        assert val_random_1 == val_random_2
        assert np.array_equal(val_numpy_1, val_numpy_2)
        assert torch.equal(val_torch_1, val_torch_2)

    def test_negative_seed_raises(self) -> None:
        with pytest.raises(ValueError, match="seed must be non-negative"):
            set_seeds(-1)

    def test_seed_worker_execution(self) -> None:
        # seed_worker runs within a DataLoader worker context
        # It should derive the worker seed from torch.initial_seed() without error
        seed_worker(0)
        # Check that numpy and random are in a valid state
        val_np = np.random.rand()
        val_py = random.random()
        assert 0.0 <= val_np <= 1.0
        assert 0.0 <= val_py <= 1.0
