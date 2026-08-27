from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seeds(seed: int = 42069, deterministic: bool = True) -> None:
    """Set random seeds across standard library, NumPy, and PyTorch for reproducible runs."""
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not deterministic


def seed_worker(_worker_id: int) -> None:
    """Worker initialization function for PyTorch DataLoader to ensure worker-level determinism."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
