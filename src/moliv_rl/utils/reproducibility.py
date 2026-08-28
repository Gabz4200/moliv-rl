from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seeds(seed: int = 42069, deterministic: bool = True) -> None:
    r"""set_seeds(seed=42069, deterministic=True) -> None

    Set pseudo-random number generator seeds across Python, NumPy, and PyTorch for reproducible execution.

    Args:
        seed (int, optional): Global seed integer value. Default: ``42069``
        deterministic (bool, optional): If ``True``, configures CuDNN backends for strict determinism. Default: ``True``
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not deterministic


def seed_worker(_worker_id: int) -> None:
    r"""seed_worker(_worker_id) -> None

    Worker initialization callable for PyTorch DataLoader subprocesses to ensure distinct, reproducible seeding.

    Args:
        _worker_id (int): PyTorch DataLoader worker process integer index.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


__all__ = [
    "seed_worker",
    "set_seeds",
]
