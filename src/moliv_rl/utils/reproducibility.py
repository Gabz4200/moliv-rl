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
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True


def seed_worker(_worker_id: int) -> None:
    """Worker initialization function for PyTorch DataLoader to ensure worker-level determinism."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
