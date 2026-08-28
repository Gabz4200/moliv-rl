from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml


def load_config(config_path: Path | str | None) -> dict[str, Any]:
    r"""load_config(config_path) -> dict

    Load structured configuration dictionary from a YAML configuration file.

    Args:
        config_path (Path or str or None): File path to YAML config. If ``None``, returns empty dict.

    Returns:
        dict: Parsed YAML configuration dictionary.
    """
    if config_path is None:
        return {}

    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data or {}


def resolve_device(device: str | None) -> torch.device:
    r"""resolve_device(device) -> torch.device

    Resolve target compute device string or detect the optimal available accelerator.

    Args:
        device (str or None): Explicit device string (``'cpu'``, ``'cuda'``, ``'mps'``, ``'auto'``) or ``None``.

    Returns:
        torch.device: Resolved PyTorch device object.
    """
    if device is not None and device != "auto":
        resolved = torch.device(device)

        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

        if resolved.type == "mps":
            if not hasattr(torch.backends, "mps"):
                raise RuntimeError("MPS is not available in this PyTorch build")

            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS was requested but is not available")

        return resolved

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")
