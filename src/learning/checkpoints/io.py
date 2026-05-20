"""Checkpoint save/load helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    metadata: Optional[dict] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint

