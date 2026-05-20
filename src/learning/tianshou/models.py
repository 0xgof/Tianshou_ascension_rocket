"""Small shared types and utilities for Tianshou-compatible models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
import torch
import torch.nn as nn

TorchDevice = Union[torch.device, str, None]


@dataclass(frozen=True)
class NetworkConfig:
    """Visible NN architecture choices passed into Tianshou policies."""

    hidden_sizes: tuple[int, ...] = (64, 64)
    activation: str = "tanh"


@dataclass(frozen=True)
class GaussianPolicyConfig:
    """Visible Gaussian policy choices for continuous rocket actions."""

    initial_log_std: float = 0.0


def obs_to_tensor(obs: np.ndarray | torch.Tensor,
                  device: torch.device) -> torch.Tensor:
    obs_tensor = torch.as_tensor(obs,
                                 dtype=torch.float32,
                                 device=device)

    if obs_tensor.ndim == 1:
        obs_tensor = obs_tensor.unsqueeze(0)

    return obs_tensor


def resolve_activation(activation: type[nn.Module] | str) -> type[nn.Module]:
    if isinstance(activation, str):
        activation_map = {
            "elu": nn.ELU,
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
        }

        try:
            activation_type = activation_map[activation.lower()]
        except KeyError as exc:
            available = ", ".join(sorted(activation_map))
            raise ValueError(f"Unknown activation '{activation}'. Available: {available}") from exc

        return activation_type

    return activation
