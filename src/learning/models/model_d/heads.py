"""Model D fixed actor and critic heads."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


class GaussianActorHead(nn.Module):
    """Mean head plus learned log standard deviation for two actions."""

    def __init__(self,
                 input_dim: int,
                 action_dim: int,
                 initial_log_std: float = -0.5) -> None:
        super().__init__()

        self.mean = nn.Linear(input_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,),
                                               float(initial_log_std)))

    def distribution(self,
                     features: torch.Tensor) -> Normal:
        mean = self.mean(features)
        std = torch.exp(self.log_std).expand_as(mean)
        distribution = Normal(mean, std)

        return distribution


class ValueHead(nn.Module):
    """One scalar value estimate per observation."""

    def __init__(self,
                 input_dim: int) -> None:
        super().__init__()

        self.value = nn.Linear(input_dim, 1)

    def forward(self,
                features: torch.Tensor) -> torch.Tensor:
        values = self.value(features).squeeze(-1)

        return values
