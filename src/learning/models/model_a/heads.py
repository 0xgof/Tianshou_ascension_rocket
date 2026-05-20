"""Model A output-head scaffold."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


class GaussianActorHead(nn.Module):
    """Model A Gaussian policy head."""

    def __init__(self,
                 input_dim: int,
                 action_dim: int,
                 initial_log_std: float = 0.0) -> None:
        super().__init__()

        # YOUR CODE AREA:
        # - mean maps hidden features to two action means.
        # - log_std controls exploration noise.
        # - Replace this with state-dependent std when you are ready.
        self.mean = nn.Linear(input_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), initial_log_std))

    def distribution(self,
                     features: torch.Tensor) -> Normal:
        mean = self.mean(features)
        std = torch.exp(self.log_std).expand_as(mean)
        distribution = Normal(mean, std)

        return distribution


class ValueHead(nn.Module):
    """Model A critic output head."""

    def __init__(self,
                 input_dim: int) -> None:
        super().__init__()

        # YOUR CODE AREA:
        # Change this if you want a deeper critic head.
        self.value = nn.Linear(input_dim, 1)

    def forward(self,
                features: torch.Tensor) -> torch.Tensor:
        values = self.value(features).squeeze(-1)

        return values
