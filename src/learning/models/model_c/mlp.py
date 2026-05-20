"""Model C fixed 64x64 MLP."""

from __future__ import annotations

import torch.nn as nn


def build_actor_mlp(input_dim: int) -> nn.Sequential:
    actor_mlp = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
    )

    return actor_mlp


def build_critic_mlp(input_dim: int) -> nn.Sequential:
    critic_mlp = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
    )

    return critic_mlp
