"""Model D wider actor and critic MLPs."""

from __future__ import annotations

import torch.nn as nn


def build_actor_mlp(input_dim: int) -> nn.Sequential:
    actor_mlp = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.Tanh(),
        nn.Linear(128, 128),
        nn.Tanh(),
    )

    return actor_mlp


def build_critic_mlp(input_dim: int) -> nn.Sequential:
    critic_mlp = nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.Tanh(),
        nn.Linear(256, 256),
        nn.Tanh(),
    )

    return critic_mlp
