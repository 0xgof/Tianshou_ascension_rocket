"""Model A MLP scaffold."""

from __future__ import annotations

import torch.nn as nn


def build_mlp(input_dim: int,
              hidden_sizes: tuple[int, ...],
              activation: type[nn.Module]) -> nn.Sequential:
    """Build Model A's hidden layers.

    YOUR CODE AREA:
    - Add, remove, or reorder layers here.
    - Try normalization layers here.
    - Try different hidden-layer structure here.
    """

    layers = []
    current_dim = input_dim

    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(current_dim, hidden_size))
        layers.append(activation())
        current_dim = hidden_size

    mlp = nn.Sequential(*layers)

    return mlp
