"""Model C: implemented 13 -> 64 -> 64 baseline."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from learning.models.model_c.heads import GaussianActorHead, ValueHead
from learning.models.model_c.mlp import build_actor_mlp, build_critic_mlp
from learning.tianshou.models import (GaussianPolicyConfig, NetworkConfig,
                                      TorchDevice, obs_to_tensor)


class ModelCActor(nn.Module):
    """Baseline Gaussian actor.

    Shape:

    ```text
    observation: 13
    hidden:      64
    hidden:      64
    mean:        2
    log_std:     2 learned parameters
    ```
    """

    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 device: TorchDevice = None) -> None:
        super().__init__()

        self.device = torch.device(device or "cpu")
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)

        self.backbone = build_actor_mlp(self.obs_dim)
        self.policy = GaussianActorHead(input_dim=64,
                                        action_dim=self.action_dim)

        self.to(self.device)

    def forward(self,
                obs: np.ndarray | torch.Tensor,
                state: object | None = None,
                info: dict | None = None) -> tuple[tuple[torch.Tensor, torch.Tensor],
                                                   object | None]:
        obs_tensor = obs_to_tensor(obs, self.device)
        features = self.backbone(obs_tensor)
        distribution = self.policy.distribution(features)

        logits = distribution.mean, distribution.stddev
        actor_output = logits, state

        return actor_output


class ModelCCritic(nn.Module):
    """Baseline value critic.

    Shape:

    ```text
    observation: 13
    hidden:      64
    hidden:      64
    value:       1
    ```
    """

    def __init__(self,
                 obs_dim: int,
                 device: TorchDevice = None) -> None:
        super().__init__()

        self.device = torch.device(device or "cpu")
        self.obs_dim = int(obs_dim)

        self.backbone = build_critic_mlp(self.obs_dim)
        self.value_head = ValueHead(input_dim=64)

        self.to(self.device)

    def forward(self,
                obs: np.ndarray | torch.Tensor,
                act: np.ndarray | torch.Tensor | None = None,
                info: dict | None = None) -> torch.Tensor:
        obs_tensor = obs_to_tensor(obs, self.device)
        features = self.backbone(obs_tensor)
        values = self.value_head(features)

        return values


def build_model(obs_dim: int,
                action_dim: int,
                actor_network: NetworkConfig,
                critic_network: NetworkConfig,
                gaussian_policy: GaussianPolicyConfig,
                device: TorchDevice = None) -> tuple[ModelCActor, ModelCCritic]:
    """Build the fixed Model C baseline.

    The config arguments are accepted so the runner can call every model with
    the same signature. Model C ignores them to remain the fixed baseline.
    """

    actor = ModelCActor(obs_dim=obs_dim,
                        action_dim=action_dim,
                        device=device)
    critic = ModelCCritic(obs_dim=obs_dim,
                          device=device)
    model_pair = actor, critic

    return model_pair
