"""Model D: wider actor and critic baseline."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from learning.models.model_d.heads import GaussianActorHead, ValueHead
from learning.models.model_d.mlp import build_actor_mlp, build_critic_mlp
from learning.tianshou.models import (GaussianPolicyConfig, NetworkConfig,
                                      TorchDevice, obs_to_tensor)


class ModelDActor(nn.Module):
    """Wider Gaussian actor.

    Shape:

    ```text
    observation: 13
    hidden:      128
    hidden:      128
    mean:        2
    log_std:     2 learned parameters
    ```
    """

    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 initial_log_std: float = -0.5,
                 device: TorchDevice = None) -> None:
        super().__init__()

        self.device = torch.device(device or "cpu")
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)

        self.backbone = build_actor_mlp(self.obs_dim)
        self.policy = GaussianActorHead(input_dim=128,
                                        action_dim=self.action_dim,
                                        initial_log_std=initial_log_std)

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


class ModelDCritic(nn.Module):
    """Wider value critic.

    Shape:

    ```text
    observation: 13
    hidden:      256
    hidden:      256
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
        self.value_head = ValueHead(input_dim=256)

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
                device: TorchDevice = None) -> tuple[ModelDActor, ModelDCritic]:
    """Build the fixed Model D baseline.

    The config arguments are accepted so the runner can call every model with
    the same signature. Model D owns its layer sizes but accepts the Gaussian
    initial log standard deviation so experiments can tune exploration.
    """

    actor = ModelDActor(obs_dim=obs_dim,
                        action_dim=action_dim,
                        initial_log_std=gaussian_policy.initial_log_std,
                        device=device)
    critic = ModelDCritic(obs_dim=obs_dim,
                          device=device)
    model_pair = actor, critic

    return model_pair
