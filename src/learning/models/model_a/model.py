"""Model A: editable actor and critic scaffold."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from learning.models.model_a.heads import GaussianActorHead, ValueHead
from learning.models.model_a.mlp import build_mlp
from learning.tianshou.models import (GaussianPolicyConfig, NetworkConfig,
                                      TorchDevice, obs_to_tensor,
                                      resolve_activation)


class ModelAActor(nn.Module):
    """Editable Gaussian actor scaffold."""

    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 network_config: NetworkConfig,
                 gaussian_config: GaussianPolicyConfig,
                 device: TorchDevice = None) -> None:
        super().__init__()

        self.device = torch.device(device or "cpu")
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)

        activation = resolve_activation(network_config.activation)
        hidden_sizes = tuple(network_config.hidden_sizes)

        # YOUR CODE AREA:
        # This is the actor feature extractor. It sees the NN input vector.
        self.backbone = build_mlp(self.obs_dim, hidden_sizes, activation)
        feature_dim = hidden_sizes[-1] if hidden_sizes else self.obs_dim

        # YOUR CODE AREA:
        # This head turns features into a Gaussian action distribution.
        self.policy = GaussianActorHead(feature_dim,
                                        self.action_dim,
                                        initial_log_std=gaussian_config.initial_log_std)

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


class ModelACritic(nn.Module):
    """Editable value critic scaffold."""

    def __init__(self,
                 obs_dim: int,
                 network_config: NetworkConfig,
                 device: TorchDevice = None) -> None:
        super().__init__()

        self.device = torch.device(device or "cpu")
        self.obs_dim = int(obs_dim)

        activation = resolve_activation(network_config.activation)
        hidden_sizes = tuple(network_config.hidden_sizes)

        # YOUR CODE AREA:
        # This is the critic feature extractor. It can be different from actor.
        self.backbone = build_mlp(self.obs_dim, hidden_sizes, activation)
        feature_dim = hidden_sizes[-1] if hidden_sizes else self.obs_dim

        # YOUR CODE AREA:
        # This head maps critic features to one state-value estimate.
        self.value_head = ValueHead(feature_dim)

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
                device: TorchDevice = None) -> tuple[ModelAActor, ModelACritic]:
    actor = ModelAActor(obs_dim=obs_dim,
                        action_dim=action_dim,
                        network_config=actor_network,
                        gaussian_config=gaussian_policy,
                        device=device)
    critic = ModelACritic(obs_dim=obs_dim,
                          network_config=critic_network,
                          device=device)
    model_pair = actor, critic

    return model_pair
