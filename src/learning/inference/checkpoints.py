"""Load trained Tianshou checkpoints back into explicit model modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import torch
import torch.nn as nn

from learning.tianshou import (GaussianPolicyConfig, NetworkConfig,
                               TianshouPPOConfig, build_named_model)

TorchDevice = Union[torch.device, str, None]


@dataclass(frozen=True)
class LoadedTianshouPolicy:
    actor: nn.Module
    critic: nn.Module
    config: TianshouPPOConfig
    checkpoint: dict[str, Any]


def load_tianshou_policy(checkpoint_path: str | Path,
                         device: TorchDevice = None) -> LoadedTianshouPolicy:
    """Restore actor and critic modules from a Tianshou training checkpoint."""

    target_device = torch.device(device or "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=target_device)
    config = _config_from_checkpoint(checkpoint)
    actor, critic = build_named_model(config)

    actor.load_state_dict(checkpoint["actor_state_dict"])
    critic.load_state_dict(checkpoint["critic_state_dict"])
    actor.to(target_device)
    critic.to(target_device)
    actor.eval()
    critic.eval()

    loaded_policy = LoadedTianshouPolicy(actor=actor,
                                         critic=critic,
                                         config=config,
                                         checkpoint=checkpoint)

    return loaded_policy


def _config_from_checkpoint(checkpoint: dict[str, Any]) -> TianshouPPOConfig:
    raw_config = dict(checkpoint["config"])
    raw_config["actor_network"] = _network_config(raw_config["actor_network"])
    raw_config["critic_network"] = _network_config(raw_config["critic_network"])
    raw_config["gaussian_policy"] = GaussianPolicyConfig(
        **raw_config["gaussian_policy"]
    )
    config = TianshouPPOConfig(**raw_config)

    return config


def _network_config(raw_config: dict[str, Any]) -> NetworkConfig:
    hidden_sizes = tuple(raw_config["hidden_sizes"])
    network_config = NetworkConfig(hidden_sizes=hidden_sizes,
                                   activation=raw_config["activation"])

    return network_config
