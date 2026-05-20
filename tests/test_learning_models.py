import pytest
import torch
import torch.nn as nn

from learning.adapters import DEFAULT_OBSERVATION_SIZE
from learning.models import model_a, model_b, model_c
from learning.tianshou import GaussianPolicyConfig, NetworkConfig


def test_model_a_has_own_mlp_and_heads():
    actor, critic = model_a.build_model(
        obs_dim=DEFAULT_OBSERVATION_SIZE,
        action_dim=2,
        actor_network=NetworkConfig(hidden_sizes=(32, 16), activation="relu"),
        critic_network=NetworkConfig(hidden_sizes=(16,), activation="silu"),
        gaussian_policy=GaussianPolicyConfig(initial_log_std=-0.5),
    )

    logits, state = actor(torch.zeros((4, DEFAULT_OBSERVATION_SIZE)))
    mean, std = logits
    values = critic(torch.zeros((4, DEFAULT_OBSERVATION_SIZE)))

    assert state is None
    assert mean.shape == (4, 2)
    assert std.shape == (4, 2)
    assert values.shape == (4,)
    assert isinstance(actor.backbone[1], nn.ReLU)
    assert isinstance(actor.backbone[3], nn.ReLU)
    assert isinstance(critic.backbone[1], nn.SiLU)
    torch.testing.assert_close(actor.policy.log_std.detach(),
                               torch.full((2,), -0.5))


def test_model_b_is_empty_until_implemented():
    with pytest.raises(NotImplementedError, match="Model B is intentionally empty"):
        model_b.build_model(obs_dim=DEFAULT_OBSERVATION_SIZE,
                            action_dim=2,
                            actor_network=NetworkConfig(),
                            critic_network=NetworkConfig(),
                            gaussian_policy=GaussianPolicyConfig())


def test_model_c_is_fixed_13_64_64_2_baseline():
    actor, critic = model_c.build_model(
        obs_dim=DEFAULT_OBSERVATION_SIZE,
        action_dim=2,
        actor_network=NetworkConfig(hidden_sizes=(8,), activation="relu"),
        critic_network=NetworkConfig(hidden_sizes=(8,), activation="relu"),
        gaussian_policy=GaussianPolicyConfig(initial_log_std=-1.0),
    )

    logits, state = actor(torch.zeros((4, DEFAULT_OBSERVATION_SIZE)))
    mean, std = logits
    values = critic(torch.zeros((4, DEFAULT_OBSERVATION_SIZE)))

    assert state is None
    assert mean.shape == (4, 2)
    assert std.shape == (4, 2)
    assert values.shape == (4,)
    assert actor.backbone[0].in_features == DEFAULT_OBSERVATION_SIZE
    assert actor.backbone[0].out_features == 64
    assert actor.backbone[2].in_features == 64
    assert actor.backbone[2].out_features == 64
    assert actor.policy.mean.out_features == 2
    assert critic.value_head.value.out_features == 1
    torch.testing.assert_close(actor.policy.log_std.detach(), torch.zeros(2))


def test_model_c_backpropagates_through_local_parts():
    actor, critic = model_c.build_model(obs_dim=DEFAULT_OBSERVATION_SIZE,
                                        action_dim=2,
                                        actor_network=NetworkConfig(),
                                        critic_network=NetworkConfig(),
                                        gaussian_policy=GaussianPolicyConfig())
    obs = torch.zeros((4, DEFAULT_OBSERVATION_SIZE))
    mean = actor(obs)[0][0]
    values = critic(obs)
    loss = mean.mean() + values.mean()
    loss.backward()

    parameters = list(actor.parameters()) + list(critic.parameters())
    gradients = [param.grad for param in parameters if param.grad is not None]

    assert gradients
    assert any(torch.isfinite(gradient).all() for gradient in gradients)
