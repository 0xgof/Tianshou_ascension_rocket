import inspect

import numpy as np
import torch

from learning.adapters import DEFAULT_OBSERVATION_SIZE
from learning.checkpoints import load_checkpoint, save_checkpoint
from learning.evaluation import run_policy_episode, scripted_baseline_policy
from learning.models import model_c
from learning.tianshou import GaussianPolicyConfig, NetworkConfig
from rocket_env.env import EnvConfig, RocketAscentEnv


def _model_c_actor():
    actor, _critic = model_c.build_model(obs_dim=DEFAULT_OBSERVATION_SIZE,
                                         action_dim=2,
                                         actor_network=NetworkConfig(),
                                         critic_network=NetworkConfig(),
                                         gaussian_policy=GaussianPolicyConfig())

    return actor


def test_checkpoint_round_trip_preserves_deterministic_action(tmp_path):
    torch.manual_seed(0)
    model = _model_c_actor()
    observation = torch.zeros((1, DEFAULT_OBSERVATION_SIZE))
    expected = model(observation)[0][0]
    path = tmp_path / "model.pt"

    save_checkpoint(str(path),
                    model,
                    metadata={"obs_dim": DEFAULT_OBSERVATION_SIZE, "action_dim": 2})

    loaded = _model_c_actor()
    load_checkpoint(str(path), loaded)
    actual = loaded(observation)[0][0]

    torch.testing.assert_close(actual, expected)


def test_evaluation_runner_returns_trace_and_metrics():
    env = RocketAscentEnv(EnvConfig(max_steps=3))
    model = _model_c_actor()
    result = run_policy_episode(model, env, max_steps=3, seed=0)

    assert "trace" in result
    assert "metrics" in result
    assert result["metrics"]["steps"] >= 1.0


def test_evaluation_trace_is_replay_compatible():
    env = RocketAscentEnv(EnvConfig(max_steps=1))
    model = _model_c_actor()
    result = run_policy_episode(model, env, max_steps=1, seed=0)
    entry = result["trace"][0]

    for key in ["observation", "env_action", "action", "reward", "terminated", "truncated", "info"]:
        assert key in entry

    assert "model_action" in entry
    assert isinstance(entry["model_action"], np.ndarray)


def test_scripted_baseline_uses_same_evaluation_shape():
    env = RocketAscentEnv(EnvConfig(max_steps=1))
    result = run_policy_episode(scripted_baseline_policy, env, max_steps=1, seed=0)

    assert set(result) == {"trace", "metrics"}
    assert "env_action" in result["trace"][0]


def test_evaluation_modules_do_not_import_training_algorithms():
    import learning.evaluation.policy_runner as policy_runner
    import learning.checkpoints.io as checkpoint_io

    source = inspect.getsource(policy_runner) + inspect.getsource(checkpoint_io)

    assert "learning.algorithms" not in source
    assert "learning.data" not in source
    assert "learning.training" not in source
