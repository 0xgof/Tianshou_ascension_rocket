"""Run a tiny learning evaluation smoke episode."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from learning.adapters import DEFAULT_OBSERVATION_SIZE
from learning.evaluation import run_policy_episode
from learning.models import model_c
from learning.tianshou import GaussianPolicyConfig, NetworkConfig
from rocket_env.env import EnvConfig, RocketAscentEnv


def main() -> None:
    env = RocketAscentEnv(EnvConfig(max_steps=3))
    model, _critic = model_c.build_model(obs_dim=DEFAULT_OBSERVATION_SIZE,
                                         action_dim=2,
                                         actor_network=NetworkConfig(),
                                         critic_network=NetworkConfig(),
                                         gaussian_policy=GaussianPolicyConfig())
    result = run_policy_episode(model, env, max_steps=3)
    print(result["metrics"])


if __name__ == "__main__":
    main()
