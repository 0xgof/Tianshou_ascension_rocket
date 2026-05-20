"""Reusable rollout helpers for smoke tests and display scripts."""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np

from .env import RocketAscentEnv


TraceStep = Dict[str, object]
PolicyFn = Callable[[Dict[str, np.ndarray], Dict[str, object], int], np.ndarray]


def random_policy(env: RocketAscentEnv) -> PolicyFn:
    def _policy(_obs: Dict[str, np.ndarray], _info: Dict[str, object], _step: int):
        return env.np_random.uniform(env.action_space.low, env.action_space.high).astype(
            np.float32
        )

    return _policy


def radial_thrust_policy(
    _obs: Dict[str, np.ndarray],
    _info: Dict[str, object],
    _step: int,
) -> np.ndarray:
    return np.array([1.0, 0.0], dtype=np.float32)


def prograde_thrust_policy(
    _obs: Dict[str, np.ndarray],
    _info: Dict[str, object],
    _step: int,
) -> np.ndarray:
    return np.array([1.0, -1.0], dtype=np.float32)


def no_thrust_policy(
    _obs: Dict[str, np.ndarray],
    _info: Dict[str, object],
    _step: int,
) -> np.ndarray:
    return np.array([0.0, 0.0], dtype=np.float32)


def run_episode(
    env: RocketAscentEnv,
    policy: PolicyFn,
    max_steps: int = 500,
    seed: int = 0,
) -> Tuple[List[TraceStep], Dict[str, object]]:
    obs, info = env.reset(seed=seed)
    trace: List[TraceStep] = []
    final_info = info
    for step_index in range(max_steps):
        action = np.asarray(policy(obs, info, step_index), dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        trace.append(
            {
                "observation": obs,
                "action": action.copy(),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info": dict(info),
            }
        )
        final_info = info
        if terminated or truncated:
            break
    return trace, final_info
