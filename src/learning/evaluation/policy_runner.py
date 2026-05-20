"""Run deterministic policy episodes and export replay-compatible traces."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch.distributions import Normal

from learning.adapters import (decode_physical_action, encode_observation,
                               to_tensor)

Observation = dict[str, np.ndarray]
Info = dict[str, object]
TraceStep = dict[str, object]
Trace = list[TraceStep]
EpisodeMetrics = dict[str, float]
EpisodeResult = dict[str, object]
ScriptedPolicy = Callable[[np.ndarray], object]
PolicyActionResult = tuple[np.ndarray, np.ndarray]


def run_policy_episode(policy: torch.nn.Module | ScriptedPolicy,
                       env: Any,
                       deterministic: bool = True,
                       max_steps: int = 100,
                       seed: int = 0) -> EpisodeResult:
    obs: Observation
    _info: Info
    obs, _info = env.reset(seed=seed)

    trace: Trace = []
    total_reward = 0.0

    for _step in range(max_steps):
        encoded = encode_observation(obs)
        model_action, env_action = _policy_actions(policy, encoded, deterministic)
        next_obs, reward, terminated, truncated, info = env.step(env_action)

        step_trace: TraceStep = {"observation": obs,
                                 "model_action": model_action,
                                 "env_action": env_action,
                                 "action": env_action,
                                 "reward": float(reward),
                                 "terminated": bool(terminated),
                                 "truncated": bool(truncated),
                                 "info": dict(info)}

        trace.append(step_trace)
        total_reward += float(reward)
        obs = next_obs

        if terminated or truncated:
            break

    final_altitude = float(trace[-1]["info"]["altitude"]) if trace else 0.0

    metrics: EpisodeMetrics = {"total_reward": total_reward,
                               "steps": float(len(trace)),
                               "final_altitude": final_altitude}

    episode_result: EpisodeResult = {"trace": trace,
                                     "metrics": metrics}

    return episode_result


def _policy_actions(policy: torch.nn.Module | ScriptedPolicy,
                    encoded_observation: np.ndarray,
                    deterministic: bool) -> PolicyActionResult:
    if isinstance(policy, torch.nn.Module):
        model_action = _torch_policy_action(policy,
                                            encoded_observation,
                                            deterministic)
        action_pair = decode_physical_action(model_action)

        policy_action_result = (action_pair.model_action, action_pair.env_action)

        return policy_action_result

    env_action = np.asarray(policy(encoded_observation), dtype=np.float32)
    model_action = env_action.copy()
    policy_action_result = (model_action, env_action)

    return policy_action_result


def _torch_policy_action(policy: torch.nn.Module,
                         encoded_observation: np.ndarray,
                         deterministic: bool) -> np.ndarray:
    obs_tensor = to_tensor(encoded_observation).unsqueeze(0)

    with torch.no_grad():
        if hasattr(policy, "act"):
            decision = policy.act(obs_tensor, deterministic=deterministic)
            action_tensor = decision["action"].squeeze(0)
        else:
            output = policy(obs_tensor)
            action_tensor = _action_from_module_output(output, deterministic)

    model_action = action_tensor.detach().cpu().numpy().astype(np.float32)

    return model_action


def _action_from_module_output(output: object,
                               deterministic: bool) -> torch.Tensor:
    if _looks_like_tianshou_actor_output(output):
        logits, _state = output
        mean, std = logits
        distribution = Normal(mean, std)
        action_tensor = mean if deterministic else distribution.rsample()
        action_tensor = action_tensor.squeeze(0)

        return action_tensor

    if isinstance(output, torch.Tensor):
        action_tensor = output.squeeze(0)

        return action_tensor

    raise TypeError("Unsupported torch policy output shape.")


def _looks_like_tianshou_actor_output(output: object) -> bool:
    if not isinstance(output, tuple) or len(output) != 2:
        return False

    logits, _state = output
    if not isinstance(logits, tuple) or len(logits) != 2:
        return False

    mean, std = logits
    looks_like_output = isinstance(mean, torch.Tensor) and isinstance(std, torch.Tensor)

    return looks_like_output
