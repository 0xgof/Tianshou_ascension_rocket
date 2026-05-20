"""Run restored trained policies through rocket episodes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from learning.evaluation import run_policy_episode
from learning.inference.checkpoints import load_tianshou_policy
from rocket_env.env import EnvConfig, RocketAscentEnv
from settings import ProjectSettings


@dataclass(frozen=True)
class InferenceResult:
    metrics: dict[str, float]
    trace_path: str
    metrics_path: str


def run_tianshou_checkpoint_episode(checkpoint_path: str | Path,
                                    settings_path: str | Path | None = None,
                                    output_dir: str | Path = "results/inference",
                                    run_name: str = "latest",
                                    deterministic: bool = True,
                                    max_steps: int | None = None,
                                    seed: int = 0) -> InferenceResult:
    loaded_policy = load_tianshou_policy(checkpoint_path)
    env_config = _env_config_for_inference(loaded_policy.checkpoint,
                                           settings_path=settings_path,
                                           max_steps=max_steps)
    episode_max_steps = max_steps or loaded_policy.config.max_steps
    env = RocketAscentEnv(env_config)
    episode = run_policy_episode(loaded_policy.actor,
                                 env,
                                 deterministic=deterministic,
                                 max_steps=episode_max_steps,
                                 seed=seed)

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    trace_path = target_dir / f"{run_name}_trace.json"
    metrics_path = target_dir / f"{run_name}_metrics.json"

    trace_payload = {"checkpoint_path": str(checkpoint_path),
                     "settings_path": str(settings_path) if settings_path else None,
                     "deterministic": deterministic,
                     "seed": seed,
                     "max_steps": episode_max_steps,
                     "trace": episode["trace"]}
    metrics_payload = {"checkpoint_path": str(checkpoint_path),
                       "settings_path": str(settings_path) if settings_path else None,
                       "deterministic": deterministic,
                       "seed": seed,
                       "max_steps": episode_max_steps,
                       "metrics": episode["metrics"]}

    trace_path.write_text(json.dumps(_json_safe(trace_payload), indent=2),
                          encoding="utf-8")
    metrics_path.write_text(json.dumps(_json_safe(metrics_payload), indent=2),
                            encoding="utf-8")

    inference_result = InferenceResult(metrics=episode["metrics"],
                                       trace_path=str(trace_path),
                                       metrics_path=str(metrics_path))

    return inference_result


def _env_config_for_inference(checkpoint: dict[str, Any],
                              settings_path: str | Path | None,
                              max_steps: int | None) -> EnvConfig:
    settings_payload = _load_settings_payload(checkpoint, settings_path)
    if settings_payload is None or "resolved" not in settings_payload:
        env_config = EnvConfig(max_steps=max_steps or checkpoint["config"]["max_steps"])

        return env_config

    project_settings = ProjectSettings.model_validate(settings_payload["resolved"])
    env_config = project_settings.to_env_config(
        max_steps=max_steps or checkpoint["config"]["max_steps"]
    )

    return env_config


def _load_settings_payload(checkpoint: dict[str, Any],
                           settings_path: str | Path | None) -> dict[str, Any] | None:
    if settings_path is not None:
        payload = json.loads(Path(settings_path).read_text(encoding="utf-8"))

        return payload

    payload = checkpoint.get("settings_input")

    return payload


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    return value
