"""Small Tianshou PPO runner for smoke tests and first experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal

from learning.adapters import DEFAULT_OBSERVATION_SIZE
from learning.models import model_a, model_b, model_c
from learning.tianshou.diagnostic_plots import plot_training_diagnostics
from learning.tianshou.envs import make_encoded_env
from learning.tianshou.episode_monitor import (EpisodeAltitudeRecord,
                                               EpisodeAltitudeRecorder,
                                               plot_episode_altitudes)
from learning.tianshou.models import GaussianPolicyConfig, NetworkConfig
from learning.tianshou.training_logger import TrainingDiagnosticsLogger
from learning.tianshou.training_display import TrainingDisplayConfig
from rocket_env.env import EnvConfig
from rocket_env.reward import TARGET_ORBIT_REWARD_V1
from rocket_env.reward_plots import plot_target_orbit_reward_functions
from settings import TRAINING_LOG_DIR, write_run_log

MODEL_A = "model_a"
MODEL_B = "model_b"
MODEL_C = "model_c"
MODEL_NAMES = (MODEL_A, MODEL_B, MODEL_C)


@dataclass(frozen=True)
class TrainingArtifacts:
    metrics_path: str
    checkpoint_path: str
    settings_path: str
    log_path: str
    episode_altitudes_path: str | None = None
    altitude_plot_path: str | None = None
    reward_plot_path: str | None = None
    model_metrics_plot_path: str | None = None
    training_diagnostics_path: str | None = None


@dataclass(frozen=True)
class TianshouPPOConfig:
    seed: int = 0
    train_envs: int = 1
    test_envs: int = 1
    max_steps: int = 32
    max_epoch: int = 1
    step_per_epoch: int = 32
    step_per_collect: int = 32
    episode_per_test: int = 1
    repeat_per_collect: int = 1
    batch_size: int = 16
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    eps_clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    obs_dim: int = DEFAULT_OBSERVATION_SIZE
    action_dim: int = 2
    model_name: str = MODEL_C
    actor_network: NetworkConfig = NetworkConfig()
    critic_network: NetworkConfig = NetworkConfig()
    gaussian_policy: GaussianPolicyConfig = GaussianPolicyConfig()
    training_display: TrainingDisplayConfig = TrainingDisplayConfig()
    save_artifacts: bool = True
    output_dir: str = "results"
    run_name: str = "latest"
    settings_input: dict[str, Any] | None = None
    training_input: dict[str, Any] | None = None


def build_tianshou_ppo_policy(actor: nn.Module,
                              critic: nn.Module,
                              optimizer: torch.optim.Optimizer,
                              action_space,
                              config: TianshouPPOConfig) -> Any:
    from tianshou.policy import PPOPolicy

    policy = PPOPolicy(actor=actor,
                       critic=critic,
                       optim=optimizer,
                       dist_fn=_independent_normal,
                       action_space=action_space,
                       eps_clip=config.eps_clip,
                       vf_coef=config.value_coef,
                       ent_coef=config.entropy_coef,
                       gae_lambda=config.gae_lambda,
                       discount_factor=config.gamma,
                       deterministic_eval=True,
                       action_scaling=False,
                       action_bound_method="clip")

    return policy


def run_tianshou_ppo_smoke(config: TianshouPPOConfig = TianshouPPOConfig(),
                           actor: nn.Module | None = None,
                           critic: nn.Module | None = None,
                           env_config: EnvConfig | None = None) -> dict:
    """Run a tiny PPO training loop through Tianshou."""

    from tianshou.data import Collector, VectorReplayBuffer
    from tianshou.env import DummyVectorEnv
    from tianshou.trainer import OnpolicyTrainer
    from tianshou.utils.net.common import ActorCritic

    torch.manual_seed(config.seed)

    env_config = replace(env_config or EnvConfig(), max_steps=config.max_steps)
    sample_env = make_encoded_env(env_config)
    altitude_recorder = EpisodeAltitudeRecorder()
    diagnostics_logger = TrainingDiagnosticsLogger()

    train_envs = DummyVectorEnv([
        _make_train_env_factory(env_config,
                                config.training_display,
                                display_enabled=index == 0,
                                altitude_recorder=altitude_recorder,
                                env_index=index)
        for index in range(config.train_envs)
    ])
    test_envs = DummyVectorEnv([
        lambda: make_encoded_env(env_config)
        for _ in range(config.test_envs)
    ])

    if (actor is None) != (critic is None):
        raise ValueError("Pass both actor and critic, or pass neither and use model_name.")

    if actor is None and critic is None:
        actor, critic = build_named_model(config)

    actor_critic = ActorCritic(actor=actor, critic=critic)
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=config.learning_rate)
    policy = build_tianshou_ppo_policy(actor,
                                       critic,
                                       optimizer,
                                       sample_env.action_space,
                                       config)

    train_collector = Collector(policy=policy,
                                env=train_envs,
                                buffer=VectorReplayBuffer(config.step_per_collect * 2,
                                                          config.train_envs))
    test_collector = Collector(policy=policy, env=test_envs)

    def track_epoch(epoch: int,
                    _env_step: int) -> None:
        altitude_recorder.current_epoch = epoch

    trainer = OnpolicyTrainer(policy=policy,
                              train_collector=train_collector,
                              test_collector=test_collector,
                              max_epoch=config.max_epoch,
                              step_per_epoch=config.step_per_epoch,
                              repeat_per_collect=config.repeat_per_collect,
                              episode_per_test=config.episode_per_test,
                              batch_size=config.batch_size,
                              step_per_collect=config.step_per_collect,
                              train_fn=track_epoch,
                              logger=diagnostics_logger)

    result = trainer.run()
    metrics = dict(result)
    metrics["episode_altitudes"] = altitude_recorder.records
    metrics["training_diagnostics"] = diagnostics_logger.payload()

    if config.save_artifacts:
        artifacts = save_training_artifacts(config=config,
                                            metrics=metrics,
                                            actor=actor,
                                            critic=critic,
                                            optimizer=optimizer,
                                            episode_altitudes=altitude_recorder.records,
                                            training_diagnostics=diagnostics_logger.payload())
        metrics["metrics_path"] = artifacts.metrics_path
        metrics["checkpoint_path"] = artifacts.checkpoint_path
        metrics["settings_path"] = artifacts.settings_path
        metrics["log_path"] = artifacts.log_path
        metrics["episode_altitudes_path"] = artifacts.episode_altitudes_path
        metrics["altitude_plot_path"] = artifacts.altitude_plot_path
        metrics["reward_plot_path"] = artifacts.reward_plot_path
        metrics["model_metrics_plot_path"] = artifacts.model_metrics_plot_path
        metrics["training_diagnostics_path"] = artifacts.training_diagnostics_path

    return metrics


def build_named_model(config: TianshouPPOConfig) -> tuple[nn.Module, nn.Module]:
    """Build one of the explicit model files."""

    if config.model_name == MODEL_A:
        builder = model_a.build_model
    elif config.model_name == MODEL_B:
        builder = model_b.build_model
    elif config.model_name == MODEL_C:
        builder = model_c.build_model
    else:
        available = ", ".join(MODEL_NAMES)
        raise ValueError(f"Unknown model '{config.model_name}'. Available: {available}")

    model_pair = builder(obs_dim=config.obs_dim,
                         action_dim=config.action_dim,
                         actor_network=config.actor_network,
                         critic_network=config.critic_network,
                         gaussian_policy=config.gaussian_policy)

    return model_pair


def _make_train_env_factory(env_config: EnvConfig,
                            training_display: TrainingDisplayConfig,
                            display_enabled: bool,
                            altitude_recorder: EpisodeAltitudeRecorder,
                            env_index: int):
    def make_train_env():
        display_config = training_display if display_enabled else None
        env = make_encoded_env(env_config,
                               training_display=display_config,
                               altitude_recorder=altitude_recorder,
                               env_index=env_index)

        return env

    return make_train_env


def save_training_artifacts(config: TianshouPPOConfig,
                            metrics: dict,
                            actor: nn.Module,
                            critic: nn.Module,
                            optimizer: torch.optim.Optimizer,
                            episode_altitudes: list[EpisodeAltitudeRecord] | None = None,
                            training_diagnostics: dict[str, Any] | None = None
                            ) -> TrainingArtifacts:
    """Save metrics and model state for one training run."""

    run_dir = Path(config.output_dir)
    metrics_dir = run_dir / "metrics" / config.model_name
    checkpoint_dir = run_dir / "checkpoints" / config.model_name
    settings_dir = run_dir / "settings" / config.model_name
    traces_dir = run_dir / "traces" / config.model_name
    figures_root_dir = run_dir / "figures" / config.model_name

    metrics_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    settings_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)

    artifact_run_name = _timestamped_artifact_name(config.run_name)
    figures_dir = figures_root_dir / artifact_run_name
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = metrics_dir / f"{artifact_run_name}.json"
    checkpoint_path = checkpoint_dir / f"{artifact_run_name}.pt"
    settings_path = settings_dir / f"{artifact_run_name}.json"
    episode_altitudes_path = traces_dir / f"{artifact_run_name}_episode_altitudes.json"
    training_diagnostics_path = traces_dir / f"{artifact_run_name}_training_diagnostics.json"
    altitude_plot_path = figures_dir / "episode_altitudes.png"
    reward_plot_path = figures_dir / "reward_functions.png"
    model_metrics_plot_path = figures_dir / "model_metrics.png"

    episode_altitudes = episode_altitudes or []
    episode_altitudes_payload = {"model_name": config.model_name,
                                 "run_name": config.run_name,
                                 "artifact_run_name": artifact_run_name,
                                 "records": _json_safe(episode_altitudes)}
    episode_altitudes_path.write_text(json.dumps(episode_altitudes_payload, indent=2),
                                      encoding="utf-8")
    plot_episode_altitudes(episode_altitudes,
                           altitude_plot_path,
                           title=f"{config.run_name}: max altitude by episode")
    training_diagnostics_payload = {"model_name": config.model_name,
                                    "run_name": config.run_name,
                                    "artifact_run_name": artifact_run_name,
                                    "diagnostics": _json_safe(training_diagnostics or {})}
    training_diagnostics_path.write_text(json.dumps(training_diagnostics_payload, indent=2),
                                         encoding="utf-8")
    plot_training_diagnostics(training_diagnostics or {},
                              model_metrics_plot_path,
                              title=f"{config.run_name}: model metrics")

    settings_payload = _settings_payload(config)
    settings_path.write_text(json.dumps(settings_payload, indent=2),
                             encoding="utf-8")
    if _selected_reward_name(settings_payload) == TARGET_ORBIT_REWARD_V1:
        plot_target_orbit_reward_functions(reward_plot_path)
    else:
        reward_plot_path = None

    metrics_payload = {"config": _serializable_config(config),
                       "artifact_run_name": artifact_run_name,
                       "settings_input": settings_payload,
                       "settings_path": settings_path,
                       "altitude_plot_path": altitude_plot_path,
                       "reward_plot_path": reward_plot_path,
                       "model_metrics_plot_path": model_metrics_plot_path,
                       "training_diagnostics_path": training_diagnostics_path,
                       "metrics": _json_safe(metrics)}
    metrics_path.write_text(json.dumps(_json_safe(metrics_payload), indent=2),
                            encoding="utf-8")

    checkpoint_payload = {
        "config": asdict(config),
        "settings_input": settings_payload,
        "settings_path": str(settings_path),
        "model_name": config.model_name,
        "artifact_run_name": artifact_run_name,
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": _json_safe(metrics),
        "altitude_plot_path": str(altitude_plot_path),
        "reward_plot_path": str(reward_plot_path) if reward_plot_path else None,
        "model_metrics_plot_path": str(model_metrics_plot_path),
        "training_diagnostics_path": str(training_diagnostics_path),
    }
    torch.save(checkpoint_payload, checkpoint_path)

    log_payload = {"run_type": "training",
                   "model_name": config.model_name,
                   "run_name": config.run_name,
                   "artifact_run_name": artifact_run_name,
                   "config": _serializable_config(config),
                   "settings_input": settings_payload,
                   "metrics": _json_safe(metrics),
                   "artifacts": {
                       "metrics_path": metrics_path,
                       "checkpoint_path": checkpoint_path,
                       "settings_path": settings_path,
                       "episode_altitudes_path": episode_altitudes_path,
                       "altitude_plot_path": altitude_plot_path,
                       "reward_plot_path": reward_plot_path,
                       "model_metrics_plot_path": model_metrics_plot_path,
                       "training_diagnostics_path": training_diagnostics_path,
                   }}
    log_path = write_run_log(TRAINING_LOG_DIR,
                             f"{config.model_name}_{artifact_run_name}",
                             log_payload)

    artifacts = TrainingArtifacts(metrics_path=str(metrics_path),
                                  checkpoint_path=str(checkpoint_path),
                                  settings_path=str(settings_path),
                                  log_path=str(log_path),
                                  episode_altitudes_path=str(episode_altitudes_path),
                                  altitude_plot_path=str(altitude_plot_path),
                                  reward_plot_path=str(reward_plot_path) if reward_plot_path else None,
                                  model_metrics_plot_path=str(model_metrics_plot_path),
                                  training_diagnostics_path=str(training_diagnostics_path))

    return artifacts


def _serializable_config(config: TianshouPPOConfig) -> dict:
    config_dict = asdict(config)
    safe_config = _json_safe(config_dict)

    return safe_config


def _settings_payload(config: TianshouPPOConfig) -> dict[str, Any]:
    raw_payload = config.settings_input or {}
    settings_payload = _json_safe(raw_payload)
    if not isinstance(settings_payload, dict):
        settings_payload = {"environment_settings": settings_payload}

    if config.training_input is not None:
        settings_payload["training"] = _json_safe(config.training_input)

    return settings_payload


def _selected_reward_name(settings_payload: dict[str, Any]) -> str | None:
    resolved_settings = settings_payload.get("resolved", {})
    if not isinstance(resolved_settings, dict):
        return None

    reward_settings = resolved_settings.get("reward", {})
    if not isinstance(reward_settings, dict):
        return None

    reward_name = reward_settings.get("selected_reward")
    if not isinstance(reward_name, str):
        return None

    return reward_name


def _timestamped_artifact_name(run_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_name = f"{run_name}_{timestamp}"

    return artifact_name


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()

    if isinstance(value, Path):
        return str(value)

    return value


def _independent_normal(mean: torch.Tensor,
                        std: torch.Tensor) -> Independent:
    distribution = Independent(Normal(mean, std), 1)

    return distribution
