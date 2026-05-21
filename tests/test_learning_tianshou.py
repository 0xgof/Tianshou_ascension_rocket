import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from learning.adapters import DEFAULT_OBSERVATION_SIZE
from learning.tianshou import (MODEL_A, MODEL_B, MODEL_C, MODEL_D,
                               GaussianPolicyConfig, NetworkConfig,
                               build_named_model, make_encoded_env)
from learning.tianshou.models import resolve_activation
from learning.tianshou.ppo_runner import TianshouPPOConfig, run_tianshou_ppo_smoke
from learning.tianshou.training_settings import (build_tianshou_ppo_config,
                                                extract_settings_config,
                                                extract_training_config,
                                                load_training_file)
from rocket_env.env import EnvConfig


def test_encoded_env_exposes_flat_observations():
    env = make_encoded_env(EnvConfig(max_steps=2))
    observation, info = env.reset(seed=0)

    assert observation.shape == (DEFAULT_OBSERVATION_SIZE,)
    assert observation.dtype == np.float32
    assert env.observation_space.shape == (DEFAULT_OBSERVATION_SIZE,)


def test_unknown_activation_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown activation"):
        resolve_activation("definitely-not-an-activation")


def test_model_a_uses_editable_network_config():
    config = TianshouPPOConfig(
        model_name=MODEL_A,
        actor_network=NetworkConfig(hidden_sizes=(16, 8), activation="relu"),
        critic_network=NetworkConfig(hidden_sizes=(8,), activation="silu"),
        gaussian_policy=GaussianPolicyConfig(initial_log_std=-0.25),
    )
    actor, critic = build_named_model(config)

    assert isinstance(actor.backbone[1], nn.ReLU)
    assert isinstance(actor.backbone[3], nn.ReLU)
    assert isinstance(critic.backbone[1], nn.SiLU)
    torch.testing.assert_close(actor.policy.log_std.detach(),
                               torch.full((2,), -0.25))


def test_model_b_is_empty_shell_until_implemented():
    config = TianshouPPOConfig(model_name=MODEL_B)

    with pytest.raises(NotImplementedError, match="Model B is intentionally empty"):
        build_named_model(config)


def test_model_c_is_current_implemented_baseline():
    config = TianshouPPOConfig(model_name=MODEL_C,
                               actor_network=NetworkConfig(hidden_sizes=(8,),
                                                           activation="relu"),
                               gaussian_policy=GaussianPolicyConfig(initial_log_std=-1.0))
    actor, critic = build_named_model(config)

    assert actor.obs_dim == DEFAULT_OBSERVATION_SIZE
    assert actor.action_dim == 2
    assert actor.backbone[0].in_features == DEFAULT_OBSERVATION_SIZE
    assert actor.backbone[0].out_features == 64
    assert actor.backbone[2].in_features == 64
    assert actor.backbone[2].out_features == 64
    assert isinstance(actor.backbone[1], nn.Tanh)
    assert isinstance(actor.backbone[3], nn.Tanh)
    assert critic.backbone[0].out_features == 64
    torch.testing.assert_close(actor.policy.log_std.detach(), torch.zeros(2))


def test_model_d_is_registered_wider_model():
    config = TianshouPPOConfig(model_name=MODEL_D,
                               actor_network=NetworkConfig(hidden_sizes=(8,),
                                                           activation="relu"),
                               gaussian_policy=GaussianPolicyConfig(initial_log_std=-1.0))
    actor, critic = build_named_model(config)

    assert actor.obs_dim == DEFAULT_OBSERVATION_SIZE
    assert actor.action_dim == 2
    assert actor.backbone[0].in_features == DEFAULT_OBSERVATION_SIZE
    assert actor.backbone[0].out_features == 128
    assert actor.backbone[2].in_features == 128
    assert actor.backbone[2].out_features == 128
    assert isinstance(actor.backbone[1], nn.Tanh)
    assert isinstance(actor.backbone[3], nn.Tanh)
    assert critic.backbone[0].out_features == 256
    assert critic.backbone[2].out_features == 256
    torch.testing.assert_close(actor.policy.log_std.detach(),
                               torch.full((2,), -1.0))


def test_runner_rejects_partial_custom_model_pair():
    actor, _critic = build_named_model(TianshouPPOConfig(model_name=MODEL_C))

    with pytest.raises(ValueError, match="Pass both actor and critic"):
        run_tianshou_ppo_smoke(TianshouPPOConfig(max_steps=4), actor=actor)


def test_training_yaml_builds_reproducible_ppo_config():
    raw_config = load_training_file("experiments/model_c/smoke.yaml")
    training_config = extract_training_config(raw_config)
    config = build_tianshou_ppo_config(training_config)

    assert config.model_name == MODEL_C
    assert config.run_name == "model_c_smoke_from_yaml"
    assert config.max_epoch == 1
    assert config.step_per_epoch == 32
    assert config.actor_network.hidden_sizes == (64, 64)
    assert config.gaussian_policy.initial_log_std == 0.0
    assert config.training_display.enabled is False
    assert config.training_display.step_delay_seconds == 0.0


def test_training_yaml_accepts_resume_checkpoint_path():
    raw_config = {"model_name": MODEL_C,
                  "resume_checkpoint_path": "results/checkpoints/model_c/run/epoch_011.pt"}

    config = build_tianshou_ppo_config(raw_config)

    assert config.resume_checkpoint_path == "results/checkpoints/model_c/run/epoch_011.pt"


def test_training_yaml_extracts_settings_overlay():
    raw_config = {
        "settings": {
            "reward": {
                "selected_reward": "curriculum_low_altitude_v1",
            },
        },
        "training": {
            "run_name": "unit",
        },
    }

    settings_config = extract_settings_config(raw_config)

    assert settings_config == {
        "reward": {
            "selected_reward": "curriculum_low_altitude_v1",
        },
    }


@pytest.mark.skipif(importlib.util.find_spec("tianshou") is None,
                    reason="Tianshou is not installed in this environment.")
def test_tianshou_ppo_saves_metrics_and_checkpoint(tmp_path):
    settings_input = {"path": "unit.yaml",
                      "resolved": {
                          "environment": {"initial_fuel": 123.0},
                          "reward": {"selected_reward": "target_orbit_v1"},
                      }}
    training_input = {"resolved": {"max_epoch": 1, "step_per_epoch": 4}}
    config = TianshouPPOConfig(max_steps=4,
                               step_per_epoch=4,
                               step_per_collect=4,
                               batch_size=4,
                               model_name=MODEL_C,
                               output_dir=str(tmp_path),
                               run_name="unit_smoke",
                               settings_input=settings_input,
                               training_input=training_input)
    env_config = EnvConfig(max_steps=4, initial_fuel=123.0)
    metrics = run_tianshou_ppo_smoke(config, env_config=env_config)

    metrics_path = Path(metrics["metrics_path"])
    checkpoint_path = Path(metrics["checkpoint_path"])
    checkpoint_dir = Path(metrics["checkpoint_dir"])
    settings_path = Path(metrics["settings_path"])
    diagnostics_path = Path(metrics["training_diagnostics_path"])
    best_episode_trace_path = Path(metrics["best_episode_trace_path"])
    altitude_plot_path = Path(metrics["altitude_plot_path"])
    best_episode_trajectory_plot_path = Path(metrics["best_episode_trajectory_plot_path"])
    reward_plot_path = Path(metrics["reward_plot_path"])
    model_metrics_plot_path = Path(metrics["model_metrics_plot_path"])
    resolved_log_path = Path(metrics["log_path"])

    assert metrics_path.parent == tmp_path / "metrics" / MODEL_C
    assert checkpoint_dir.parent == tmp_path / "checkpoints" / MODEL_C
    assert checkpoint_dir.name == metrics_path.stem
    assert checkpoint_path.parent == checkpoint_dir
    assert settings_path.parent == tmp_path / "settings" / MODEL_C
    assert diagnostics_path.parent == tmp_path / "traces" / MODEL_C
    assert best_episode_trace_path.parent == diagnostics_path.parent
    assert altitude_plot_path.parent == tmp_path / "figures" / MODEL_C / metrics_path.stem
    assert best_episode_trajectory_plot_path.parent == altitude_plot_path.parent
    assert reward_plot_path.parent == altitude_plot_path.parent
    assert model_metrics_plot_path.parent == altitude_plot_path.parent
    assert resolved_log_path.parent == Path.cwd() / "logs" / "training"
    assert metrics_path.stem.startswith("unit_smoke_")
    assert checkpoint_path.name == "final.pt"
    assert settings_path.stem == metrics_path.stem
    assert diagnostics_path.stem == f"{metrics_path.stem}_training_diagnostics"
    assert best_episode_trace_path.stem == f"{metrics_path.stem}_best_episode_trace"
    assert altitude_plot_path.name == "episode_altitudes.png"
    assert best_episode_trajectory_plot_path.name == "best_episode_trajectory.png"
    assert reward_plot_path.name == "reward_functions.png"
    assert model_metrics_plot_path.name == "model_metrics.png"
    assert resolved_log_path.stem == f"model_c_{metrics_path.stem}"
    assert metrics_path.is_file()
    assert checkpoint_path.is_file()
    assert (checkpoint_dir / "epoch_001.pt").is_file()
    assert settings_path.is_file()
    assert diagnostics_path.is_file()
    assert best_episode_trace_path.is_file()
    assert altitude_plot_path.is_file()
    assert best_episode_trajectory_plot_path.is_file()
    assert reward_plot_path.is_file()
    assert model_metrics_plot_path.is_file()
    assert resolved_log_path.is_file()

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    settings_payload = json.loads(settings_path.read_text(encoding="utf-8"))
    diagnostics_payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    best_episode_trace_payload = json.loads(best_episode_trace_path.read_text(encoding="utf-8"))
    log_payload = json.loads(resolved_log_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    assert payload["config"]["model_name"] == MODEL_C
    expected_settings_payload = dict(settings_input)
    expected_settings_payload["training"] = training_input

    assert payload["settings_input"] == expected_settings_payload
    assert payload["config"]["settings_input"] == settings_input
    assert payload["config"]["training_input"] == training_input
    assert payload["settings_path"] == str(settings_path)
    assert payload["checkpoint_path"] == str(checkpoint_path)
    assert payload["checkpoint_dir"] == str(checkpoint_dir)
    assert payload["altitude_plot_path"] == str(altitude_plot_path)
    assert payload["best_episode_trace_path"] == str(best_episode_trace_path)
    assert payload["best_episode_trajectory_plot_path"] == str(best_episode_trajectory_plot_path)
    assert payload["reward_plot_path"] == str(reward_plot_path)
    assert payload["model_metrics_plot_path"] == str(model_metrics_plot_path)
    assert payload["training_diagnostics_path"] == str(diagnostics_path)
    assert payload["metrics"]["best_reward"] == metrics["best_reward"]
    assert diagnostics_payload["model_name"] == MODEL_C
    assert diagnostics_payload["run_name"] == "unit_smoke"
    assert diagnostics_payload["diagnostics"]["records"]
    assert any(record["step_type"] == "update/gradient_step"
               for record in diagnostics_payload["diagnostics"]["records"])
    assert diagnostics_payload["diagnostics"]["checkpoints"][0]["checkpoint_path"] == (
        str(checkpoint_dir / "epoch_001.pt")
    )
    assert best_episode_trace_payload["best_episode_record"]
    assert best_episode_trace_payload["trace"]
    assert settings_payload == expected_settings_payload
    assert checkpoint["model_name"] == MODEL_C
    assert checkpoint["checkpoint_kind"] == "final"
    assert checkpoint["settings_input"] == expected_settings_payload
    assert checkpoint["settings_path"] == str(settings_path)
    assert checkpoint["altitude_plot_path"] == str(altitude_plot_path)
    assert checkpoint["best_episode_trajectory_plot_path"] == str(best_episode_trajectory_plot_path)
    assert checkpoint["reward_plot_path"] == str(reward_plot_path)
    assert checkpoint["model_metrics_plot_path"] == str(model_metrics_plot_path)
    assert checkpoint["training_diagnostics_path"] == str(diagnostics_path)
    assert checkpoint["config"]["resume_checkpoint_path"] is None
    epoch_checkpoint = torch.load(checkpoint_dir / "epoch_001.pt",
                                  map_location="cpu")
    assert epoch_checkpoint["checkpoint_kind"] == "epoch"
    assert epoch_checkpoint["epoch"] == 1
    assert epoch_checkpoint["settings_input"] == expected_settings_payload
    assert log_payload["run_type"] == "training"
    assert log_payload["settings_input"] == expected_settings_payload
    assert log_payload["artifacts"]["settings_path"] == str(settings_path)
    assert log_payload["artifacts"]["checkpoint_path"] == str(checkpoint_path)
    assert log_payload["artifacts"]["checkpoint_dir"] == str(checkpoint_dir)
    assert log_payload["artifacts"]["altitude_plot_path"] == str(altitude_plot_path)
    assert log_payload["artifacts"]["best_episode_trace_path"] == str(best_episode_trace_path)
    assert log_payload["artifacts"]["best_episode_trajectory_plot_path"] == str(best_episode_trajectory_plot_path)
    assert log_payload["artifacts"]["reward_plot_path"] == str(reward_plot_path)
    assert log_payload["artifacts"]["model_metrics_plot_path"] == str(model_metrics_plot_path)
    assert log_payload["artifacts"]["training_diagnostics_path"] == str(diagnostics_path)
    assert "actor_state_dict" in checkpoint
    assert "critic_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint


@pytest.mark.skipif(importlib.util.find_spec("tianshou") is None,
                    reason="Tianshou is not installed in this environment.")
def test_tianshou_ppo_smoke_run_returns_metrics():
    config = TianshouPPOConfig(max_steps=4,
                               step_per_epoch=4,
                               step_per_collect=4,
                               batch_size=4,
                               model_name=MODEL_A,
                               actor_network=NetworkConfig(hidden_sizes=(16,),
                                                           activation="tanh"),
                               critic_network=NetworkConfig(hidden_sizes=(16,),
                                                            activation="tanh"),
                               gaussian_policy=GaussianPolicyConfig(initial_log_std=-0.5),
                               save_artifacts=False)
    metrics = run_tianshou_ppo_smoke(config)

    assert metrics
