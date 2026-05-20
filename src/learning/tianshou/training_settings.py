"""Training-configuration loading helpers for Tianshou PPO runs."""

from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Mapping

from learning.tianshou.models import GaussianPolicyConfig, NetworkConfig
from learning.tianshou.ppo_runner import TianshouPPOConfig
from learning.tianshou.training_display import TrainingDisplayConfig
from settings.loaders import deep_merge, read_settings_file

TRAINING_SECTION = "training"
SETTINGS_SECTION = "settings"


def load_training_file(config_path: str | Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}

    training_file = read_settings_file(config_path)

    return training_file


def extract_training_config(config_data: Mapping[str, Any]) -> dict[str, Any]:
    """Extract trainer config from a raw YAML file or saved settings artifact."""

    if TRAINING_SECTION not in config_data:
        training_config = dict(config_data)

        return training_config

    training_section = config_data[TRAINING_SECTION]
    if not isinstance(training_section, Mapping):
        raise ValueError("training section must be a mapping.")

    resolved_config = training_section.get("resolved")
    if isinstance(resolved_config, Mapping):
        training_config = dict(resolved_config)
    else:
        training_config = dict(training_section)

    return training_config


def extract_settings_config(config_data: Mapping[str, Any]) -> dict[str, Any]:
    settings_section = config_data.get(SETTINGS_SECTION, {})
    if not isinstance(settings_section, Mapping):
        raise ValueError("settings section must be a mapping.")

    settings_config = dict(settings_section)

    return settings_config


def merge_training_config(file_config: Mapping[str, Any],
                          runtime_overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged_config = deep_merge(file_config, runtime_overrides)

    return merged_config


def build_tianshou_ppo_config(config_data: Mapping[str, Any],
                              settings_input: dict[str, Any] | None = None,
                              training_input: dict[str, Any] | None = None
                              ) -> TianshouPPOConfig:
    raw_config = dict(config_data)
    valid_fields = {field.name for field in fields(TianshouPPOConfig)}
    unknown_fields = set(raw_config) - valid_fields

    if unknown_fields:
        unknown = ", ".join(sorted(unknown_fields))
        raise ValueError(f"Unknown Tianshou PPO config field(s): {unknown}")

    if "actor_network" in raw_config:
        raw_config["actor_network"] = _network_config(raw_config["actor_network"])

    if "critic_network" in raw_config:
        raw_config["critic_network"] = _network_config(raw_config["critic_network"])

    if "gaussian_policy" in raw_config:
        raw_config["gaussian_policy"] = _gaussian_policy_config(
            raw_config["gaussian_policy"]
        )

    if "training_display" in raw_config:
        raw_config["training_display"] = _training_display_config(
            raw_config["training_display"]
        )

    raw_config["settings_input"] = settings_input
    raw_config["training_input"] = training_input
    config = TianshouPPOConfig(**raw_config)

    return config


def build_training_audit(config: TianshouPPOConfig,
                         config_path: str | Path | None,
                         raw_file: Mapping[str, Any],
                         file_config: Mapping[str, Any],
                         runtime_overrides: Mapping[str, Any]) -> dict[str, Any]:
    audit = {
        "training_config_file": _file_record(config_path),
        "raw_inputs": {
            "training_config_file": dict(raw_file),
            "training_config": dict(file_config),
            "runtime_overrides": dict(runtime_overrides),
        },
        "resolved": serializable_training_config(config),
    }

    return audit


def serializable_training_config(config: TianshouPPOConfig) -> dict[str, Any]:
    config_dict = asdict(config)
    config_dict.pop("settings_input", None)
    config_dict.pop("training_input", None)

    return config_dict


def _network_config(raw_config: Mapping[str, Any]) -> NetworkConfig:
    hidden_sizes = raw_config.get("hidden_sizes", NetworkConfig().hidden_sizes)
    network_config = NetworkConfig(hidden_sizes=tuple(hidden_sizes),
                                   activation=raw_config.get("activation", "tanh"))

    return network_config


def _gaussian_policy_config(raw_config: Mapping[str, Any]) -> GaussianPolicyConfig:
    gaussian_policy = GaussianPolicyConfig(
        initial_log_std=float(raw_config.get("initial_log_std", 0.0))
    )

    return gaussian_policy


def _training_display_config(raw_config: Mapping[str, Any]) -> TrainingDisplayConfig:
    training_display = TrainingDisplayConfig(**raw_config)

    return training_display


def _file_record(config_path: str | Path | None) -> dict[str, Any]:
    if config_path is None:
        record = {"path": None,
                  "status": "not_provided",
                  "raw_text": None}

        return record

    path = Path(config_path)
    status = "loaded" if path.is_file() else "missing"
    raw_text = path.read_text(encoding="utf-8") if path.is_file() else None
    record = {"path": str(path),
              "status": status,
              "raw_text": raw_text}

    return record
