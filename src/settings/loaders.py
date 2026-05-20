"""Settings loading helpers for defaults, files, and explicit overrides."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import yaml

from .schema import ProjectSettings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_APP_SETTINGS_PATH = PROJECT_ROOT / "configs" / "app.yaml"


@dataclass(frozen=True)
class LoadedSettings:
    settings: ProjectSettings
    audit: dict[str, Any]


def deep_merge(
    base: Mapping[str, Any],
    updates: Mapping[str, Any],
) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def read_settings_file(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Settings file not found: {path}")

    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text) or {}
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported settings file type: {suffix}")

    if not isinstance(data, Mapping):
        raise ValueError("Settings file must contain a mapping at the top level.")
    return dict(data)


def resolve_default_config_path() -> Optional[Path]:
    if DEFAULT_APP_SETTINGS_PATH.is_file():
        return DEFAULT_APP_SETTINGS_PATH
    return None


def load_settings(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> ProjectSettings:
    loaded = load_settings_with_audit(config_path=config_path,
                                      overrides=overrides)

    return loaded.settings


def load_settings_with_audit(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> LoadedSettings:
    data: dict[str, Any] = {}
    default_data: dict[str, Any] = {}
    explicit_data: dict[str, Any] = {}
    override_data = deepcopy(dict(overrides or {}))

    default_path = resolve_default_config_path()

    if default_path is not None:
        default_data = read_settings_file(default_path)
        data = default_data

    if config_path is not None:
        explicit_data = read_settings_file(config_path)
        data = deep_merge(data, explicit_data)

    if override_data:
        data = deep_merge(data, override_data)

    settings = ProjectSettings.model_validate(data)
    audit = build_settings_audit(settings=settings,
                                 default_path=default_path,
                                 explicit_path=Path(config_path) if config_path else None,
                                 default_data=default_data,
                                 explicit_data=explicit_data,
                                 override_data=override_data)

    loaded_settings = LoadedSettings(settings=settings, audit=audit)

    return loaded_settings


def build_settings_audit(settings: ProjectSettings,
                         default_path: Path | None,
                         explicit_path: Path | None,
                         default_data: Mapping[str, Any],
                         explicit_data: Mapping[str, Any],
                         override_data: Mapping[str, Any]) -> dict[str, Any]:
    resolved_data = settings.model_dump()

    default_paths = _leaf_paths(default_data)
    explicit_paths = _leaf_paths(explicit_data)
    override_paths = _leaf_paths(override_data)
    configured_paths = default_paths | explicit_paths | override_paths
    resolved_paths = _leaf_paths(resolved_data)

    source_fields = {
        "app_file": sorted(default_paths - explicit_paths - override_paths),
        "explicit_config": sorted(explicit_paths - override_paths),
        "runtime_overrides": sorted(override_paths),
        "pydantic_fallback": sorted(resolved_paths - configured_paths),
    }

    audit = {
        "app_settings_file": _settings_file_record(default_path),
        "explicit_settings_file": _settings_file_record(explicit_path),
        "source_fields": source_fields,
        "raw_inputs": {
            "app_file": deepcopy(dict(default_data)),
            "explicit_config": deepcopy(dict(explicit_data)),
            "runtime_overrides": deepcopy(dict(override_data)),
        },
        "resolved": resolved_data,
    }

    return audit


def _settings_file_record(path: Path | None) -> dict[str, Any]:
    if path is None:
        status = "not_provided_or_missing"
        path_text = None
        raw_text = None
    else:
        status = "loaded" if path.is_file() else "missing"
        path_text = str(path)
        raw_text = path.read_text(encoding="utf-8") if path.is_file() else None

    record = {"path": path_text,
              "status": status,
              "raw_text": raw_text}

    return record


def _leaf_paths(data: Mapping[str, Any],
                prefix: str = "") -> set[str]:
    paths: set[str] = set()

    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            paths.update(_leaf_paths(value, path))
        else:
            paths.add(path)

    return paths
