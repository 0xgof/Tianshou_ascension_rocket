import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import settings.loaders as loaders
from rocket_env.env import EnvConfig, RocketAscentEnv
from rocket_env.physics import PhysicsConfig
from settings import (
    DisplaySettings,
    EnvironmentSettings,
    PhysicsSettings,
    ProjectSettings,
    RewardSettings,
    load_settings,
    load_settings_with_audit,
)


ROOT = Path(__file__).resolve().parents[1]


def test_project_settings_defaults_load():
    project_settings = ProjectSettings()
    assert isinstance(project_settings.physics, PhysicsSettings)
    assert isinstance(project_settings.environment, EnvironmentSettings)
    assert isinstance(project_settings.reward, RewardSettings)
    assert isinstance(project_settings.display, DisplaySettings)
    assert project_settings.display.manual_control_throttle_step == 0.01
    assert project_settings.display.manual_simulation_rate_hz == 5.0
    assert project_settings.reward.selected_reward == "altitude_gain_v1"


def test_project_settings_default_thrust_to_weight_is_plausible():
    project_settings = ProjectSettings()
    env_config = project_settings.to_env_config()
    initial_mass = (
        env_config.dry_mass
        + env_config.payload_mass
        + env_config.initial_fuel
    )
    full_throttle_twr = env_config.physics.max_thrust / (
        initial_mass * env_config.physics.g0
    )
    assert 1.05 <= full_throttle_twr <= 1.8


@pytest.mark.parametrize(
    "field,value",
    [
        ("dt", 0.0),
        ("planet_radius", 0.0),
        ("mu", 0.0),
        ("g0", 0.0),
        ("rho0", -0.1),
        ("sea_level_pressure", 0.0),
        ("scale_height", 0.0),
        ("max_thrust", 0.0),
        ("max_thrust_angle", 0.0),
        ("sea_level_relative_efficiency", 0.0),
        ("sea_level_relative_efficiency", 1.1),
        ("vacuum_relative_efficiency", 0.0),
        ("vacuum_relative_efficiency", 1.1),
        ("reference_area", 0.0),
        ("min_radius", 0.0),
    ],
)
def test_physics_settings_reject_invalid_values(field, value):
    with pytest.raises(ValidationError):
        PhysicsSettings(**{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("dry_mass", 0.0),
        ("payload_mass", 0.0),
        ("initial_fuel", -1.0),
        ("initial_altitude", -1.0),
        ("target_altitude", -1.0),
        ("max_steps", 0),
        ("vertical_speed_tolerance", 0.0),
        ("throttle_change_per_second", 0.0),
        ("angle_change_per_second", 0.0),
    ],
)
def test_environment_settings_reject_invalid_values(field, value):
    with pytest.raises(ValidationError):
        EnvironmentSettings(**{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("default_steps", 0),
        ("width", 0),
        ("height", 0),
        ("fps", 0),
        ("manual_simulation_rate_hz", 0.0),
        ("figure_width", 0.0),
        ("figure_height", 0.0),
        ("manual_control_throttle_step", 0.0),
        ("manual_control_angle_step", 0.0),
        ("output_dir", ""),
    ],
)
def test_display_settings_reject_invalid_values(field, value):
    with pytest.raises(ValidationError):
        DisplaySettings(**{field: value})


def test_settings_module_has_no_rl_imports():
    for source_path in (ROOT / "src/settings").glob("*.py"):
        text = source_path.read_text(encoding="utf-8")
        assert "agents" not in text
        assert "models" not in text


def test_load_settings_without_file_returns_defaults(monkeypatch, tmp_path):
    monkeypatch.setattr(loaders, "DEFAULT_APP_SETTINGS_PATH", tmp_path / "missing.yaml")
    project_settings = load_settings()
    assert isinstance(project_settings, ProjectSettings)


def test_load_settings_from_yaml_overrides_defaults(tmp_path):
    config_path = tmp_path / "settings.yaml"
    config_path.write_text("physics:\n  dt: 0.25\n", encoding="utf-8")
    project_settings = load_settings(config_path)
    assert project_settings.physics.dt == 0.25


def test_load_settings_from_json_overrides_defaults(tmp_path):
    config_path = tmp_path / "settings.json"
    config_path.write_text(
        json.dumps({"environment": {"max_steps": 17}}),
        encoding="utf-8",
    )
    project_settings = load_settings(config_path)
    assert project_settings.environment.max_steps == 17


def test_explicit_overrides_win_over_file_values(tmp_path):
    config_path = tmp_path / "settings.yaml"
    config_path.write_text("display:\n  default_steps: 10\n", encoding="utf-8")
    project_settings = load_settings(
        config_path,
        overrides={"display": {"default_steps": 99}},
    )
    assert project_settings.display.default_steps == 99


def test_load_settings_does_not_mutate_defaults(monkeypatch, tmp_path):
    monkeypatch.setattr(loaders, "DEFAULT_APP_SETTINGS_PATH", tmp_path / "missing.yaml")
    first = load_settings(overrides={"display": {"default_steps": 22}})
    second = load_settings()
    assert first.display.default_steps == 22
    assert second.display.default_steps == DisplaySettings().default_steps


def test_load_settings_rejects_missing_file():
    with pytest.raises(FileNotFoundError):
        load_settings("configs/does-not-exist.yaml")


def test_load_settings_rejects_unsupported_extension(tmp_path):
    config_path = tmp_path / "settings.txt"
    config_path.write_text("display.default_steps=10", encoding="utf-8")
    with pytest.raises(ValueError):
        load_settings(config_path)


def test_load_settings_rejects_invalid_loaded_settings(tmp_path):
    config_path = tmp_path / "settings.yaml"
    config_path.write_text("environment:\n  max_steps: 0\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_settings(config_path)


def test_physics_settings_convert_to_physics_config():
    config = PhysicsSettings(dt=0.5).to_physics_config()
    assert isinstance(config, PhysicsConfig)
    assert config.dt == 0.5


def test_project_settings_convert_to_env_config():
    env_config = ProjectSettings(environment={"max_steps": 12}).to_env_config()
    assert isinstance(env_config, EnvConfig)
    assert env_config.max_steps == 12
    assert env_config.reward_name == "altitude_gain_v1"


def test_project_settings_pass_reward_selection_to_env_config():
    env_config = ProjectSettings(
        reward={"selected_reward": "altitude_gain_v1"},
    ).to_env_config()

    assert env_config.reward_name == "altitude_gain_v1"


def test_settings_converted_env_can_reset_and_step():
    env = RocketAscentEnv(ProjectSettings().to_env_config(max_steps=2))
    obs, _info = env.reset(seed=0)
    next_obs, _reward, _terminated, _truncated, _info = env.step(
        env.action_space.sample()
    )
    assert set(obs) == {"state", "goal"}
    assert set(next_obs) == {"state", "goal"}


def test_display_settings_expose_manual_control_steps():
    display = DisplaySettings(
        manual_control_throttle_step=0.2,
        manual_control_angle_step=0.3,
    )
    assert display.manual_control_throttle_step == 0.2
    assert display.manual_control_angle_step == 0.3


def test_display_settings_pass_manual_simulation_rate_to_display_config():
    display = DisplaySettings(manual_simulation_rate_hz=7.5)
    assert display.to_display_config().manual_simulation_rate_hz == 7.5


def test_runtime_modules_do_not_parse_settings_files():
    for path in [ROOT / "src/rocket_env/env.py", ROOT / "src/rocket_env/physics.py"]:
        text = path.read_text(encoding="utf-8")
        assert "load_settings" not in text
        assert "yaml" not in text.lower()
        assert "json" not in text.lower()


def test_default_app_settings_file_loads_when_present(monkeypatch, tmp_path):
    config_path = tmp_path / "app.yaml"
    config_path.write_text("display:\n  default_steps: 123\n", encoding="utf-8")
    monkeypatch.setattr(loaders, "DEFAULT_APP_SETTINGS_PATH", config_path)
    assert load_settings().display.default_steps == 123


def test_missing_default_app_settings_file_falls_back_to_hardcoded_defaults(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(loaders, "DEFAULT_APP_SETTINGS_PATH", tmp_path / "missing.yaml")
    project_settings = load_settings()
    assert project_settings.display.default_steps == DisplaySettings().default_steps


def test_explicit_config_path_overrides_default_app_settings_file(monkeypatch, tmp_path):
    default_path = tmp_path / "app.yaml"
    explicit_path = tmp_path / "explicit.yaml"
    default_path.write_text("display:\n  default_steps: 123\n", encoding="utf-8")
    explicit_path.write_text("display:\n  default_steps: 456\n", encoding="utf-8")
    monkeypatch.setattr(loaders, "DEFAULT_APP_SETTINGS_PATH", default_path)
    assert load_settings(explicit_path).display.default_steps == 456


def test_explicit_overrides_win_over_default_app_settings_file(monkeypatch, tmp_path):
    config_path = tmp_path / "app.yaml"
    config_path.write_text("display:\n  default_steps: 123\n", encoding="utf-8")
    monkeypatch.setattr(loaders, "DEFAULT_APP_SETTINGS_PATH", config_path)
    project_settings = load_settings(overrides={"display": {"default_steps": 789}})
    assert project_settings.display.default_steps == 789


def test_app_yaml_matches_project_settings_schema():
    project_settings = load_settings(ROOT / "configs/app.yaml")
    assert isinstance(project_settings, ProjectSettings)


def test_display_manual_config_loads():
    app_settings = load_settings(ROOT / "configs/app.yaml")
    project_settings = load_settings(ROOT / "configs/display_manual.yaml")
    assert project_settings.display.default_steps == 5000
    assert project_settings.environment.initial_fuel == app_settings.environment.initial_fuel


def test_settings_audit_records_sources_for_display_overlay():
    loaded_settings = load_settings_with_audit(ROOT / "configs/display_manual.yaml")
    audit = loaded_settings.audit

    assert audit["app_settings_file"]["status"] == "loaded"
    assert audit["explicit_settings_file"]["status"] == "loaded"
    assert "environment:" in audit["app_settings_file"]["raw_text"]
    assert "display:" in audit["explicit_settings_file"]["raw_text"]
    assert "environment.initial_fuel" in audit["source_fields"]["app_file"]
    assert "display.default_steps" in audit["source_fields"]["explicit_config"]
    assert audit["source_fields"]["runtime_overrides"] == []
    assert audit["resolved"]["environment"]["initial_fuel"] == load_settings(
        ROOT / "configs/app.yaml"
    ).environment.initial_fuel


def test_manual_fly_script_uses_settings_loader():
    text = (ROOT / "scripts/display/manual_fly.py").read_text(encoding="utf-8")
    assert "load_settings" in text
    assert "yaml" not in text.lower()
    assert "json" not in text.lower()


def test_script_wrappers_still_exist():
    for script in [
        "manual_fly.py",
        "watch_random.py",
        "watch_scripted.py",
        "run_visual_manual_checks.py",
    ]:
        assert (ROOT / "scripts" / script).is_file()


def test_validation_script_accepts_config_argument():
    text = (
        ROOT / "scripts/validation/run_visual_manual_checks.py"
    ).read_text(encoding="utf-8")
    assert 'add_argument("--config")' in text


def test_scripts_default_to_app_settings_file():
    for script in [
        ROOT / "scripts/display/manual_fly.py",
        ROOT / "scripts/display/watch_random.py",
        ROOT / "scripts/display/watch_scripted.py",
        ROOT / "scripts/validation/run_visual_manual_checks.py",
    ]:
        text = script.read_text(encoding="utf-8")
        assert (
            "load_settings(args.config" in text
            or "load_settings_with_audit(args.config" in text
        )
