from math import log

import numpy as np
import gymnasium as gym

from rocket_env.env import EnvConfig, RocketAscentEnv
from rocket_env.physics import PhysicsConfig, RocketState
from rocket_env.rollouts import radial_thrust_policy, random_policy, run_episode


def test_reset_returns_observation_and_info():
    env = RocketAscentEnv()
    assert isinstance(env, gym.Env)
    observation, info = env.reset(seed=1)
    assert isinstance(observation, dict)
    assert isinstance(info, dict)


def test_observation_contains_state_and_goal():
    env = RocketAscentEnv()
    observation, _info = env.reset(seed=1)
    assert set(observation) == {"state", "goal"}
    assert observation["state"].shape == (10,)
    assert np.all(np.isfinite(observation["state"]))
    assert np.all(np.isfinite(observation["goal"]))


def test_observation_tracks_episode_initial_fuel():
    env = RocketAscentEnv()
    observation, _info = env.reset(seed=1, options={"initial_fuel": 123.0})
    assert observation["state"][9] == 123.0


def test_action_space_has_signed_command_rates():
    env = RocketAscentEnv()
    assert env.action_space.shape == (2,)
    np.testing.assert_allclose(env.action_space.low, np.array([-1.0, -1.0]))
    np.testing.assert_allclose(env.action_space.high, np.array([1.0, 1.0]))
    assert not env.action_space.contains(np.array([1.1, 0.0], dtype=np.float32))
    assert env.action_space.contains(np.array([1.0, -1.0], dtype=np.float32))


def test_reset_clears_applied_command_state():
    env = RocketAscentEnv()
    env.reset(seed=1)
    env.step(np.array([1.0, -1.0], dtype=np.float32))
    _obs, info = env.reset(seed=1)
    np.testing.assert_allclose(info["command_delta"], np.array([0.0, 0.0]))
    np.testing.assert_allclose(info["applied_action"], np.array([0.0, 90.0]))


def test_default_rocket_has_realistic_liftoff_throttle():
    env = RocketAscentEnv()
    initial_mass = (
        env.config.dry_mass
        + env.config.payload_mass
        + env.config.initial_fuel
    )
    liftoff_throttle = (
        initial_mass
        * env.config.physics.g0
        / env.config.physics.max_thrust
    )
    assert 0.55 <= liftoff_throttle <= 0.95


def test_default_rocket_has_realistic_single_stage_delta_v_budget():
    env = RocketAscentEnv()
    initial_mass = (
        env.config.dry_mass
        + env.config.payload_mass
        + env.config.initial_fuel
    )
    final_mass = env.config.dry_mass + env.config.payload_mass
    exhaust_velocity = (
        env.config.physics.max_thrust / env.config.physics.max_fuel_burn_rate
    )
    ideal_delta_v = exhaust_velocity * log(initial_mass / final_mass)
    final_thrust_to_weight = (
        env.config.physics.max_thrust / (final_mass * env.config.physics.g0)
    )
    assert 4_000.0 <= ideal_delta_v <= 6_000.0
    assert final_thrust_to_weight <= 7.0


def test_step_return_contract():
    env = RocketAscentEnv()
    env.reset(seed=1)
    observation, reward, terminated, truncated, info = env.step(np.array([1.0, 0.0]))
    assert isinstance(observation, dict)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_step_advances_time():
    env = RocketAscentEnv()
    env.reset(seed=1)
    observation, *_ = env.step(np.array([1.0, 0.0]))
    assert observation["state"][6] == env.config.physics.dt


def test_step_keeps_observation_finite():
    env = RocketAscentEnv()
    env.reset(seed=1)
    observation, *_ = env.step(np.array([1.0, 0.0]))
    assert np.all(np.isfinite(observation["state"]))
    assert np.all(np.isfinite(observation["goal"]))


def test_crash_termination_when_altitude_below_ground():
    env = RocketAscentEnv()
    env.reset(seed=1)
    env.force_state(
        RocketState(
            position=np.array([env.config.physics.planet_radius - 1.0, 0.0]),
            velocity=np.array([-1.0, 0.0]),
            dry_mass=env.config.dry_mass,
            payload_mass=env.config.payload_mass,
            remaining_fuel=env.config.initial_fuel,
            time=0.0,
        )
    )
    _obs, _reward, terminated, _truncated, info = env.step(np.array([0.0, 0.0]))
    assert terminated
    assert info["crashed"] is True


def test_low_thrust_on_launch_pad_does_not_end_episode():
    env = RocketAscentEnv()
    env.reset(seed=1)
    observation, _reward, terminated, truncated, info = env.step(
        np.array([1.0, 0.0], dtype=np.float32)
    )
    assert terminated is False
    assert truncated is False
    assert info["crashed"] is False
    assert info["altitude"] == 0.0
    assert abs(info["vertical_speed"]) < 1e-9
    assert info["fuel_used_ratio"] > 0.0
    assert info["fuel_used_ratio"] < 1.0
    np.testing.assert_allclose(info["fuel_used_ratio"],
                               info["fuel_used_mass"] / env.initial_fuel)
    assert observation["state"][6] == env.config.physics.dt


def test_reset_info_contains_sea_level_density():
    env = RocketAscentEnv()
    _observation, info = env.reset(seed=1)
    assert info["density"] == env.config.physics.rho0


def test_one_percent_throttle_does_not_lift_default_rocket():
    env = RocketAscentEnv()
    env.reset(seed=1)
    _obs, _reward, terminated, truncated, info = env.step(
        np.array([1.0, 0.0], dtype=np.float32)
    )
    assert terminated is False
    assert truncated is False
    assert info["crashed"] is False
    assert info["altitude"] == 0.0


def test_environment_applies_command_rate_limits():
    env = RocketAscentEnv()
    env.reset(seed=1)
    _obs, _reward, _terminated, _truncated, info = env.step(
        np.array([1.0, -1.0], dtype=np.float32)
    )
    np.testing.assert_allclose(
        info["command_delta"],
        np.array([
            env.config.throttle_change_per_second * env.config.physics.dt,
            0.0,
        ]),
    )
    np.testing.assert_allclose(info["applied_action"], np.array([0.5, 90.0]))


def test_environment_command_rate_limits_scale_with_dt():
    env = RocketAscentEnv(
        EnvConfig(
            physics=PhysicsConfig(dt=0.25),
            throttle_change_per_second=0.5,
            angle_change_per_second=4.0,
        )
    )
    env.reset(seed=1)
    _obs, _reward, _terminated, _truncated, info = env.step(
        np.array([1.0, -1.0], dtype=np.float32)
    )
    np.testing.assert_allclose(info["command_delta"], np.array([0.125, 0.0]))
    np.testing.assert_allclose(info["applied_action"], np.array([0.125, 90.0]))


def test_environment_allows_angle_changes_after_launch_lock_altitude():
    env = RocketAscentEnv()
    env.reset(seed=1)
    env.force_state(
        RocketState(
            position=np.array([env.config.physics.planet_radius + 11.0, 0.0]),
            velocity=np.zeros(2, dtype=float),
            dry_mass=env.config.dry_mass,
            payload_mass=env.config.payload_mass,
            remaining_fuel=env.config.initial_fuel,
            time=0.0,
        )
    )

    _obs, _reward, _terminated, _truncated, info = env.step(
        np.array([1.0, -1.0], dtype=np.float32)
    )

    np.testing.assert_allclose(
        info["command_delta"],
        np.array([
            env.config.throttle_change_per_second * env.config.physics.dt,
            -env.config.angle_change_per_second * env.config.physics.dt,
        ]),
    )
    np.testing.assert_allclose(info["applied_action"], np.array([0.5, 85.0]))


def test_low_throttle_rate_ramp_stays_on_launch_pad_without_crash():
    env = RocketAscentEnv(EnvConfig(throttle_change_per_second=0.01))
    env.reset(seed=1)
    for _ in range(20):
        _obs, _reward, terminated, truncated, info = env.step(
            np.array([1.0, -1.0], dtype=np.float32)
        )
        assert terminated is False
        assert truncated is False
        assert info["crashed"] is False
        assert abs(info["altitude"]) <= 1e-6


def test_launch_pad_allows_gradual_throttle_until_liftoff():
    env = RocketAscentEnv(EnvConfig(throttle_change_per_second=0.01))
    env.reset(seed=1)

    for _ in range(3):
        _obs, _reward, terminated, truncated, info = env.step(
            np.array([1.0, 0.0], dtype=np.float32)
        )
        assert terminated is False
        assert truncated is False
        assert info["crashed"] is False
        assert info["altitude"] == 0.0

    fast_env = RocketAscentEnv(EnvConfig(throttle_change_per_second=1.0))
    fast_env.reset(seed=1)
    _obs, _reward, terminated, _truncated, info = fast_env.step(
        np.array([1.0, 0.0], dtype=np.float32)
    )
    assert terminated is False
    assert info["crashed"] is False
    assert info["altitude"] > 0.0


def test_max_step_truncation():
    env = RocketAscentEnv(EnvConfig(max_steps=1))
    env.reset(seed=1)
    _obs, _reward, _terminated, truncated, _info = env.step(np.array([1.0, 0.0]))
    assert truncated is True


def test_info_contains_required_diagnostics():
    env = RocketAscentEnv()
    env.reset(seed=1)
    *_unused, info = env.step(np.array([1.0, 0.0]))
    required = {
        "altitude",
        "altitude_gain",
        "speed",
        "vertical_speed",
        "tangential_velocity",
        "fuel_remaining",
        "fuel_used_ratio",
        "fuel_used_mass",
        "density",
        "ambient_pressure",
        "engine_efficiency",
        "gravity",
        "drag",
        "command_delta",
        "applied_action",
        "target_altitude",
        "target_orbital_velocity",
        "altitude_error",
        "tangential_velocity_error",
        "vertical_speed_error",
        "crashed",
        "success",
        "reward_components",
    }
    assert required.issubset(info)
    assert info["density"] >= 0.0
    assert info["ambient_pressure"] >= 0.0
    assert (
        env.config.physics.vacuum_relative_efficiency
        <= info["engine_efficiency"]
        <= env.config.physics.sea_level_relative_efficiency
    )
    assert np.asarray(info["gravity"]).shape == (2,)
    assert np.asarray(info["drag"]).shape == (2,)


def test_random_rollout_completes():
    env = RocketAscentEnv(EnvConfig(max_steps=25))
    trace, _info = run_episode(env, random_policy(env), max_steps=25, seed=1)
    assert len(trace) > 0
    assert len(trace) <= 25


def test_scripted_radial_thrust_rollout_completes():
    env = RocketAscentEnv(EnvConfig(max_steps=10))
    trace, _info = run_episode(env, radial_thrust_policy, max_steps=10, seed=1)
    assert len(trace) > 0
    assert any(step["info"]["fuel_used_ratio"] > 0.0 for step in trace)
