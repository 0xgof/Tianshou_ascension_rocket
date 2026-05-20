import numpy as np
import pytest

from learning.adapters import (
    DEFAULT_FRACTIONAL_OBSERVATION_SIZE,
    DEFAULT_OBSERVATION_SIZE,
    FEATURE_NAMES,
    FRACTIONAL_FEATURE_NAMES,
    decode_physical_action,
    encode_observation,
    fractionalize_observation,
    to_tensor,
)
from rocket_env.env import EnvConfig, RocketAscentEnv


def test_encode_observation_returns_flat_vector():
    env = RocketAscentEnv(EnvConfig(max_steps=3))
    obs, _info = env.reset(seed=0)
    encoded = encode_observation(obs)
    assert encoded.shape == (DEFAULT_OBSERVATION_SIZE,)
    assert np.all(np.isfinite(encoded))
    assert FEATURE_NAMES[0] == "altitude_scaled"
    assert "fuel_fraction" in FEATURE_NAMES
    assert "initial_fuel_scaled" in FEATURE_NAMES
    assert FEATURE_NAMES[-1] == "tangential_velocity_error_scaled"


def test_feature_names_are_derived_from_feature_enums():
    import learning.adapters.observations as observations

    fractional_names = tuple(feature.value for feature in observations.FractionalFeature)
    encoded_names = tuple(feature.value for feature in observations.EncodedFeature)

    assert FRACTIONAL_FEATURE_NAMES == fractional_names
    assert FEATURE_NAMES == encoded_names


def test_encode_observation_shape_is_stable_after_step():
    env = RocketAscentEnv(EnvConfig(max_steps=3))
    obs, _info = env.reset(seed=0)
    first = encode_observation(obs)
    next_obs, *_ = env.step(np.array([0.0, 0.0], dtype=np.float32))
    second = encode_observation(next_obs)
    assert first.shape == second.shape == (DEFAULT_OBSERVATION_SIZE,)


def test_encode_observation_rejects_malformed_state_shape():
    observation = {"state": np.zeros(9, dtype=np.float32),
                   "goal": np.zeros(3, dtype=np.float32)}

    with pytest.raises(ValueError, match="Expected state vector"):
        encode_observation(observation)


def test_fractionalize_observation_returns_fractional_values_only():
    env = RocketAscentEnv(EnvConfig(max_steps=3))
    obs, _info = env.reset(seed=0)
    fractionalized = fractionalize_observation(obs)

    assert fractionalized.shape == (DEFAULT_FRACTIONAL_OBSERVATION_SIZE,)
    assert "current_throttle_command" not in FRACTIONAL_FEATURE_NAMES
    assert "current_angle_sin" not in FRACTIONAL_FEATURE_NAMES
    assert "current_angle_cos" not in FRACTIONAL_FEATURE_NAMES


def test_encode_observation_combines_fractional_and_command_features():
    env = RocketAscentEnv(EnvConfig(max_steps=3))
    obs, _info = env.reset(seed=0)
    encoded = encode_observation(obs)
    fractionalized = fractionalize_observation(obs)

    np.testing.assert_allclose(encoded[0:5], fractionalized[0:5])
    np.testing.assert_allclose(encoded[8:], fractionalized[5:])


def test_encode_observation_includes_command_features():
    env = RocketAscentEnv(EnvConfig(max_steps=3))
    obs, _info = env.reset(seed=0)
    encoded = encode_observation(obs)
    feature_index = {name: index for index, name in enumerate(FEATURE_NAMES)}
    np.testing.assert_allclose(
        encoded[feature_index["current_throttle_command"]],
        0.0,
    )
    np.testing.assert_allclose(encoded[feature_index["current_angle_sin"]], 1.0)
    np.testing.assert_allclose(encoded[feature_index["current_angle_cos"]], 0.0, atol=1e-7)


def test_encode_observation_uses_episode_initial_fuel_fraction():
    env = RocketAscentEnv(EnvConfig(max_steps=3))
    obs, _info = env.reset(seed=0, options={"initial_fuel": 100.0})
    encoded = encode_observation(obs)
    feature_index = {name: index for index, name in enumerate(FEATURE_NAMES)}

    np.testing.assert_allclose(encoded[feature_index["fuel_fraction"]], 1.0)
    np.testing.assert_allclose(encoded[feature_index["initial_fuel_scaled"]], 100.0 / 600_000.0)

    next_obs, *_ = env.step(np.array([1.0, 0.0], dtype=np.float32))
    next_encoded = encode_observation(next_obs)

    assert 0.0 <= next_encoded[feature_index["fuel_fraction"]] <= 1.0
    assert next_encoded[feature_index["fuel_fraction"]] < 1.0


def test_encode_observation_scales_large_physical_values():
    env = RocketAscentEnv(EnvConfig(max_steps=3))
    obs, _info = env.reset(seed=0)
    encoded = encode_observation(obs)
    feature_index = {name: index for index, name in enumerate(FEATURE_NAMES)}

    np.testing.assert_allclose(encoded[feature_index["altitude_scaled"]], 0.0)
    np.testing.assert_allclose(encoded[feature_index["vertical_speed_scaled"]], 0.0)
    np.testing.assert_allclose(encoded[feature_index["tangential_velocity_scaled"]], 0.0)
    np.testing.assert_allclose(encoded[feature_index["initial_fuel_scaled"]], 1.0)
    np.testing.assert_allclose(encoded[feature_index["target_altitude_scaled"]], 1.0)
    np.testing.assert_allclose(encoded[feature_index["target_orbital_velocity_scaled"]], 1.0)
    np.testing.assert_allclose(encoded[feature_index["altitude_error_scaled"]], 1.0)
    np.testing.assert_allclose(encoded[feature_index["vertical_speed_error_scaled"]], 0.0)
    np.testing.assert_allclose(encoded[feature_index["tangential_velocity_error_scaled"]], 1.0)


def test_decode_action_clips_to_environment_bounds():
    action_pair = decode_physical_action(np.array([2.0, 120.0], dtype=np.float32))
    np.testing.assert_allclose(action_pair.env_action, np.array([1.0, 1.0]))


def test_action_adapter_preserves_model_and_environment_actions():
    action_pair = decode_physical_action(np.array([2.0, 120.0], dtype=np.float32))
    np.testing.assert_allclose(action_pair.model_action, np.array([2.0, 120.0]))
    np.testing.assert_allclose(action_pair.env_action, np.array([1.0, 1.0]))


def test_tensor_conversion_uses_cpu_by_default():
    tensor = to_tensor(np.zeros(DEFAULT_OBSERVATION_SIZE, dtype=np.float32))
    assert tensor.device.type == "cpu"
    assert tensor.shape == (DEFAULT_OBSERVATION_SIZE,)


def test_adapters_do_not_import_learning_algorithms():
    import inspect
    import learning.adapters.actions as actions
    import learning.adapters.observations as observations

    source = inspect.getsource(actions) + inspect.getsource(observations)
    assert "learning.algorithms" not in source
