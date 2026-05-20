import pytest

from rocket_env.reward import get_reward_function, reward_function


def test_default_reward_uses_altitude_gain_and_fuel_components():
    info = {
        "altitude_gain": 100.0,
        "altitude_error": 100_000.0,
        "target_altitude": 200_000.0,
        "vertical_speed_error": 100.0,
        "tangential_velocity_error": 1_000.0,
        "target_orbital_velocity": 8_000.0,
        "fuel_used_ratio": 0.25,
        "applied_action": [1.0, 90.0],
        "crashed": False,
        "success": False,
    }

    reward, components = reward_function(info)

    assert components["altitude_gain_bonus"] == 5.0
    assert components["fuel_penalty"] == -25.0
    assert components["crash_penalty"] == 0.0
    assert components["success_bonus"] == 0.0
    assert "altitude_error_penalty" not in components
    assert "vertical_speed_error_penalty" not in components
    assert "tangential_velocity_error_penalty" not in components
    assert reward == sum(components.values())


def test_target_orbit_reward_uses_altitude_and_velocity_bonuses():
    info = {
        "altitude": 200_000.0,
        "altitude_error": 100_000.0,
        "target_altitude": 200_000.0,
        "tangential_velocity": 8_000.0,
        "tangential_velocity_error": 2_000.0,
        "target_orbital_velocity": 8_000.0,
        "fuel_used_ratio": 0.25,
        "crashed": False,
        "success": False,
    }

    reward, components = reward_function(info, "target_orbit_v1")

    assert components["target_altitude_bonus"] == 1.0
    assert components["tangential_velocity_bonus"] == 1.0
    assert components["fuel_penalty"] == pytest.approx(-0.058526, rel=1e-5)
    assert "altitude_gain_bonus" not in components
    assert "altitude_error_penalty" not in components
    assert "orbit_match_bonus" not in components
    assert "success_bonus" not in components
    assert reward == sum(components.values())


def test_target_orbit_reward_full_fuel_consumption_penalty_is_one():
    info = {
        "altitude": 200_000.0,
        "target_altitude": 200_000.0,
        "tangential_velocity": 8_000.0,
        "target_orbital_velocity": 8_000.0,
        "fuel_used_ratio": 1.0,
        "applied_action": [1.0, 90.0],
        "crashed": False,
        "success": False,
    }

    _reward, components = reward_function(info, "target_orbit_v1")

    assert components["fuel_penalty"] == pytest.approx(-1.0)


def test_target_orbit_reward_falls_off_away_from_target_ratios():
    exact_info = {
        "altitude": 200_000.0,
        "target_altitude": 200_000.0,
        "tangential_velocity": 8_000.0,
        "target_orbital_velocity": 8_000.0,
        "fuel_used_ratio": 0.0,
        "applied_action": [1.0, 90.0],
        "crashed": False,
        "success": False,
    }
    low_info = dict(exact_info)
    low_info["altitude"] = 100_000.0
    low_info["tangential_velocity"] = 4_000.0

    _exact_reward, exact_components = reward_function(exact_info,
                                                      "target_orbit_v1")
    _low_reward, low_components = reward_function(low_info,
                                                  "target_orbit_v1")

    assert exact_components["target_altitude_bonus"] == pytest.approx(1.0)
    assert exact_components["tangential_velocity_bonus"] == pytest.approx(1.0)
    assert low_components["target_altitude_bonus"] == pytest.approx(0.544946,
                                                                    rel=1e-5)
    assert low_components["tangential_velocity_bonus"] == pytest.approx(0.544946,
                                                                        rel=1e-5)


def test_target_orbit_reward_is_zero_at_zero_altitude_and_velocity_ratios():
    info = {
        "altitude": 0.0,
        "target_altitude": 200_000.0,
        "tangential_velocity": 0.0,
        "target_orbital_velocity": 8_000.0,
        "fuel_used_ratio": 0.0,
        "applied_action": [1.0, 90.0],
        "crashed": False,
        "success": False,
    }

    _reward, components = reward_function(info, "target_orbit_v1")

    assert components["target_altitude_bonus"] == 0.0
    assert components["tangential_velocity_bonus"] == 0.0


def test_target_orbit_reward_penalizes_pad_inactivity():
    info = {
        "altitude": 0.0,
        "target_altitude": 200_000.0,
        "tangential_velocity": 0.0,
        "target_orbital_velocity": 8_000.0,
        "fuel_used_ratio": 0.0,
        "applied_action": [0.0, 90.0],
        "crashed": False,
        "success": False,
    }

    _reward, components = reward_function(info, "target_orbit_v1")

    assert components["inactivity_penalty"] == -0.01


def test_target_orbit_reward_does_not_penalize_active_pad_throttle():
    info = {
        "altitude": 0.0,
        "target_altitude": 200_000.0,
        "tangential_velocity": 0.0,
        "target_orbital_velocity": 8_000.0,
        "fuel_used_ratio": 0.0,
        "applied_action": [0.5, 90.0],
        "crashed": False,
        "success": False,
    }

    _reward, components = reward_function(info, "target_orbit_v1")

    assert components["inactivity_penalty"] == 0.0


def test_target_orbit_reward_applies_moderate_crash_penalty():
    info = {
        "altitude": 200_000.0,
        "target_altitude": 200_000.0,
        "tangential_velocity": 8_000.0,
        "target_orbital_velocity": 8_000.0,
        "fuel_used_ratio": 0.0,
        "applied_action": [1.0, 90.0],
        "crashed": True,
        "success": False,
    }

    _reward, components = reward_function(info, "target_orbit_v1")

    assert components["crash_penalty"] == -0.5


def test_reward_ignores_velocity_error_values():
    base_info = {
        "altitude_gain": 0.0,
        "altitude_error": 100_000.0,
        "target_altitude": 200_000.0,
        "vertical_speed_error": 0.0,
        "tangential_velocity_error": 0.0,
        "target_orbital_velocity": 8_000.0,
        "fuel_used_ratio": 0.25,
        "crashed": False,
        "success": False,
    }
    velocity_error_info = dict(base_info)
    velocity_error_info["vertical_speed_error"] = 10_000.0
    velocity_error_info["tangential_velocity_error"] = 10_000.0

    base_reward, _base_components = reward_function(base_info)
    velocity_error_reward, _velocity_error_components = reward_function(
        velocity_error_info,
    )

    assert velocity_error_reward == pytest.approx(base_reward)


def test_reward_does_not_bonus_negative_altitude_gain():
    info = {
        "altitude_gain": -100.0,
        "altitude_error": 100_000.0,
        "target_altitude": 200_000.0,
        "fuel_used_ratio": 0.25,
        "crashed": False,
        "success": False,
    }

    _reward, components = reward_function(info)

    assert components["altitude_gain_bonus"] == 0.0


def test_curriculum_low_altitude_reward_uses_reached_altitude_bonus():
    info = {
        "altitude": 500.0,
        "altitude_gain": 100.0,
        "altitude_error": 199_500.0,
        "target_altitude": 200_000.0,
        "fuel_used_ratio": 0.25,
        "crashed": False,
        "success": False,
    }

    _reward, components = reward_function(info, "curriculum_low_altitude_v1")

    assert components["leaving_pad_bonus"] == 25.0
    assert components["altitude_gain_bonus"] == 5.0
    assert components["altitude_reached_bonus"] == 100.0
    assert components["fuel_penalty"] == -6.25
    assert "altitude_error_penalty" not in components
    assert "success_bonus" not in components


def test_reward_applies_large_crash_penalty():
    info = {
        "altitude_gain": 0.0,
        "altitude_error": 100_000.0,
        "target_altitude": 200_000.0,
        "fuel_used_ratio": 0.25,
        "crashed": True,
        "success": False,
    }

    _reward, components = reward_function(info)

    assert components["crash_penalty"] == -100_000.0


def test_reward_repository_rejects_unknown_reward():
    with pytest.raises(ValueError, match="Unknown reward"):
        get_reward_function("missing_reward")
