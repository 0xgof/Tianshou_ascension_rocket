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
    assert components["precision_orbit_bonus"] == pytest.approx(2.0)
    assert components["altitude_overshoot_penalty"] == 0.0
    assert components["tangential_velocity_overspeed_penalty"] == 0.0
    assert components["fuel_penalty"] == pytest.approx(-0.165296176, rel=1e-5)
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
    assert exact_components["precision_orbit_bonus"] == pytest.approx(2.0)
    assert exact_components["altitude_overshoot_penalty"] == 0.0
    assert exact_components["tangential_velocity_overspeed_penalty"] == 0.0
    assert low_components["target_altitude_bonus"] == pytest.approx(0.581344,
                                                                    rel=1e-5)
    assert low_components["tangential_velocity_bonus"] == pytest.approx(0.072196,
                                                                        rel=1e-5)
    assert low_components["precision_orbit_bonus"] < 1e-8
    assert low_components["altitude_overshoot_penalty"] == 0.0
    assert low_components["tangential_velocity_overspeed_penalty"] == 0.0


def test_target_orbit_altitude_bonus_is_stricter_above_target_than_below():
    target_info = {
        "altitude": 200_000.0,
        "target_altitude": 200_000.0,
        "tangential_velocity": 8_000.0,
        "target_orbital_velocity": 8_000.0,
        "fuel_used_ratio": 0.0,
        "applied_action": [1.0, 90.0],
        "crashed": False,
        "success": False,
    }
    undershoot_info = dict(target_info)
    undershoot_info["altitude"] = 150_000.0
    overshoot_info = dict(target_info)
    overshoot_info["altitude"] = 250_000.0

    _undershoot_reward, undershoot_components = reward_function(
        undershoot_info,
        "target_orbit_v1",
    )
    _overshoot_reward, overshoot_components = reward_function(
        overshoot_info,
        "target_orbit_v1",
    )

    assert undershoot_components["target_altitude_bonus"] > (
        overshoot_components["target_altitude_bonus"]
    )


def test_target_orbit_tangential_velocity_bonus_compresses_low_tail():
    target_info = {
        "altitude": 200_000.0,
        "target_altitude": 200_000.0,
        "tangential_velocity": 8_000.0,
        "target_orbital_velocity": 8_000.0,
        "fuel_used_ratio": 0.0,
        "applied_action": [1.0, 90.0],
        "crashed": False,
        "success": False,
    }
    low_tail_info = dict(target_info)
    low_tail_info["tangential_velocity"] = 4_000.0
    useful_info = dict(target_info)
    useful_info["tangential_velocity"] = 6_400.0

    _target_reward, target_components = reward_function(target_info,
                                                        "target_orbit_v1")
    _low_tail_reward, low_tail_components = reward_function(low_tail_info,
                                                            "target_orbit_v1")
    _useful_reward, useful_components = reward_function(useful_info,
                                                        "target_orbit_v1")

    assert target_components["tangential_velocity_bonus"] == 1.0
    assert low_tail_components["tangential_velocity_bonus"] > 0.0
    assert low_tail_components["tangential_velocity_bonus"] < 0.544946
    assert useful_components["tangential_velocity_bonus"] > (
        low_tail_components["tangential_velocity_bonus"]
    )


def test_target_orbit_tangential_velocity_bonus_is_skewed_right():
    target_info = {
        "altitude": 200_000.0,
        "target_altitude": 200_000.0,
        "tangential_velocity": 8_000.0,
        "target_orbital_velocity": 8_000.0,
        "fuel_used_ratio": 0.0,
        "applied_action": [1.0, 90.0],
        "crashed": False,
        "success": False,
    }
    underspeed_info = dict(target_info)
    underspeed_info["tangential_velocity"] = 6_400.0
    overspeed_info = dict(target_info)
    overspeed_info["tangential_velocity"] = 9_600.0

    _underspeed_reward, underspeed_components = reward_function(
        underspeed_info,
        "target_orbit_v1",
    )
    _overspeed_reward, overspeed_components = reward_function(
        overspeed_info,
        "target_orbit_v1",
    )

    assert overspeed_components["tangential_velocity_bonus"] > (
        underspeed_components["tangential_velocity_bonus"]
    )


def test_target_orbit_precision_bonus_requires_both_orbit_terms():
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
    altitude_only_info = dict(exact_info)
    altitude_only_info["tangential_velocity"] = 4_000.0
    velocity_only_info = dict(exact_info)
    velocity_only_info["altitude"] = 100_000.0

    _exact_reward, exact_components = reward_function(exact_info,
                                                      "target_orbit_v1")
    _altitude_reward, altitude_components = reward_function(altitude_only_info,
                                                            "target_orbit_v1")
    _velocity_reward, velocity_components = reward_function(velocity_only_info,
                                                            "target_orbit_v1")

    assert exact_components["precision_orbit_bonus"] == pytest.approx(2.0)
    assert altitude_components["precision_orbit_bonus"] < 1e-4
    assert velocity_components["precision_orbit_bonus"] < 1e-4


def test_target_orbit_bonuses_are_zero_at_zero_altitude_and_velocity_ratios():
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
    assert components["precision_orbit_bonus"] == 0.0
    assert components["altitude_overshoot_penalty"] == 0.0
    assert components["tangential_velocity_overspeed_penalty"] == 0.0


def test_target_orbit_altitude_overshoot_penalty_is_smooth_and_capped():
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
    undershoot_info = dict(exact_info)
    undershoot_info["altitude"] = 100_000.0
    small_overshoot_info = dict(exact_info)
    small_overshoot_info["altitude"] = 202_000.0
    near_limit_info = dict(exact_info)
    near_limit_info["altitude"] = 203_750.0
    far_info = dict(exact_info)
    far_info["altitude"] = 400_000.0

    _exact_reward, exact_components = reward_function(exact_info,
                                                      "target_orbit_v1")
    _undershoot_reward, undershoot_components = reward_function(
        undershoot_info,
        "target_orbit_v1",
    )
    _small_overshoot_reward, small_overshoot_components = reward_function(
        small_overshoot_info,
        "target_orbit_v1",
    )
    _near_limit_reward, near_limit_components = reward_function(
        near_limit_info,
        "target_orbit_v1",
    )
    _far_reward, far_components = reward_function(far_info,
                                                  "target_orbit_v1")

    assert exact_components["altitude_overshoot_penalty"] == 0.0
    assert undershoot_components["altitude_overshoot_penalty"] == 0.0
    assert small_overshoot_components["altitude_overshoot_penalty"] == pytest.approx(-0.25)
    assert near_limit_components["altitude_overshoot_penalty"] == -0.5
    assert far_components["altitude_overshoot_penalty"] == -0.5


def test_target_orbit_velocity_overspeed_penalty_is_smooth_and_capped():
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
    underspeed_info = dict(exact_info)
    underspeed_info["tangential_velocity"] = 4_000.0
    small_overspeed_info = dict(exact_info)
    small_overspeed_info["tangential_velocity"] = 8_080.0
    near_limit_info = dict(exact_info)
    near_limit_info["tangential_velocity"] = 8_150.0
    far_info = dict(exact_info)
    far_info["tangential_velocity"] = 80_000.0

    _exact_reward, exact_components = reward_function(exact_info,
                                                      "target_orbit_v1")
    _underspeed_reward, underspeed_components = reward_function(
        underspeed_info,
        "target_orbit_v1",
    )
    _small_overspeed_reward, small_overspeed_components = reward_function(
        small_overspeed_info,
        "target_orbit_v1",
    )
    _near_limit_reward, near_limit_components = reward_function(
        near_limit_info,
        "target_orbit_v1",
    )
    _far_reward, far_components = reward_function(far_info,
                                                  "target_orbit_v1")

    assert exact_components["tangential_velocity_overspeed_penalty"] == 0.0
    assert underspeed_components["tangential_velocity_overspeed_penalty"] == 0.0
    assert small_overspeed_components["tangential_velocity_overspeed_penalty"] == pytest.approx(-0.25)
    assert near_limit_components["tangential_velocity_overspeed_penalty"] == -0.5
    assert far_components["tangential_velocity_overspeed_penalty"] == -0.5


def test_target_orbit_reward_keeps_pad_inactivity_penalty_disabled():
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

    assert components["inactivity_penalty"] == 0.0


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
