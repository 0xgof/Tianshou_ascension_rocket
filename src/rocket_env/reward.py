"""Reward calculation for the rocket ascent environment."""

from __future__ import annotations

from collections.abc import Callable
import math


RewardComponents = dict[str, float]
RewardResult = tuple[float, RewardComponents]
RewardFunction = Callable[[dict[str, object]], RewardResult]


ALTITUDE_GAIN_REWARD_V1 = "altitude_gain_v1"
TARGET_ORBIT_REWARD_V1 = "target_orbit_v1"
CURRICULUM_LOW_ALTITUDE_REWARD_V1 = "curriculum_low_altitude_v1"
DEFAULT_REWARD_NAME = ALTITUDE_GAIN_REWARD_V1


def altitude_gain_reward_v1(info: dict[str, object]) -> RewardResult:
    """Reward altitude gain, fuel efficiency, crash avoidance, and success."""

    altitude_gain_reference_m = 1_000.0
    altitude_gain_bonus_scale = 50.0
    fuel_used_ratio_penalty_scale = 100.0
    crash_penalty_value = -100_000.0
    success_bonus_value = 1_000.0

    altitude_gain_ratio = max(_safe_ratio(info["altitude_gain"],
                                          altitude_gain_reference_m),
                              0.0)

    # Bonus
    altitude_gain_bonus = altitude_gain_bonus_scale * altitude_gain_ratio
    success_bonus = success_bonus_value if info["success"] else 0.0

    # Penalties
    fuel_penalty = -fuel_used_ratio_penalty_scale * float(info["fuel_used_ratio"])
    crash_penalty = crash_penalty_value if info["crashed"] else 0.0

    components = {"altitude_gain_bonus": altitude_gain_bonus,
                  "success_bonus": success_bonus,
                  "fuel_penalty": fuel_penalty,
                  "crash_penalty": crash_penalty}

    total_reward = sum(components.values())
    reward_result = total_reward, components

    return reward_result


def target_orbit_reward_v1(info: dict[str, object]) -> RewardResult:
    """Reward matching the designed orbit altitude and tangential velocity."""

    target_altitude_below_std = 0.55
    target_altitude_above_std = 0.25
    target_tangential_velocity_below_std = 0.35
    target_tangential_velocity_above_std = 0.60
    target_tangential_velocity_tail_exponent = 2.5
    precision_orbit_std = 0.10
    precision_orbit_bonus_scale = 2.0
    altitude_overshoot_limit_ratio = 1.02
    altitude_overshoot_penalty_scale = 2.0
    altitude_overshoot_penalty_cap = 0.5
    tangential_velocity_overspeed_limit_ratio = 1.02
    tangential_velocity_overspeed_penalty_scale = 2.0
    tangential_velocity_overspeed_penalty_cap = 0.5
    fuel_penalty_curve = 1.0
    inactivity_altitude_m = 1.0
    inactivity_throttle_threshold = 0.01
    inactivity_penalty_value = 0.0
    crash_penalty_value = -0.5

    altitude_ratio = _safe_ratio(info["altitude"],
                                 info["target_altitude"])
    tangential_velocity_ratio = _safe_ratio(info["tangential_velocity"],
                                            info["target_orbital_velocity"])
    applied_throttle = _applied_throttle(info)

    # Bonus
    target_altitude_bonus = _asymmetric_gaussian_target_reward(
        altitude_ratio,
        target_altitude_below_std,
        target_altitude_above_std,
    )
    tangential_velocity_bonus = _tail_compressed_asymmetric_gaussian_target_reward(
        tangential_velocity_ratio,
        target_tangential_velocity_below_std,
        target_tangential_velocity_above_std,
        target_tangential_velocity_tail_exponent,
    )
    precision_altitude_match = _gaussian_target_reward(altitude_ratio,
                                                       precision_orbit_std)
    precision_velocity_match = _gaussian_target_reward(tangential_velocity_ratio,
                                                       precision_orbit_std)
    precision_orbit_bonus = (precision_orbit_bonus_scale
                             * precision_altitude_match
                             * precision_velocity_match)

    # Penalties
    altitude_overshoot_penalty = _upper_overshoot_penalty(
        altitude_ratio,
        altitude_overshoot_limit_ratio,
        altitude_overshoot_penalty_scale,
        altitude_overshoot_penalty_cap,
    )
    tangential_velocity_overspeed_penalty = _upper_overshoot_penalty(
        tangential_velocity_ratio,
        tangential_velocity_overspeed_limit_ratio,
        tangential_velocity_overspeed_penalty_scale,
        tangential_velocity_overspeed_penalty_cap,
    )
    fuel_penalty = -_exponential_unit_penalty(float(info["fuel_used_ratio"]),
                                              fuel_penalty_curve)
    inactivity_penalty = (
        inactivity_penalty_value
        if float(info["altitude"]) <= inactivity_altitude_m
        and applied_throttle <= inactivity_throttle_threshold
        else 0.0
    )
    crash_penalty = crash_penalty_value if info["crashed"] else 0.0

    components = {"target_altitude_bonus": target_altitude_bonus,
                  "tangential_velocity_bonus": tangential_velocity_bonus,
                  "precision_orbit_bonus": precision_orbit_bonus,
                  "altitude_overshoot_penalty": altitude_overshoot_penalty,
                  "tangential_velocity_overspeed_penalty": (
                      tangential_velocity_overspeed_penalty
                  ),
                  "fuel_penalty": fuel_penalty,
                  "inactivity_penalty": inactivity_penalty,
                  "crash_penalty": crash_penalty}

    total_reward = sum(components.values())
    reward_result = total_reward, components

    return reward_result


def curriculum_low_altitude_reward_v1(info: dict[str, object]) -> RewardResult:
    """Early curriculum reward for learning liftoff and low-altitude milestones."""

    altitude_gain_reference_m = 1_000.0
    altitude_gain_bonus_scale = 50.0
    low_altitude_reference_m = 1_000.0
    leaving_pad_altitude_m = 1.0
    leaving_pad_bonus_value = 25.0
    low_altitude_reached_bonus_scale = 200.0
    fuel_used_ratio_penalty_scale = 25.0
    crash_penalty_value = -500.0

    altitude_gain_ratio = max(_safe_ratio(info["altitude_gain"],
                                          altitude_gain_reference_m),
                              0.0)

    altitude_reached_ratio = max(_safe_ratio(info["altitude"],
                                             low_altitude_reference_m),
                                 0.0)
    altitude_reached_ratio = min(altitude_reached_ratio, 1.0)

    # Bonus
    leaving_pad_bonus = (
        leaving_pad_bonus_value
        if float(info["altitude"]) > leaving_pad_altitude_m
        else 0.0
    )
    altitude_gain_bonus = altitude_gain_bonus_scale * altitude_gain_ratio
    altitude_reached_bonus = low_altitude_reached_bonus_scale * altitude_reached_ratio

    # Penalties
    fuel_penalty = -fuel_used_ratio_penalty_scale * float(info["fuel_used_ratio"])
    crash_penalty = crash_penalty_value if info["crashed"] else 0.0

    components = {"leaving_pad_bonus": leaving_pad_bonus,
                  "altitude_gain_bonus": altitude_gain_bonus,
                  "altitude_reached_bonus": altitude_reached_bonus,
                  "fuel_penalty": fuel_penalty,
                  "crash_penalty": crash_penalty}

    total_reward = sum(components.values())
    reward_result = total_reward, components

    return reward_result


def reward_function(info: dict[str, object],
                    reward_name: str = DEFAULT_REWARD_NAME) -> RewardResult:
    """Compute total reward using one named reward from the repository."""

    selected_reward = get_reward_function(reward_name)
    reward = selected_reward(info)

    return reward


def get_reward_function(reward_name: str) -> RewardFunction:
    try:
        reward = REWARD_REPOSITORY[reward_name]
    except KeyError as exc:
        available_rewards = ", ".join(sorted(REWARD_REPOSITORY))
        message = f"Unknown reward '{reward_name}'. Available rewards: {available_rewards}"

        raise ValueError(message) from exc

    return reward


REWARD_REPOSITORY: dict[str, RewardFunction] = {
    ALTITUDE_GAIN_REWARD_V1: altitude_gain_reward_v1,
    TARGET_ORBIT_REWARD_V1: target_orbit_reward_v1,
    CURRICULUM_LOW_ALTITUDE_REWARD_V1: curriculum_low_altitude_reward_v1,
}


def _safe_ratio(value: object,
                scale: object) -> float:
    ratio = float(value) / max(abs(float(scale)), 1e-9)

    return ratio


def _gaussian_target_reward(ratio: float,
                            std: float) -> float:
    baseline_reward = _raw_gaussian_target_reward(0.0, std)
    raw_reward = _raw_gaussian_target_reward(ratio, std)
    reward = (raw_reward - baseline_reward) / (1.0 - baseline_reward)
    reward = max(reward, 0.0)

    return reward


def _asymmetric_gaussian_target_reward(ratio: float,
                                       below_target_std: float,
                                       above_target_std: float) -> float:
    target_std = below_target_std if ratio <= 1.0 else above_target_std
    reward = _gaussian_target_reward(ratio, target_std)

    return reward


def _tail_compressed_gaussian_target_reward(ratio: float,
                                            std: float,
                                            tail_exponent: float) -> float:
    raw_reward = _gaussian_target_reward(ratio, std)
    reward = raw_reward ** tail_exponent

    return reward


def _tail_compressed_asymmetric_gaussian_target_reward(ratio: float,
                                                       below_target_std: float,
                                                       above_target_std: float,
                                                       tail_exponent: float) -> float:
    raw_reward = _asymmetric_gaussian_target_reward(ratio,
                                                   below_target_std,
                                                   above_target_std)
    reward = raw_reward ** tail_exponent

    return reward


def _raw_gaussian_target_reward(ratio: float,
                                std: float) -> float:
    normalized_error = (ratio - 1.0) / std
    reward = math.exp(-0.5 * normalized_error * normalized_error)

    return reward


def _exponential_unit_penalty(value_ratio: float,
                              curve: float) -> float:
    clamped_ratio = min(max(value_ratio, 0.0), 1.0)
    numerator = math.exp(curve * clamped_ratio) - 1.0
    denominator = math.exp(curve) - 1.0
    penalty = numerator / denominator

    return penalty


def _upper_overshoot_penalty(value_ratio: float,
                             limit_ratio: float,
                             scale: float,
                             cap: float) -> float:
    if value_ratio <= 1.0:
        return 0.0

    if value_ratio >= limit_ratio:
        return -cap

    overshoot_fraction = (value_ratio - 1.0) / (limit_ratio - 1.0)
    raw_penalty = scale * cap * overshoot_fraction ** 2
    penalty = -min(raw_penalty, cap)

    return penalty


def _applied_throttle(info: dict[str, object]) -> float:
    applied_action = info.get("applied_action")
    if applied_action is None:
        return 0.0

    try:
        throttle = float(applied_action[0])  # type: ignore[index]
    except (IndexError, TypeError, ValueError):
        throttle = 0.0

    return throttle
