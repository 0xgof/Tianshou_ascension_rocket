"""Reward-function plotting helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rocket_env.reward import TARGET_ORBIT_REWARD_V1, reward_function


TARGET_ALTITUDE = 200_000.0
TARGET_ORBITAL_VELOCITY = 8_000.0


def plot_target_orbit_reward_functions(output_path: str | Path) -> None:
    ratio_values = np.linspace(0.0, 2.0, 401)
    fuel_ratio_values = np.linspace(0.0, 1.0, 201)

    altitude_rewards = np.array([_target_orbit_component(altitude_ratio=ratio,
                                                         velocity_ratio=1.0,
                                                         fuel_ratio=0.0,
                                                         component="target_altitude_bonus")
                                 for ratio in ratio_values])
    velocity_rewards = np.array([_target_orbit_component(altitude_ratio=1.0,
                                                         velocity_ratio=ratio,
                                                         fuel_ratio=0.0,
                                                         component="tangential_velocity_bonus")
                                 for ratio in ratio_values])
    total_match_rewards = altitude_rewards + velocity_rewards
    fuel_penalties = np.array([_target_orbit_component(altitude_ratio=1.0,
                                                       velocity_ratio=1.0,
                                                       fuel_ratio=fuel_ratio,
                                                       component="fuel_penalty")
                               for fuel_ratio in fuel_ratio_values])

    inactivity_penalty = _target_orbit_component(altitude_ratio=0.0,
                                                 velocity_ratio=0.0,
                                                 fuel_ratio=0.0,
                                                 throttle=0.0,
                                                 component="inactivity_penalty")
    crash_penalty = _target_orbit_component(altitude_ratio=1.0,
                                            velocity_ratio=1.0,
                                            fuel_ratio=0.0,
                                            crashed=True,
                                            component="crash_penalty")

    fig, axes = plt.subplots(nrows=2,
                             ncols=2,
                             figsize=(12, 8))
    altitude_axis = axes[0, 0]
    velocity_axis = axes[0, 1]
    fuel_axis = axes[1, 0]
    event_axis = axes[1, 1]

    altitude_axis.plot(ratio_values,
                       altitude_rewards,
                       color="#1f77b4",
                       linewidth=2.0,
                       label="altitude target reward")
    altitude_axis.axvline(1.0,
                          color="#444444",
                          linestyle="--",
                          linewidth=1.0,
                          label="target ratio")
    altitude_axis.set_title("Altitude Match Reward")
    altitude_axis.set_xlabel("altitude / target_altitude")
    altitude_axis.set_ylabel("reward")
    altitude_axis.set_ylim(-0.05, 1.05)
    altitude_axis.grid(True, alpha=0.25)
    altitude_axis.legend(loc="upper right")

    velocity_axis.plot(ratio_values,
                       velocity_rewards,
                       color="#9467bd",
                       linewidth=2.0,
                       label="velocity target reward")
    velocity_axis.plot(ratio_values,
                       total_match_rewards,
                       color="#2ca02c",
                       linewidth=1.4,
                       alpha=0.7,
                       label="altitude + velocity if ratios match")
    velocity_axis.axvline(1.0,
                          color="#444444",
                          linestyle="--",
                          linewidth=1.0,
                          label="target ratio")
    velocity_axis.set_title("Tangential Velocity Match Reward")
    velocity_axis.set_xlabel("tangential_velocity / target_orbital_velocity")
    velocity_axis.set_ylabel("reward")
    velocity_axis.grid(True, alpha=0.25)
    velocity_axis.legend(loc="upper right")

    fuel_axis.plot(fuel_ratio_values,
                   fuel_penalties,
                   color="#ff7f0e",
                   linewidth=2.0,
                   label="fuel penalty")
    fuel_axis.axhline(-1.0,
                      color="#444444",
                      linestyle="--",
                      linewidth=1.0,
                      label="full fuel use")
    fuel_axis.set_title("Fuel Consumption Penalty")
    fuel_axis.set_xlabel("fuel_used / initial_fuel")
    fuel_axis.set_ylabel("penalty")
    fuel_axis.grid(True, alpha=0.25)
    fuel_axis.legend(loc="lower left")

    event_names = ["inactivity", "crash"]
    event_penalties = [inactivity_penalty, crash_penalty]
    event_axis.bar(event_names,
                   event_penalties,
                   color=["#8c564b", "#d62728"])
    event_axis.axhline(0.0,
                       color="#444444",
                       linewidth=1.0)
    event_axis.set_title("Discrete Penalties")
    event_axis.set_ylabel("penalty")
    event_axis.grid(True, axis="y", alpha=0.25)

    fig.suptitle("target_orbit_v1 reward and penalty functions")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _target_orbit_component(*,
                            altitude_ratio: float,
                            velocity_ratio: float,
                            fuel_ratio: float,
                            component: str,
                            throttle: float = 1.0,
                            crashed: bool = False) -> float:
    info = {"altitude": altitude_ratio * TARGET_ALTITUDE,
            "target_altitude": TARGET_ALTITUDE,
            "tangential_velocity": velocity_ratio * TARGET_ORBITAL_VELOCITY,
            "target_orbital_velocity": TARGET_ORBITAL_VELOCITY,
            "fuel_used_ratio": fuel_ratio,
            "applied_action": [throttle, 90.0],
            "crashed": crashed,
            "success": False}

    _reward, components = reward_function(info, TARGET_ORBIT_REWARD_V1)
    value = float(components[component])

    return value
