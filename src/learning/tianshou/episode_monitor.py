"""Episode-level training monitors for Tianshou runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


EpisodeAltitudeRecord = dict[str, Any]


@dataclass
class EpisodeAltitudeRecorder:
    """Collect max-altitude records for completed training episodes."""

    current_epoch: int = 0
    records: list[EpisodeAltitudeRecord] = field(default_factory=list)
    best_orbit_record: EpisodeAltitudeRecord | None = None
    best_orbit_trace: list[dict[str, Any]] = field(default_factory=list)

    def append(self,
               *,
               env_index: int,
               max_altitude: float,
               final_altitude: float,
               final_altitude_error: float,
               final_tangential_velocity_error: float,
               target_altitude: float,
               target_orbital_velocity: float,
               average_throttle: float,
               max_throttle: float,
               final_throttle: float,
               average_angle_degrees: float,
               final_angle_degrees: float,
               steps: int,
               total_reward: float,
               terminated: bool,
               truncated: bool) -> EpisodeAltitudeRecord:
        record = {"episode": len(self.records) + 1,
                  "epoch": int(self.current_epoch),
                  "env_index": int(env_index),
                  "max_altitude": float(max_altitude),
                  "final_altitude": float(final_altitude),
                  "final_altitude_error": float(final_altitude_error),
                  "final_tangential_velocity_error": float(final_tangential_velocity_error),
                  "target_altitude": float(target_altitude),
                  "target_orbital_velocity": float(target_orbital_velocity),
                  "average_throttle": float(average_throttle),
                  "max_throttle": float(max_throttle),
                  "final_throttle": float(final_throttle),
                  "average_angle_degrees": float(average_angle_degrees),
                  "final_angle_degrees": float(final_angle_degrees),
                  "steps": int(steps),
                  "total_reward": float(total_reward),
                  "terminated": bool(terminated),
                  "truncated": bool(truncated)}

        self.records.append(record)

        return record

    def consider_best_orbit_trace(self,
                                  record: EpisodeAltitudeRecord,
                                  trace: list[dict[str, Any]]) -> None:
        """Keep the full trace for the episode closest to the target orbit."""

        if self.best_orbit_record is None:
            self.best_orbit_record = dict(record)
            self.best_orbit_trace = list(trace)

            return

        previous_score = _orbit_closeness_score(self.best_orbit_record)
        current_score = _orbit_closeness_score(record)
        if current_score < previous_score:
            self.best_orbit_record = dict(record)
            self.best_orbit_trace = list(trace)


class EpisodeAltitudeMonitor(gym.Wrapper):
    """Record the maximum altitude reached in each completed episode."""

    def __init__(self,
                 env: gym.Env,
                 recorder: EpisodeAltitudeRecorder,
                 env_index: int) -> None:
        super().__init__(env)

        self.recorder = recorder
        self.env_index = env_index
        self.max_altitude = 0.0
        self.max_throttle = 0.0
        self.throttle_sum = 0.0
        self.final_throttle = 0.0
        self.angle_sum = 0.0
        self.final_angle_degrees = 90.0
        self.total_reward = 0.0
        self.steps = 0
        self.trace: list[dict[str, Any]] = []

    def reset(self,
              **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        initial_altitude = _info_altitude(info)

        self.max_altitude = initial_altitude
        self.max_throttle = 0.0
        self.throttle_sum = 0.0
        self.final_throttle = 0.0
        self.angle_sum = 0.0
        self.final_angle_degrees = 90.0
        self.total_reward = 0.0
        self.steps = 0
        self.trace = []

        return observation, info

    def step(self,
             action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        current_altitude = _info_altitude(info)
        current_altitude_error = _info_altitude_error(info)
        current_tangential_velocity_error = _info_tangential_velocity_error(info)
        target_altitude = _info_target_altitude(info)
        target_orbital_velocity = _info_target_orbital_velocity(info)
        current_throttle = _info_throttle(info)
        current_angle_degrees = _info_angle_degrees(info)

        self.max_altitude = max(self.max_altitude, current_altitude)
        self.max_throttle = max(self.max_throttle, current_throttle)
        self.throttle_sum += current_throttle
        self.final_throttle = current_throttle
        self.angle_sum += current_angle_degrees
        self.final_angle_degrees = current_angle_degrees
        self.total_reward += float(reward)
        self.steps += 1
        self.trace.append(_trace_entry(observation,
                                       action,
                                       reward,
                                       terminated,
                                       truncated,
                                       info))

        if terminated or truncated:
            average_throttle = self.throttle_sum / max(self.steps, 1)
            average_angle_degrees = self.angle_sum / max(self.steps, 1)
            record = self.recorder.append(
                env_index=self.env_index,
                max_altitude=self.max_altitude,
                final_altitude=current_altitude,
                final_altitude_error=current_altitude_error,
                final_tangential_velocity_error=current_tangential_velocity_error,
                target_altitude=target_altitude,
                target_orbital_velocity=target_orbital_velocity,
                average_throttle=average_throttle,
                max_throttle=self.max_throttle,
                final_throttle=self.final_throttle,
                average_angle_degrees=average_angle_degrees,
                final_angle_degrees=self.final_angle_degrees,
                steps=self.steps,
                total_reward=self.total_reward,
                terminated=terminated,
                truncated=truncated,
            )
            self.recorder.consider_best_orbit_trace(record, self.trace)

        return observation, reward, terminated, truncated, info


def plot_episode_altitudes(records: list[EpisodeAltitudeRecord],
                           output_path: str | Path,
                           title: str) -> None:
    """Plot per-episode max altitude with epoch boundaries."""

    if not records:
        return

    episodes = np.array([record["episode"] for record in records], dtype=float)
    max_altitudes = np.array([record["max_altitude"] for record in records],
                             dtype=float)
    final_altitude_errors = np.array([record.get("final_altitude_error", 0.0)
                                      for record in records],
                                     dtype=float)
    final_tangential_velocity_errors = np.array(
        [record.get("final_tangential_velocity_error", 0.0)
         for record in records],
        dtype=float,
    )
    average_throttles = np.array([record.get("average_throttle", 0.0)
                                  for record in records],
                                 dtype=float)
    final_throttles = np.array([record.get("final_throttle", 0.0)
                                for record in records],
                               dtype=float)
    average_angles = np.array([record.get("average_angle_degrees", 90.0)
                               for record in records],
                              dtype=float)
    final_angles = np.array([record.get("final_angle_degrees", 90.0)
                             for record in records],
                            dtype=float)
    total_rewards = np.array([record.get("total_reward", 0.0)
                              for record in records],
                             dtype=float)
    epochs = np.array([record["epoch"] for record in records], dtype=int)

    fig, axes = plt.subplots(nrows=6,
                             ncols=1,
                             figsize=(11, 15),
                             sharex=True)
    altitude_axis = axes[0]
    altitude_error_axis = axes[1]
    tangential_velocity_error_axis = axes[2]
    throttle_axis = axes[3]
    angle_axis = axes[4]
    reward_axis = axes[5]

    altitude_axis.plot(episodes,
                       max_altitudes,
                       color="#1f77b4",
                       linewidth=1.6,
                       marker="o",
                       markersize=2.8,
                       label="max altitude")
    crashed_mask = np.array([bool(record.get("terminated", False))
                             for record in records],
                            dtype=bool)
    if crashed_mask.any():
        altitude_axis.scatter(episodes[crashed_mask],
                              max_altitudes[crashed_mask],
                              color="#d62728",
                              edgecolors="#ffffff",
                              linewidths=0.5,
                              s=36,
                              zorder=4,
                              label="crash")

    altitude_error_axis.plot(episodes,
                             final_altitude_errors,
                             color="#d62728",
                             linewidth=1.4,
                             marker="o",
                             markersize=2.8,
                             label="final altitude error")
    tangential_velocity_error_axis.plot(
        episodes,
        final_tangential_velocity_errors,
        color="#17becf",
        linewidth=1.4,
        marker="o",
        markersize=2.8,
        label="final tangential velocity error",
    )
    throttle_axis.plot(episodes,
                       average_throttles,
                       color="#2ca02c",
                       linewidth=1.4,
                       label="average throttle")
    throttle_axis.plot(episodes,
                       final_throttles,
                       color="#ff7f0e",
                       linewidth=1.1,
                       alpha=0.7,
                       label="final throttle")
    angle_axis.plot(episodes,
                    average_angles,
                    color="#9467bd",
                    linewidth=1.4,
                    label="average angle")
    angle_axis.plot(episodes,
                    final_angles,
                    color="#8c564b",
                    linewidth=1.1,
                    alpha=0.7,
                    label="final angle")
    angle_axis.axhline(90.0,
                       color="#444444",
                       linewidth=1.0,
                       linestyle="--",
                       alpha=0.65,
                       label="vertical reference")
    reward_axis.plot(episodes,
                     total_rewards,
                     color="#111111",
                     linewidth=1.4,
                     marker="o",
                     markersize=2.8,
                     label="total reward")
    if crashed_mask.any():
        reward_axis.scatter(episodes[crashed_mask],
                            total_rewards[crashed_mask],
                            color="#d62728",
                            edgecolors="#ffffff",
                            linewidths=0.5,
                            s=36,
                            zorder=4,
                            label="crash")

    epoch_boundaries = _epoch_boundaries(episodes, epochs)
    for episode, epoch in epoch_boundaries:
        for axis in axes:
            axis.axvline(episode,
                         color="#444444",
                         alpha=0.25,
                         linewidth=1.0,
                         linestyle="--")
        altitude_axis.text(episode,
                           altitude_axis.get_ylim()[1],
                           f"epoch {epoch}",
                           rotation=90,
                           va="top",
                           ha="right",
                           fontsize=8,
                           color="#444444")

    altitude_axis.set_title(title)
    altitude_axis.set_ylabel("Max altitude reached (m)")
    altitude_axis.grid(True, alpha=0.25)
    altitude_axis.legend(loc="upper right")

    altitude_error_axis.set_ylabel("Final altitude error (m)")
    altitude_error_axis.grid(True, alpha=0.25)
    altitude_error_axis.legend(loc="upper right")

    tangential_velocity_error_axis.set_ylabel("Final tangential velocity error (m/s)")
    tangential_velocity_error_axis.grid(True, alpha=0.25)
    tangential_velocity_error_axis.legend(loc="upper right")

    throttle_axis.set_ylabel("Throttle")
    throttle_axis.set_ylim(-0.05, 1.05)
    throttle_axis.grid(True, alpha=0.25)
    throttle_axis.legend(loc="upper right")

    angle_axis.set_ylabel("Angle (deg)")
    angle_axis.grid(True, alpha=0.25)
    angle_axis.legend(loc="upper right")

    reward_axis.set_xlabel("Completed training episode")
    reward_axis.set_ylabel("Total reward")
    reward_axis.grid(True, alpha=0.25)
    reward_axis.legend(loc="upper right")

    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_best_episode_trajectory(trace: list[dict[str, Any]],
                                 output_path: str | Path,
                                 title: str) -> None:
    """Plot trajectory and controls for the best orbit-closeness episode."""

    if not trace:
        return

    steps = np.arange(1, len(trace) + 1, dtype=float)
    positions = np.array([_trace_position(entry) for entry in trace],
                         dtype=float)
    altitudes = np.array([_trace_info_float(entry, "altitude")
                          for entry in trace],
                         dtype=float)
    tangential_velocities = np.array([
        _trace_info_float(entry, "tangential_velocity")
        for entry in trace
    ], dtype=float)
    rewards = np.array([float(entry.get("reward", 0.0)) for entry in trace],
                       dtype=float)
    throttles = np.array([_trace_action(entry, index=0, default=0.0)
                          for entry in trace],
                         dtype=float)
    angles = np.array([_trace_action(entry, index=1, default=90.0)
                       for entry in trace],
                      dtype=float)

    final_info = trace[-1]["info"]
    target_altitude = float(final_info.get("target_altitude", 0.0))
    target_velocity = float(final_info.get("target_orbital_velocity", 0.0))
    target_radius = _trace_target_radius(trace[-1])
    planet_radius = max(target_radius - target_altitude, 0.0)
    trajectory_ranges = _trajectory_ranges(positions, planet_radius)

    fig, axes = plt.subplots(nrows=5,
                             ncols=1,
                             figsize=(11, 15))
    trajectory_axis = axes[0]
    altitude_axis = axes[1]
    velocity_axis = axes[2]
    control_axis = axes[3]
    reward_axis = axes[4]

    trajectory_axis.plot(trajectory_ranges,
                         altitudes,
                         color="#222222",
                         linewidth=1.5,
                         label="trajectory")
    trajectory_axis.scatter(trajectory_ranges[-1],
                            altitudes[-1],
                            color="#e31a1c",
                            s=28,
                            zorder=3,
                            label="final")
    trajectory_axis.axhline(target_altitude,
                            color="#d95f02",
                            linestyle="--",
                            linewidth=1.1,
                            alpha=0.75,
                            label="target altitude")
    trajectory_axis.axhline(0.0,
                            color="#444444",
                            linestyle=":",
                            linewidth=1.0,
                            alpha=0.6,
                            label="ground")
    trajectory_axis.set_title(title)
    trajectory_axis.set_xlabel("Range from launch point (m)")
    trajectory_axis.set_ylabel("Altitude (m)")
    trajectory_axis.grid(True, alpha=0.25)
    trajectory_axis.legend(loc="upper right")

    altitude_axis.plot(steps,
                       altitudes,
                       color="#1f77b4",
                       linewidth=1.4,
                       label="altitude")
    altitude_axis.axhline(target_altitude,
                          color="#444444",
                          linestyle="--",
                          linewidth=1.0,
                          alpha=0.6,
                          label="target altitude")
    altitude_axis.set_ylabel("Altitude (m)")
    altitude_axis.grid(True, alpha=0.25)
    altitude_axis.legend(loc="upper right")

    velocity_axis.plot(steps,
                       tangential_velocities,
                       color="#17becf",
                       linewidth=1.4,
                       label="tangential velocity")
    velocity_axis.axhline(target_velocity,
                          color="#444444",
                          linestyle="--",
                          linewidth=1.0,
                          alpha=0.6,
                          label="target velocity")
    velocity_axis.set_ylabel("Tangential velocity (m/s)")
    velocity_axis.grid(True, alpha=0.25)
    velocity_axis.legend(loc="upper right")

    control_axis.plot(steps,
                      throttles,
                      color="#2ca02c",
                      linewidth=1.4,
                      label="throttle")
    control_axis_right = control_axis.twinx()
    control_axis_right.plot(steps,
                            angles,
                            color="#9467bd",
                            linewidth=1.2,
                            label="angle")
    control_axis_right.axhline(90.0,
                               color="#444444",
                               linestyle="--",
                               linewidth=1.0,
                               alpha=0.5,
                               label="vertical reference")
    control_axis.set_ylabel("Throttle")
    control_axis_right.set_ylabel("Angle (deg)")
    control_axis.set_ylim(-0.05, 1.05)
    control_axis.grid(True, alpha=0.25)
    _combined_legend(control_axis, control_axis_right)

    reward_axis.plot(steps,
                     rewards,
                     color="#111111",
                     linewidth=1.3,
                     label="step reward")
    reward_axis.plot(steps,
                     np.cumsum(rewards),
                     color="#ff7f0e",
                     linewidth=1.2,
                     label="cumulative reward")
    reward_axis.set_xlabel("Step")
    reward_axis.set_ylabel("Reward")
    reward_axis.grid(True, alpha=0.25)
    reward_axis.legend(loc="upper right")

    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _info_altitude(info: dict[str, Any]) -> float:
    altitude = float(info.get("altitude", 0.0))

    return altitude


def _info_altitude_error(info: dict[str, Any]) -> float:
    altitude_error = float(info.get("altitude_error", 0.0))

    return altitude_error


def _info_tangential_velocity_error(info: dict[str, Any]) -> float:
    tangential_velocity_error = float(info.get("tangential_velocity_error", 0.0))

    return tangential_velocity_error


def _info_target_altitude(info: dict[str, Any]) -> float:
    target_altitude = float(info.get("target_altitude", 0.0))

    return target_altitude


def _info_target_orbital_velocity(info: dict[str, Any]) -> float:
    target_orbital_velocity = float(info.get("target_orbital_velocity", 0.0))

    return target_orbital_velocity


def _info_throttle(info: dict[str, Any]) -> float:
    applied_action = info.get("applied_action")
    if applied_action is None:
        return 0.0

    throttle = float(np.asarray(applied_action, dtype=float).reshape(-1)[0])

    return throttle


def _info_angle_degrees(info: dict[str, Any]) -> float:
    applied_action = info.get("applied_action")
    if applied_action is None:
        return 90.0

    angle_degrees = float(np.asarray(applied_action, dtype=float).reshape(-1)[1])

    return angle_degrees


def _epoch_boundaries(episodes: np.ndarray,
                      epochs: np.ndarray) -> list[tuple[float, int]]:
    boundaries: list[tuple[float, int]] = []
    previous_epoch = int(epochs[0])

    for episode, epoch in zip(episodes[1:], epochs[1:]):
        current_epoch = int(epoch)
        if current_epoch != previous_epoch:
            boundaries.append((float(episode), current_epoch))
            previous_epoch = current_epoch

    return boundaries


def _orbit_closeness_score(record: EpisodeAltitudeRecord) -> float:
    altitude_reference = max(float(record.get("target_altitude", 0.0)), 1.0)
    velocity_reference = max(float(record.get("target_orbital_velocity", 0.0)), 1.0)
    altitude_score = float(record.get("final_altitude_error", 0.0)) / altitude_reference
    velocity_score = float(record.get("final_tangential_velocity_error", 0.0)) / velocity_reference

    return altitude_score + velocity_score


def _trace_entry(observation: Any,
                 action: Any,
                 reward: float,
                 terminated: bool,
                 truncated: bool,
                 info: dict[str, Any]) -> dict[str, Any]:
    applied_action = np.asarray(info.get("applied_action", action),
                                dtype=float).copy()
    entry = {"observation": observation,
             "action": np.asarray(action, dtype=float).copy(),
             "applied_action": applied_action,
             "reward": float(reward),
             "terminated": bool(terminated),
             "truncated": bool(truncated),
             "info": dict(info)}

    return entry


def _trace_position(entry: dict[str, Any]) -> np.ndarray:
    observation = entry["observation"]
    state = observation["state"]
    position = np.asarray(state[:2], dtype=float)

    return position


def _trace_target_radius(entry: dict[str, Any]) -> float:
    observation = entry["observation"]
    goal = np.asarray(observation["goal"], dtype=float)
    target_radius = float(goal[0])

    return target_radius


def _trace_info_float(entry: dict[str, Any],
                      key: str) -> float:
    info = entry["info"]
    value = float(info.get(key, 0.0))

    return value


def _trace_action(entry: dict[str, Any],
                  *,
                  index: int,
                  default: float) -> float:
    action = np.asarray(entry.get("applied_action", []), dtype=float).reshape(-1)
    if action.size <= index:
        return default

    value = float(action[index])

    return value


def _trajectory_ranges(positions: np.ndarray,
                       planet_radius: float) -> np.ndarray:
    if planet_radius <= 0.0:
        ranges = positions[:, 0] - positions[0, 0]

        return ranges

    radii = np.linalg.norm(positions, axis=1)
    safe_radii = np.maximum(radii, 1e-9)
    radial_units = positions / safe_radii[:, np.newaxis]
    launch_unit = radial_units[0]
    cross_terms = (launch_unit[0] * radial_units[:, 1]
                   - launch_unit[1] * radial_units[:, 0])
    dot_terms = radial_units @ launch_unit
    angular_offsets = np.arctan2(cross_terms, dot_terms)
    ranges = planet_radius * angular_offsets

    return ranges


def _combined_legend(left_axis,
                     right_axis) -> None:
    left_handles, left_labels = left_axis.get_legend_handles_labels()
    right_handles, right_labels = right_axis.get_legend_handles_labels()
    left_axis.legend(left_handles + right_handles,
                     left_labels + right_labels,
                     loc="upper right")
