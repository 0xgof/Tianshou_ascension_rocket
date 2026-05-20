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

    def append(self,
               *,
               env_index: int,
               max_altitude: float,
               final_altitude: float,
               average_throttle: float,
               max_throttle: float,
               final_throttle: float,
               average_angle_degrees: float,
               final_angle_degrees: float,
               steps: int,
               total_reward: float,
               terminated: bool,
               truncated: bool) -> None:
        record = {"episode": len(self.records) + 1,
                  "epoch": int(self.current_epoch),
                  "env_index": int(env_index),
                  "max_altitude": float(max_altitude),
                  "final_altitude": float(final_altitude),
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

        return observation, info

    def step(self,
             action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        current_altitude = _info_altitude(info)
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

        if terminated or truncated:
            average_throttle = self.throttle_sum / max(self.steps, 1)
            average_angle_degrees = self.angle_sum / max(self.steps, 1)
            self.recorder.append(env_index=self.env_index,
                                 max_altitude=self.max_altitude,
                                 final_altitude=current_altitude,
                                 average_throttle=average_throttle,
                                 max_throttle=self.max_throttle,
                                 final_throttle=self.final_throttle,
                                 average_angle_degrees=average_angle_degrees,
                                 final_angle_degrees=self.final_angle_degrees,
                                 steps=self.steps,
                                 total_reward=self.total_reward,
                                 terminated=terminated,
                                 truncated=truncated)

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

    fig, axes = plt.subplots(nrows=4,
                             ncols=1,
                             figsize=(11, 11),
                             sharex=True)
    altitude_axis = axes[0]
    throttle_axis = axes[1]
    angle_axis = axes[2]
    reward_axis = axes[3]

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


def _info_altitude(info: dict[str, Any]) -> float:
    altitude = float(info.get("altitude", 0.0))

    return altitude


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
