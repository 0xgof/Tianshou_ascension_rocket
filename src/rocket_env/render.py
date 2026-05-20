"""Matplotlib display helpers for rocket rollout inspection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

from .env import RocketAscentEnv
from .plots import plot_trajectory
from .rollouts import PolicyFn, run_episode


@dataclass(frozen=True)
class Viewport:
    width: int = 900
    height: int = 900
    meters_per_pixel: float = 10_000.0
    center_x: float = 0.0
    center_y: float = 0.0


def world_to_screen(position: np.ndarray, viewport: Viewport) -> Tuple[float, float]:
    vector = np.asarray(position, dtype=float)
    x = viewport.width / 2.0 + (vector[0] - viewport.center_x) / viewport.meters_per_pixel
    y = viewport.height / 2.0 - (vector[1] - viewport.center_y) / viewport.meters_per_pixel
    return float(x), float(y)


def collect_display_trace(
    env: RocketAscentEnv,
    policy: PolicyFn,
    max_steps: int = 500,
    seed: int = 0,
):
    trace, info = run_episode(env, policy, max_steps=max_steps, seed=seed)
    return trace, info


def save_rollout_plot(
    env: RocketAscentEnv,
    policy: PolicyFn,
    output_path: str,
    title: str,
    max_steps: int = 500,
    seed: int = 0,
):
    trace, info = collect_display_trace(env, policy, max_steps=max_steps, seed=seed)
    fig = plot_trajectory(trace, env, title=title, output_path=output_path)
    return fig, trace, info


def print_rollout_summary(trace: Iterable[Dict[str, object]], final_info: Dict[str, object]):
    print("steps:", len(list(trace)))
    print("altitude:", final_info.get("altitude"))
    print("speed:", final_info.get("speed"))
    print("fuel_remaining:", final_info.get("fuel_remaining"))
    print("altitude_error:", final_info.get("altitude_error"))
    print("vertical_speed_error:", final_info.get("vertical_speed_error"))
    print("tangential_velocity_error:", final_info.get("tangential_velocity_error"))
    print("crashed:", final_info.get("crashed"))
    print("success:", final_info.get("success"))
