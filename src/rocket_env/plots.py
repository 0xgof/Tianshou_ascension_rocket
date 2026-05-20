"""Static plotting helpers for rocket rollout traces."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .env import RocketAscentEnv


def trace_positions(trace: Iterable[Dict[str, object]]) -> np.ndarray:
    positions: List[np.ndarray] = []
    for step in trace:
        state = step["observation"]["state"]
        positions.append(np.asarray(state[:2], dtype=float))
    if not positions:
        return np.empty((0, 2), dtype=float)
    return np.vstack(positions)


def plot_trajectory(
    trace: Iterable[Dict[str, object]],
    env: RocketAscentEnv,
    title: str = "Rocket trajectory",
    output_path: Optional[str] = None,
):
    positions = trace_positions(trace)
    if len(positions) == 0:
        raise ValueError("Cannot plot an empty trace.")

    fig, axis = plt.subplots(figsize=(7, 7))
    planet = plt.Circle((0.0, 0.0), env.config.physics.planet_radius, color="#3a6ea5")
    target_orbit = plt.Circle(
        (0.0, 0.0),
        float(env.goal[0]),
        fill=False,
        linestyle="--",
        color="#d95f02",
        linewidth=1.5,
    )
    axis.add_patch(planet)
    axis.add_patch(target_orbit)
    axis.plot(positions[:, 0], positions[:, 1], color="#222222", linewidth=2.0)
    axis.scatter(positions[-1, 0], positions[-1, 1], color="#e31a1c", s=30)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(title)
    axis.set_xlabel("x position (m)")
    axis.set_ylabel("y position (m)")

    margin = max(env.config.target_altitude * 1.4, 100_000.0)
    limit = float(env.goal[0] + margin)
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.grid(True, alpha=0.25)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=140, bbox_inches="tight")
    return fig
