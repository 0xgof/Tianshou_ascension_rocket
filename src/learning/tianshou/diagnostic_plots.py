"""Plots for model and PPO training diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DiagnosticSeries = tuple[np.ndarray, np.ndarray]


def plot_training_diagnostics(diagnostics: dict[str, Any],
                              output_path: str | Path,
                              title: str) -> None:
    """Plot reward, length, entropy, and PPO loss metrics."""

    records = diagnostics.get("records", [])
    if not records:
        return

    train_reward = _series(records, "train/reward")
    test_reward = _series(records, "test/reward")
    train_length = _series(records, "train/length")
    test_length = _series(records, "test/length")
    total_loss = _series(records, "update/loss")
    policy_loss = _series(records, "update/loss/clip")
    value_loss = _series(records, "update/loss/vf")
    entropy = _series(records, "update/loss/ent")

    fig, axes = plt.subplots(nrows=2,
                             ncols=2,
                             figsize=(13, 8))
    reward_axis = axes[0, 0]
    length_axis = axes[0, 1]
    loss_axis = axes[1, 0]
    entropy_axis = axes[1, 1]

    _plot_optional(reward_axis,
                   train_reward,
                   color="#1f77b4",
                   label="train reward")
    _plot_optional(reward_axis,
                   test_reward,
                   color="#2ca02c",
                   label="test reward")
    reward_axis.set_title("Episode Reward")
    reward_axis.set_xlabel("env step")
    reward_axis.set_ylabel("reward")
    reward_axis.grid(True, alpha=0.25)
    reward_axis.legend(loc="best")

    _plot_optional(length_axis,
                   train_length,
                   color="#9467bd",
                   label="train length")
    _plot_optional(length_axis,
                   test_length,
                   color="#8c564b",
                   label="test length")
    length_axis.set_title("Episode Length")
    length_axis.set_xlabel("env step")
    length_axis.set_ylabel("steps")
    length_axis.grid(True, alpha=0.25)
    length_axis.legend(loc="best")

    _plot_optional(loss_axis,
                   total_loss,
                   color="#111111",
                   label="total loss")
    _plot_optional(loss_axis,
                   policy_loss,
                   color="#ff7f0e",
                   label="policy clip loss")
    _plot_optional(loss_axis,
                   value_loss,
                   color="#d62728",
                   label="value loss")
    loss_axis.set_title("PPO Losses")
    loss_axis.set_xlabel("gradient step")
    loss_axis.set_ylabel("loss")
    loss_axis.grid(True, alpha=0.25)
    loss_axis.legend(loc="best")

    _plot_optional(entropy_axis,
                   entropy,
                   color="#17becf",
                   label="entropy")
    entropy_axis.set_title("Policy Entropy")
    entropy_axis.set_xlabel("gradient step")
    entropy_axis.set_ylabel("entropy")
    entropy_axis.grid(True, alpha=0.25)
    entropy_axis.legend(loc="best")

    fig.suptitle(title)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _series(records: list[dict[str, Any]],
            key: str) -> DiagnosticSeries | None:
    steps = []
    values = []

    for record in records:
        data = record.get("data", {})
        if key not in data:
            continue

        value = data[key]
        if not isinstance(value, (int, float)):
            continue

        steps.append(float(record["step"]))
        values.append(float(value))

    if not steps:
        return None

    series = np.array(steps, dtype=float), np.array(values, dtype=float)

    return series


def _plot_optional(axis,
                   series: DiagnosticSeries | None,
                   *,
                   color: str,
                   label: str) -> None:
    if series is None:
        return

    steps, values = series
    axis.plot(steps,
              values,
              color=color,
              linewidth=1.6,
              marker="o",
              markersize=2.4,
              label=label)
