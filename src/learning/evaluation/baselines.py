"""Baseline policy adapters for evaluation."""

from __future__ import annotations

import numpy as np


def scripted_baseline_policy(_observation) -> np.ndarray:
    return np.array([1.0, 90.0], dtype=np.float32)

