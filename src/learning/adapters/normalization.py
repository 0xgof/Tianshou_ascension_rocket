"""Small normalization helpers for model inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def identity_normalizer(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


@dataclass(frozen=True)
class FixedNormalizer:
    """Apply fixed feature scales.

    Learning choice: change scales here when testing observation design.
    """

    scale: np.ndarray

    def __call__(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        scale = np.asarray(self.scale, dtype=np.float32)
        if values.shape != scale.shape:
            raise ValueError("Normalizer scale must match values shape.")
        safe_scale = np.where(scale == 0.0, 1.0, scale)
        return values / safe_scale

