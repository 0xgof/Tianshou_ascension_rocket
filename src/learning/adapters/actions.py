"""Action conversion helpers between model space and environment space."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Union

import numpy as np
import torch

ActionInput = Union[Iterable[float], np.ndarray, torch.Tensor]


@dataclass(frozen=True)
class ActionPair:
    """Model action plus physical environment action.

    PPO note: log_prob must describe model_action before environment clipping.
    """

    model_action: np.ndarray
    env_action: np.ndarray


def _as_numpy(action: ActionInput) -> np.ndarray:
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()

    action_array = np.asarray(action, dtype=np.float32)

    return action_array


def decode_physical_action(model_action: ActionInput) -> ActionPair:
    """Clip a two-dimensional model action into env command-rate limits."""

    raw_action = _as_numpy(model_action).reshape(-1)

    if raw_action.shape != (2,):
        raise ValueError("Expected action with shape (2,).")

    env_action = np.array([np.clip(raw_action[0], -1.0, 1.0),
                           np.clip(raw_action[1], -1.0, 1.0)],
                          dtype=np.float32)

    action_pair = ActionPair(model_action=raw_action.astype(np.float32),
                             env_action=env_action)

    return action_pair
