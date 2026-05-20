"""Keyboard-to-action state for manual rocket control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rocket_env.physics import normalize_angle_degrees, shortest_angle_delta_degrees


@dataclass
class ManualControlState:
    throttle_command: float = 0.0
    angle_command: float = 90.0
    throttle_step: float = 0.01
    angle_step: float = 1.0
    paused: bool = False
    reset_requested: bool = False
    exit_requested: bool = False

    def action(
        self,
        applied_action=None,
        throttle_change_limit: float | None = None,
        angle_change_limit: float | None = None,
    ) -> np.ndarray:
        if applied_action is None:
            applied_action = np.array([0.0, 90.0], dtype=float)
        applied = np.asarray(applied_action, dtype=float)
        throttle_limit = throttle_change_limit or self.throttle_step
        angle_limit = angle_change_limit or self.angle_step
        throttle_rate = (self.throttle_command - applied[0]) / throttle_limit
        angle_rate = shortest_angle_delta_degrees(self.angle_command, applied[1]) / angle_limit
        return np.array(
            [
                float(np.clip(throttle_rate, -1.0, 1.0)),
                float(np.clip(angle_rate, -1.0, 1.0)),
            ],
            dtype=np.float32,
        )

    def apply_key(self, key: str) -> None:
        normalized = key.lower()
        if normalized in {"up", "w"}:
            self.throttle_command += self.throttle_step
        elif normalized in {"down", "s"}:
            self.throttle_command -= self.throttle_step
        elif normalized in {"left", "a", "<"}:
            self.angle_command += self.angle_step
        elif normalized in {"right", "d", ">"}:
            self.angle_command -= self.angle_step
        elif normalized == " ":
            self.throttle_command = 0.0
        elif normalized == "r":
            self.reset_requested = True
        elif normalized == "p":
            self.paused = not self.paused
        elif normalized == "escape":
            self.exit_requested = True

        self.throttle_command = float(np.clip(self.throttle_command, 0.0, 1.0))
        self.angle_command = normalize_angle_degrees(self.angle_command)

    def consume_reset(self) -> bool:
        requested = self.reset_requested
        self.reset_requested = False
        return requested

    def reset_commands(self, applied_action=None) -> None:
        if applied_action is None:
            applied_action = np.array([0.0, 90.0], dtype=float)
        applied = np.asarray(applied_action, dtype=float)
        self.throttle_command = float(np.clip(applied[0], 0.0, 1.0))
        self.angle_command = normalize_angle_degrees(applied[1])
        self.reset_requested = False


def action_from_keys(keys) -> np.ndarray:
    control = ManualControlState()
    for key in keys:
        control.apply_key(key)
    return control.action()
