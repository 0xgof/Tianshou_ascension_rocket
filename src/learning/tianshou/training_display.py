"""Optional Pygame display wrapper for observing training rollouts."""

from __future__ import annotations

import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np

from rocket_env.live_display import DisplayConfig, PygameFlightApp, trace_entry


@dataclass(frozen=True)
class TrainingDisplayConfig:
    enabled: bool = False
    step_delay_seconds: float = 0.0
    width: int = 1100
    height: int = 820
    fps: int = 30

    def to_display_config(self,
                          max_steps: int) -> DisplayConfig:
        display_config = DisplayConfig(width=self.width,
                                       height=self.height,
                                       fps=self.fps,
                                       max_steps=max_steps)

        return display_config


class TrainingDisplayWrapper(gym.Wrapper):
    """Draw one training environment as Tianshou steps it."""

    def __init__(self,
                 env: gym.Env,
                 config: TrainingDisplayConfig,
                 display_config: DisplayConfig) -> None:
        super().__init__(env)

        self.config = config
        self.display_config = display_config
        self._pygame = None
        self._screen = None
        self._font = None
        self._display_app = None
        self._latest_obs = None
        self._latest_info = None

    def reset(self,
              **kwargs) -> tuple[Any, dict]:
        observation, info = self.env.reset(**kwargs)
        self._latest_obs = observation
        self._latest_info = info
        self._ensure_display(observation, info)
        self._draw(action=np.array([0.0, 0.0], dtype=np.float32),
                   status="TRAINING RESET")

        return observation, info

    def step(self,
             action) -> tuple[Any, float, bool, bool, dict]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._latest_obs = observation
        self._latest_info = info
        self._append_trace(observation,
                           action,
                           reward,
                           terminated,
                           truncated,
                           info)
        status = "TRAINING DONE" if terminated or truncated else "TRAINING"
        self._draw(action=np.asarray(action, dtype=np.float32),
                   status=status)

        if self.config.step_delay_seconds > 0.0:
            time.sleep(self.config.step_delay_seconds)

        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        if self._pygame is not None:
            self._pygame.quit()
            self._pygame = None
            self._screen = None
            self._font = None

        super().close()

    def _ensure_display(self,
                        observation,
                        info) -> None:
        if self._display_app is not None:
            self._display_app.obs = observation
            self._display_app.info = info
            self._display_app.trace.clear()

            return

        import pygame

        pygame.init()
        pygame.display.set_caption("RL Ascension Rocket - Training")

        self._pygame = pygame
        self._screen = pygame.display.set_mode((self.display_config.width,
                                                self.display_config.height))
        self._font = pygame.font.SysFont("consolas", 18)
        self._display_app = PygameFlightApp.__new__(PygameFlightApp)
        self._display_app.display_config = self.display_config
        self._display_app.env = self.env
        self._display_app.obs = observation
        self._display_app.info = info
        self._display_app.initial_position = np.asarray(observation["state"][:2],
                                                        dtype=float)
        self._display_app.trace = []
        self._display_app.control = SimpleNamespace(paused=False)
        self._display_app.done = False
        self._display_app.step_count = 0
        self._display_app.step_accumulator = 0.0
        self._display_app.current_control_action = lambda: np.asarray(
            self._display_app.info.get("applied_action", [0.0, 90.0]),
            dtype=float,
        )

    def _append_trace(self,
                      observation,
                      action,
                      reward: float,
                      terminated: bool,
                      truncated: bool,
                      info: dict) -> None:
        if self._display_app is None:
            return

        entry = trace_entry(observation,
                            np.asarray(action, dtype=np.float32),
                            reward,
                            terminated,
                            truncated,
                            info)
        self._display_app.trace.append(entry)
        self._display_app.obs = observation
        self._display_app.info = info
        self._display_app.done = bool(terminated or truncated)
        self._display_app.step_count += 1

    def _draw(self,
              action: np.ndarray,
              status: str) -> None:
        if self._pygame is None or self._screen is None or self._font is None:
            return

        for event in self._pygame.event.get():
            if event.type == self._pygame.QUIT:
                self.close()
                return

        self._display_app.draw(self._pygame, self._screen, self._font)
        self._pygame.display.flip()
