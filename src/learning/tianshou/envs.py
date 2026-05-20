"""Environment wrappers used by Tianshou training."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from learning.adapters import DEFAULT_OBSERVATION_SIZE, encode_observation
from learning.tianshou.episode_monitor import (EpisodeAltitudeMonitor,
                                               EpisodeAltitudeRecorder)
from learning.tianshou.training_display import (TrainingDisplayConfig,
                                                TrainingDisplayWrapper)
from rocket_env.env import EnvConfig, RocketAscentEnv
from rocket_env.observations import Observation


class EncodedRocketObservationWrapper(gym.ObservationWrapper):
    """Expose rocket observations as the flat vector consumed by the NN."""

    def __init__(self,
                 env: gym.Env) -> None:
        super().__init__(env)

        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(DEFAULT_OBSERVATION_SIZE,),
                                            dtype=np.float32)

    def observation(self,
                    observation: Observation) -> np.ndarray:
        encoded_observation = encode_observation(observation)

        return encoded_observation


def make_encoded_env(config: EnvConfig | None = None,
                     training_display: TrainingDisplayConfig | None = None,
                     altitude_recorder: EpisodeAltitudeRecorder | None = None,
                     env_index: int = 0
                     ) -> EncodedRocketObservationWrapper:
    """Create a playable rocket environment wrapped for vector NN input."""

    env_config = config or EnvConfig()
    env = RocketAscentEnv(env_config)
    if training_display is not None and training_display.enabled:
        display_config = training_display.to_display_config(max_steps=env_config.max_steps)
        env = TrainingDisplayWrapper(env,
                                     config=training_display,
                                     display_config=display_config)

    if altitude_recorder is not None:
        env = EpisodeAltitudeMonitor(env,
                                     recorder=altitude_recorder,
                                     env_index=env_index)

    wrapped_env = EncodedRocketObservationWrapper(env)

    return wrapped_env
