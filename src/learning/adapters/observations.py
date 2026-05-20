"""Observation conversion helpers for Gymnasium rocket observations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Union

import numpy as np
import torch

from rocket_env.observations import Observation, RocketObservation
from rocket_env.physics import (circular_orbit_velocity, decompose_velocity,
                                vector_norm)
from settings import load_settings

Normalizer = Callable[[np.ndarray], np.ndarray]
Device = Union[torch.device, str, None]

DEFAULT_PROJECT_SETTINGS = load_settings()
DEFAULT_PHYSICS_CONFIG = DEFAULT_PROJECT_SETTINGS.to_physics_config()
DEFAULT_OBSERVATION_SCALING = DEFAULT_PROJECT_SETTINGS.learning.observation_scaling
DEFAULT_TARGET_ALTITUDE = DEFAULT_OBSERVATION_SCALING.reference_target_altitude
DEFAULT_INITIAL_FUEL = DEFAULT_OBSERVATION_SCALING.reference_initial_fuel
DEFAULT_TARGET_ORBITAL_VELOCITY = circular_orbit_velocity(
    DEFAULT_PHYSICS_CONFIG.planet_radius + DEFAULT_TARGET_ALTITUDE,
    DEFAULT_PHYSICS_CONFIG,
)


class FractionalFeature(str, Enum):
    ALTITUDE_SCALED = "altitude_scaled"
    VERTICAL_SPEED_SCALED = "vertical_speed_scaled"
    TANGENTIAL_VELOCITY_SCALED = "tangential_velocity_scaled"
    FUEL_FRACTION = "fuel_fraction"
    INITIAL_FUEL_SCALED = "initial_fuel_scaled"
    TARGET_ALTITUDE_SCALED = "target_altitude_scaled"
    TARGET_ORBITAL_VELOCITY_SCALED = "target_orbital_velocity_scaled"
    ALTITUDE_ERROR_SCALED = "altitude_error_scaled"
    VERTICAL_SPEED_ERROR_SCALED = "vertical_speed_error_scaled"
    TANGENTIAL_VELOCITY_ERROR_SCALED = "tangential_velocity_error_scaled"


class EncodedFeature(str, Enum):
    ALTITUDE_SCALED = "altitude_scaled"
    VERTICAL_SPEED_SCALED = "vertical_speed_scaled"
    TANGENTIAL_VELOCITY_SCALED = "tangential_velocity_scaled"
    FUEL_FRACTION = "fuel_fraction"
    INITIAL_FUEL_SCALED = "initial_fuel_scaled"
    CURRENT_THROTTLE_COMMAND = "current_throttle_command"
    CURRENT_ANGLE_SIN = "current_angle_sin"
    CURRENT_ANGLE_COS = "current_angle_cos"
    TARGET_ALTITUDE_SCALED = "target_altitude_scaled"
    TARGET_ORBITAL_VELOCITY_SCALED = "target_orbital_velocity_scaled"
    ALTITUDE_ERROR_SCALED = "altitude_error_scaled"
    VERTICAL_SPEED_ERROR_SCALED = "vertical_speed_error_scaled"
    TANGENTIAL_VELOCITY_ERROR_SCALED = "tangential_velocity_error_scaled"


FRACTIONAL_FEATURE_NAMES = tuple(feature.value for feature in FractionalFeature)
FEATURE_NAMES = tuple(feature.value for feature in EncodedFeature)


DEFAULT_FRACTIONAL_OBSERVATION_SIZE = len(FRACTIONAL_FEATURE_NAMES)
DEFAULT_OBSERVATION_SIZE = len(FEATURE_NAMES)


@dataclass(frozen=True)
class ScaledObservationFeatures:
    """Named learning features before conversion to neural-network vectors."""

    altitude_scaled: float
    vertical_speed_scaled: float
    tangential_velocity_scaled: float
    fuel_fraction: float
    initial_fuel_scaled: float
    target_altitude_scaled: float
    target_orbital_velocity_scaled: float
    altitude_error_scaled: float
    vertical_speed_error_scaled: float
    tangential_velocity_error_scaled: float

    def fractional_feature_values(self) -> dict[FractionalFeature, float]:
        feature_values = {
            FractionalFeature.ALTITUDE_SCALED: self.altitude_scaled,
            FractionalFeature.VERTICAL_SPEED_SCALED: self.vertical_speed_scaled,
            FractionalFeature.TANGENTIAL_VELOCITY_SCALED: self.tangential_velocity_scaled,
            FractionalFeature.FUEL_FRACTION: self.fuel_fraction,
            FractionalFeature.INITIAL_FUEL_SCALED: self.initial_fuel_scaled,
            FractionalFeature.TARGET_ALTITUDE_SCALED: self.target_altitude_scaled,
            FractionalFeature.TARGET_ORBITAL_VELOCITY_SCALED: (
                self.target_orbital_velocity_scaled
            ),
            FractionalFeature.ALTITUDE_ERROR_SCALED: self.altitude_error_scaled,
            FractionalFeature.VERTICAL_SPEED_ERROR_SCALED: self.vertical_speed_error_scaled,
            FractionalFeature.TANGENTIAL_VELOCITY_ERROR_SCALED: (
                self.tangential_velocity_error_scaled
            ),
        }

        return feature_values

    def encoded_feature_values(self,
                               current_throttle_command: float,
                               current_angle_degrees: float) -> dict[EncodedFeature, float]:
        angle_radians = np.deg2rad(current_angle_degrees)

        feature_values = {
            EncodedFeature.ALTITUDE_SCALED: self.altitude_scaled,
            EncodedFeature.VERTICAL_SPEED_SCALED: self.vertical_speed_scaled,
            EncodedFeature.TANGENTIAL_VELOCITY_SCALED: self.tangential_velocity_scaled,
            EncodedFeature.FUEL_FRACTION: self.fuel_fraction,
            EncodedFeature.INITIAL_FUEL_SCALED: self.initial_fuel_scaled,
            EncodedFeature.CURRENT_THROTTLE_COMMAND: current_throttle_command,
            EncodedFeature.CURRENT_ANGLE_SIN: np.sin(angle_radians),
            EncodedFeature.CURRENT_ANGLE_COS: np.cos(angle_radians),
            EncodedFeature.TARGET_ALTITUDE_SCALED: self.target_altitude_scaled,
            EncodedFeature.TARGET_ORBITAL_VELOCITY_SCALED: (
                self.target_orbital_velocity_scaled
            ),
            EncodedFeature.ALTITUDE_ERROR_SCALED: self.altitude_error_scaled,
            EncodedFeature.VERTICAL_SPEED_ERROR_SCALED: self.vertical_speed_error_scaled,
            EncodedFeature.TANGENTIAL_VELOCITY_ERROR_SCALED: (
                self.tangential_velocity_error_scaled
            ),
        }

        return feature_values

    def to_fractional_array(self) -> np.ndarray:
        feature_values = self.fractional_feature_values()
        fractional_features = np.array([feature_values[feature]
                                        for feature in FractionalFeature],
                                       dtype=np.float32)

        return fractional_features

    def to_encoded_array(self,
                         current_throttle_command: float,
                         current_angle_degrees: float) -> np.ndarray:
        feature_values = self.encoded_feature_values(current_throttle_command,
                                                     current_angle_degrees)
        encoded_features = np.array([feature_values[feature]
                                     for feature in EncodedFeature],
                                    dtype=np.float32)

        return encoded_features


def _safe_ratio(value: float,
                scale: float) -> float:
    """Return a stable ratio when the scale can be zero in edge-case tests."""

    safe_ratio = value / max(abs(scale), 1e-9)

    return safe_ratio


def fractionalize_observation(observation: Observation) -> np.ndarray:
    """Convert a raw environment observation into scaled physical features."""

    rocket_observation = RocketObservation.from_mapping(observation)
    scaled_features = _build_scaled_features(rocket_observation)
    fractionalized_observation = scaled_features.to_fractional_array()

    return fractionalized_observation


def _build_scaled_features(
        rocket_observation: RocketObservation) -> ScaledObservationFeatures:
    position = rocket_observation.position
    velocity = rocket_observation.velocity

    radius = vector_norm(position)
    altitude = radius - DEFAULT_PHYSICS_CONFIG.planet_radius
    vertical_speed, tangential_velocity = decompose_velocity(position, velocity)

    fuel_remaining = rocket_observation.fuel_remaining
    initial_fuel = rocket_observation.initial_fuel

    target_radius = rocket_observation.target_radius
    target_altitude = target_radius - DEFAULT_PHYSICS_CONFIG.planet_radius
    target_orbital_velocity = rocket_observation.target_orbital_velocity

    fuel_fraction = fuel_remaining / max(initial_fuel, 1e-9)
    fuel_fraction = float(np.clip(fuel_fraction, 0.0, 1.0))

    altitude_scaled = _safe_ratio(altitude, target_altitude)
    vertical_speed_scaled = _safe_ratio(vertical_speed, target_orbital_velocity)
    tangential_velocity_scaled = _safe_ratio(tangential_velocity,
                                             target_orbital_velocity)

    initial_fuel_scaled = _safe_ratio(initial_fuel, DEFAULT_INITIAL_FUEL)
    target_altitude_scaled = _safe_ratio(target_altitude, DEFAULT_TARGET_ALTITUDE)
    target_orbital_velocity_scaled = _safe_ratio(target_orbital_velocity,
                                                 DEFAULT_TARGET_ORBITAL_VELOCITY)

    altitude_error_scaled = abs(_safe_ratio(radius - target_radius, target_altitude))
    vertical_speed_error_scaled = abs(vertical_speed_scaled)
    tangential_velocity_error_scaled = abs(_safe_ratio(tangential_velocity - target_orbital_velocity,
                                                       target_orbital_velocity))

    scaled_features = ScaledObservationFeatures(
        altitude_scaled=altitude_scaled,
        vertical_speed_scaled=vertical_speed_scaled,
        tangential_velocity_scaled=tangential_velocity_scaled,
        fuel_fraction=fuel_fraction,
        initial_fuel_scaled=initial_fuel_scaled,
        target_altitude_scaled=target_altitude_scaled,
        target_orbital_velocity_scaled=target_orbital_velocity_scaled,
        altitude_error_scaled=altitude_error_scaled,
        vertical_speed_error_scaled=vertical_speed_error_scaled,
        tangential_velocity_error_scaled=tangential_velocity_error_scaled,
    )

    return scaled_features


def encode_observation(observation: Observation,
                       normalizer: Normalizer | None = None) -> np.ndarray:
    """Convert an environment observation into a neural-network input vector."""

    rocket_observation = RocketObservation.from_mapping(observation)
    scaled_features = _build_scaled_features(rocket_observation)
    encoded_observation = scaled_features.to_encoded_array(
        rocket_observation.current_throttle_command,
        rocket_observation.current_angle_degrees,
    )

    if normalizer is not None:
        encoded_observation = normalizer(encoded_observation)

    return encoded_observation


def to_tensor(values: np.ndarray,
              device: Device = None) -> torch.Tensor:
    """Convert NumPy values into a float32 PyTorch tensor."""

    target_device = torch.device(device or "cpu")

    nn_input_tensor = torch.as_tensor(values,
                                      dtype=torch.float32,
                                      device=target_device)

    return nn_input_tensor
