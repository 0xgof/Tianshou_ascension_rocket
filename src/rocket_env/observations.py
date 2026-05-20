"""Structured observation contract for the rocket environment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Observation = dict[str, np.ndarray]


@dataclass(frozen=True)
class RocketGoal:
    """Named target fields with Gym-compatible vector conversion."""

    target_radius: float
    target_orbital_velocity: float
    payload_mass: float

    @classmethod
    def from_array(cls,
                   goal: np.ndarray) -> "RocketGoal":
        goal = np.asarray(goal, dtype=np.float32).reshape(-1)

        try:
            target_radius, target_orbital_velocity, payload_mass = goal
        except ValueError as error:
            raise ValueError("Expected goal vector with target observation fields.") from error

        rocket_goal = cls(target_radius=float(target_radius),
                          target_orbital_velocity=float(target_orbital_velocity),
                          payload_mass=float(payload_mass))

        return rocket_goal

    def to_array(self) -> np.ndarray:
        goal = np.array([self.target_radius,
                         self.target_orbital_velocity,
                         self.payload_mass],
                        dtype=np.float32)

        return goal


@dataclass(frozen=True)
class RocketObservation:
    """Named observation fields with Gym-compatible vector conversion."""

    position: np.ndarray
    velocity: np.ndarray
    mass: float
    fuel_remaining: float
    time: float
    current_throttle_command: float
    current_angle_degrees: float
    initial_fuel: float
    target_radius: float
    target_orbital_velocity: float
    payload_mass: float

    @classmethod
    def from_mapping(cls,
                     observation: Observation) -> "RocketObservation":
        state = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
        goal = np.asarray(observation["goal"], dtype=np.float32).reshape(-1)

        try:
            (position_x,
             position_y,
             velocity_x,
             velocity_y,
             mass,
             fuel_remaining,
             time,
             current_throttle_command,
             current_angle_degrees,
             initial_fuel) = state
        except ValueError as error:
            raise ValueError("Expected state vector with rocket observation fields.") from error

        rocket_goal = RocketGoal.from_array(goal)

        position = np.array([position_x, position_y],
                            dtype=float)
        velocity = np.array([velocity_x, velocity_y],
                            dtype=float)

        rocket_observation = cls(position=position,
                                 velocity=velocity,
                                 mass=float(mass),
                                 fuel_remaining=float(fuel_remaining),
                                 time=float(time),
                                 current_throttle_command=float(current_throttle_command),
                                 current_angle_degrees=float(current_angle_degrees),
                                 initial_fuel=float(initial_fuel),
                                 target_radius=rocket_goal.target_radius,
                                 target_orbital_velocity=rocket_goal.target_orbital_velocity,
                                 payload_mass=rocket_goal.payload_mass)

        return rocket_observation

    def state_array(self) -> np.ndarray:
        state = np.array([self.position[0],
                          self.position[1],
                          self.velocity[0],
                          self.velocity[1],
                          self.mass,
                          self.fuel_remaining,
                          self.time,
                          self.current_throttle_command,
                          self.current_angle_degrees,
                          self.initial_fuel],
                         dtype=np.float32)

        return state

    def goal_array(self) -> np.ndarray:
        rocket_goal = RocketGoal(target_radius=self.target_radius,
                                 target_orbital_velocity=self.target_orbital_velocity,
                                 payload_mass=self.payload_mass)
        goal = rocket_goal.to_array()

        return goal

    def to_mapping(self) -> Observation:
        observation = {"state": self.state_array(),
                       "goal": self.goal_array()}

        return observation
