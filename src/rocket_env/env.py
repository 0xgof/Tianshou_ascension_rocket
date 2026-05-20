"""Gymnasium-compatible rocket ascent environment."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TypedDict

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rocket_env.physics import (DEFAULT_CONFIG, PhysicsConfig, RocketState,
                                altitude, ambient_pressure, atmosphere_density,
                                circular_orbit_velocity, decompose_velocity,
                                gravity_acceleration, mass_from_fuel,
                                normalize_angle_degrees, radial_unit,
                                sea_level_engine_vacuum_efficiency,
                                shortest_angle_delta_degrees, step_dynamics)
from rocket_env.observations import Observation, RocketGoal, RocketObservation
from rocket_env.reward import DEFAULT_REWARD_NAME, reward_function

Info = dict[str, object]
StepResult = tuple[Observation, float, bool, bool, Info]


GROUND_CONTACT_TOLERANCE_M = 1e-6
LAUNCH_ANGLE_LOCK_ALTITUDE_M = 10.0


class ResetOptions(TypedDict, total=False):
    target_altitude: float
    payload_mass: float
    initial_fuel: float


@dataclass(frozen=True)
class EnvConfig:
    physics: PhysicsConfig = DEFAULT_CONFIG
    dry_mass: float = 110_000.0
    payload_mass: float = 15_000.0
    initial_fuel: float = 600_000.0
    initial_altitude: float = 0.0
    target_altitude: float = 200_000.0
    max_steps: int = 1_000
    altitude_tolerance: float = 5_000.0
    vertical_speed_tolerance: float = 50.0
    tangential_velocity_tolerance: float = 100.0
    throttle_change_per_second: float = 0.5
    angle_change_per_second: float = 5.0
    terminate_on_fuel_empty: bool = False
    reward_name: str = DEFAULT_REWARD_NAME


class RocketAscentEnv(gym.Env):
    """Gymnasium environment for a 2D rocket ascent simulation."""

    metadata = {"render_modes": []}

    def __init__(self,
                 config: EnvConfig = EnvConfig()) -> None:
        super().__init__()

        self.config = config

        self.action_space = self._create_action_space()
        self.observation_space = self._create_observation_space()

        self.state: RocketState | None = None
        self.goal = self._make_goal(config.payload_mass, config.target_altitude)
        self.step_count = 0
        self.has_lifted_off = False
        self.initial_fuel = config.initial_fuel

        self.last_fuel_used = 0.0
        self.last_drag = np.zeros(2, dtype=float)
        self.last_command_delta = np.zeros(2, dtype=float)

        self.throttle_command = 0.0
        self.angle_command = 90.0

    def _create_action_space(self) -> spaces.Box:
        """Return normalized command-rate action bounds."""

        action_space = spaces.Box(low=np.array([-1.0, -1.0], dtype=np.float32),
                                  high=np.array([1.0, 1.0], dtype=np.float32),
                                  dtype=np.float32)

        return action_space

    def _create_observation_space(self) -> spaces.Dict:
        """Return state and goal observation bounds."""

        observation_space = spaces.Dict({
            "state": spaces.Box(low=-np.inf,
                                high=np.inf,
                                shape=(10,),
                                dtype=np.float32),
            "goal": spaces.Box(low=-np.inf,
                               high=np.inf,
                               shape=(3,),
                               dtype=np.float32),
        })

        return observation_space

    def reset(self,
              *,
              seed: int | None = None,
              options: ResetOptions | None = None) -> tuple[Observation, Info]:
        """Reset the environment and return the initial observation."""

        super().reset(seed=seed)

        config = self.config
        options = options or {}

        target_altitude = float(options.get("target_altitude", config.target_altitude))
        payload_mass = float(options.get("payload_mass", config.payload_mass))
        initial_fuel = float(options.get("initial_fuel", config.initial_fuel))

        self.initial_fuel = initial_fuel

        start_radius = config.physics.planet_radius + config.initial_altitude

        self.state = RocketState(position=np.array([start_radius, 0.0], dtype=float),
                                 velocity=np.zeros(2, dtype=float),
                                 dry_mass=config.dry_mass,
                                 payload_mass=payload_mass,
                                 remaining_fuel=initial_fuel,
                                 time=0.0)

        self.goal = self._make_goal(payload_mass=payload_mass,
                                    target_altitude=target_altitude)

        self.step_count = 0
        self.last_fuel_used = 0.0
        self.last_drag = np.zeros(2, dtype=float)
        self.last_command_delta = np.zeros(2, dtype=float)

        self.throttle_command = 0.0
        self.angle_command = 90.0
        self.has_lifted_off = config.initial_altitude > 0.0

        observation = self._observation()
        info = self._info(fuel_used=0.0,
                          altitude_gain=0.0)

        return observation, info

    def step(self,
             action: np.ndarray) -> StepResult:
        """Apply action and return the resulting observation, reward, and info."""

        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        previous_state = self.state
        self.last_command_delta = self._apply_command_rate(action)

        applied_action = self._applied_action()
        result = step_dynamics(previous_state, applied_action, self.config.physics)

        self.state = self._apply_launch_pad_contact(previous_state, result.state)
        self.step_count += 1
        self.last_fuel_used = result.fuel_used
        self.last_drag = result.drag.copy()
        self.has_lifted_off = self.has_lifted_off or (
            altitude(self.state.position, self.config.physics) > GROUND_CONTACT_TOLERANCE_M
        )

        previous_altitude = altitude(previous_state.position, self.config.physics)
        current_altitude = altitude(self.state.position, self.config.physics)
        altitude_gain = current_altitude - previous_altitude

        info = self._info(fuel_used=result.fuel_used,
                          altitude_gain=altitude_gain)
        reward, components = reward_function(info, self.config.reward_name)

        info["reward_components"] = components

        terminated = bool(info["crashed"] or info["success"])

        if self.config.terminate_on_fuel_empty and self.state.remaining_fuel <= 0.0:
            terminated = True
            info["fuel_empty"] = True
        else:
            info["fuel_empty"] = self.state.remaining_fuel <= 0.0

        truncated = self.step_count >= self.config.max_steps
        observation = self._observation()
        step_result = observation, float(reward), terminated, truncated, info

        return step_result

    def force_state(self,
                    state: RocketState) -> None:
        self.state = state
        self.has_lifted_off = altitude(state.position, self.config.physics) > 0.0

    def with_physics(self,
                     physics: PhysicsConfig) -> "RocketAscentEnv":
        env = RocketAscentEnv(replace(self.config, physics=physics))

        return env

    def _apply_command_rate(self,
                            action: np.ndarray) -> np.ndarray:
        command_rate = np.asarray(action, dtype=float).reshape(2)
        command_rate = np.clip(command_rate, -1.0, 1.0)

        previous_throttle = self.throttle_command
        previous_angle = self.angle_command

        self.throttle_command = float(np.clip(self.throttle_command
                                              + command_rate[0]
                                              * self.config.throttle_change_per_second
                                              * self.config.physics.dt,
                                              0.0,
                                              1.0))

        if self._angle_change_allowed():
            self.angle_command = float(normalize_angle_degrees(
                self.angle_command
                + command_rate[1]
                * self.config.angle_change_per_second
                * self.config.physics.dt,
            ))

        command_delta = np.array([self.throttle_command - previous_throttle,
                                  shortest_angle_delta_degrees(self.angle_command,
                                                               previous_angle)],
                                 dtype=float)

        return command_delta

    def _angle_change_allowed(self) -> bool:
        if self.state is None:
            raise RuntimeError("Environment has no state.")

        current_altitude = altitude(self.state.position, self.config.physics)
        angle_change_allowed = current_altitude >= LAUNCH_ANGLE_LOCK_ALTITUDE_M

        return angle_change_allowed

    def _applied_action(self) -> np.ndarray:
        applied_action = np.array([self.throttle_command, self.angle_command],
                                  dtype=np.float32)

        return applied_action

    def _apply_launch_pad_contact(self,
                                  previous_state: RocketState,
                                  candidate_state: RocketState) -> RocketState:
        previous_altitude = altitude(previous_state.position, self.config.physics)
        candidate_altitude = altitude(candidate_state.position, self.config.physics)

        if (
            self.has_lifted_off
            or previous_altitude < -GROUND_CONTACT_TOLERANCE_M
            or candidate_altitude > GROUND_CONTACT_TOLERANCE_M
        ):
            return candidate_state

        radial = radial_unit(candidate_state.position)
        velocity = np.asarray(candidate_state.velocity, dtype=float)
        vertical_speed = float(np.dot(velocity, radial))

        if vertical_speed < 0.0:
            velocity = velocity - radial * vertical_speed

        grounded_state = RocketState(position=radial * self.config.physics.planet_radius,
                                     velocity=velocity,
                                     dry_mass=candidate_state.dry_mass,
                                     payload_mass=candidate_state.payload_mass,
                                     remaining_fuel=candidate_state.remaining_fuel,
                                     time=candidate_state.time)

        return grounded_state

    def _make_goal(self,
                   payload_mass: float,
                   target_altitude: float) -> np.ndarray:
        target_radius = self.config.physics.planet_radius + target_altitude
        target_velocity = circular_orbit_velocity(target_radius, self.config.physics)

        rocket_goal = RocketGoal(target_radius=target_radius,
                                 target_orbital_velocity=target_velocity,
                                 payload_mass=payload_mass)
        goal = rocket_goal.to_array()

        return goal

    def _observation(self) -> Observation:
        if self.state is None:
            raise RuntimeError("Environment has no state.")

        mass = mass_from_fuel(self.state.dry_mass,
                              self.state.payload_mass,
                              self.state.remaining_fuel)

        rocket_goal = RocketGoal.from_array(self.goal)
        structured_observation = RocketObservation(
            position=self.state.position.copy(),
            velocity=self.state.velocity.copy(),
            mass=mass,
            fuel_remaining=self.state.remaining_fuel,
            time=self.state.time,
            current_throttle_command=self.throttle_command,
            current_angle_degrees=self.angle_command,
            initial_fuel=self.initial_fuel,
            target_radius=rocket_goal.target_radius,
            target_orbital_velocity=rocket_goal.target_orbital_velocity,
            payload_mass=rocket_goal.payload_mass,
        )
        observation = structured_observation.to_mapping()

        return observation

    def _info(self,
              fuel_used: float,
              altitude_gain: float) -> Info:
        if self.state is None:
            raise RuntimeError("Environment has no state.")

        current_altitude = altitude(self.state.position, self.config.physics)
        current_density = atmosphere_density(current_altitude, self.config.physics)
        current_pressure = ambient_pressure(current_altitude, self.config.physics)

        current_engine_efficiency = sea_level_engine_vacuum_efficiency(
            current_pressure,
            self.config.physics.sea_level_pressure,
            self.config.physics.sea_level_relative_efficiency,
            self.config.physics.vacuum_relative_efficiency,
        )

        current_gravity = gravity_acceleration(self.state.position, self.config.physics)
        speed = float(np.linalg.norm(self.state.velocity))
        vertical_speed, tangential_velocity = decompose_velocity(self.state.position,
                                                                 self.state.velocity)

        target_radius = float(self.goal[0])
        target_velocity = float(self.goal[1])
        target_altitude = target_radius - self.config.physics.planet_radius

        altitude_error = abs(float(np.linalg.norm(self.state.position)) - target_radius)
        tangential_velocity_error = abs(tangential_velocity - target_velocity)
        vertical_speed_error = abs(vertical_speed)

        success = self._success(altitude_error,
                                vertical_speed_error,
                                tangential_velocity_error,
                                current_altitude)

        fuel_used_mass = float(fuel_used)
        fuel_used_ratio = self._fuel_used_ratio(fuel_used_mass)

        info = {"altitude": float(current_altitude),
                "altitude_gain": float(altitude_gain),
                "speed": speed,
                "vertical_speed": float(vertical_speed),
                "tangential_velocity": float(tangential_velocity),
                "fuel_remaining": float(self.state.remaining_fuel),
                "fuel_used_ratio": fuel_used_ratio,
                "fuel_used_mass": fuel_used_mass,
                "density": float(current_density),
                "ambient_pressure": float(current_pressure),
                "engine_efficiency": float(current_engine_efficiency),
                "gravity": current_gravity.copy(),
                "drag": self.last_drag.copy(),
                "command_delta": self.last_command_delta.copy(),
                "applied_action": self._applied_action(),
                "target_altitude": float(target_altitude),
                "target_orbital_velocity": float(target_velocity),
                "altitude_error": float(altitude_error),
                "tangential_velocity_error": float(tangential_velocity_error),
                "vertical_speed_error": float(vertical_speed_error),
                "crashed": self._crashed(current_altitude),
                "success": success,
                "reward_components": {}}

        return info

    def _fuel_used_ratio(self,
                         fuel_used: float) -> float:
        if self.initial_fuel <= 0.0:
            return 0.0

        fuel_used_ratio = float(fuel_used) / float(self.initial_fuel)

        return fuel_used_ratio

    def _crashed(self,
                 current_altitude: float) -> bool:
        if current_altitude < -GROUND_CONTACT_TOLERANCE_M:
            return True

        crashed = bool(self.has_lifted_off
                       and current_altitude <= GROUND_CONTACT_TOLERANCE_M
                       and self.step_count > 0)

        return crashed

    def _success(self,
                 altitude_error: float,
                 vertical_speed_error: float,
                 tangential_velocity_error: float,
                 current_altitude: float) -> bool:
        
        success = bool(current_altitude > 0.0
                       and altitude_error <= self.config.altitude_tolerance
                       and vertical_speed_error <= self.config.vertical_speed_tolerance
                       and tangential_velocity_error <= self.config.tangential_velocity_tolerance)

        return success
