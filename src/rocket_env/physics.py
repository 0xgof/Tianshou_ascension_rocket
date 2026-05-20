"""Pure physics helpers for the first 2D rocket ascent benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, degrees, exp, isfinite, log, pi, radians, sin, sqrt
from typing import Optional, Tuple

import numpy as np


Vector = np.ndarray


@dataclass(frozen=True)
class PhysicsConfig:
    """Configurable constants for the first rocket physics model."""

    planet_radius: float = 6_371_000.0
    mu: float = 3.986004418e14
    g0: float = 9.80665
    rho0: float = 1.225
    sea_level_pressure: float = 101_325.0
    scale_height: float = 8_500.0
    max_thrust: float = 7_800_000.0
    max_fuel_burn_rate: float = 2_350.0
    sea_level_relative_efficiency: float = 1.0
    vacuum_relative_efficiency: float = 0.88
    max_thrust_angle: float = pi
    drag_coefficient: float = 0.5
    reference_area: float = 10.0
    dt: float = 1.0
    enable_atmosphere: bool = True
    enable_drag: bool = True
    min_radius: float = 1.0


@dataclass(frozen=True)
class RocketState:
    """Physical state needed for deterministic dynamics stepping."""

    position: Vector
    velocity: Vector
    dry_mass: float
    payload_mass: float
    remaining_fuel: float
    time: float = 0.0


@dataclass(frozen=True)
class PhysicsStep:
    """Result of a single deterministic physics step."""

    state: RocketState
    acceleration: Vector
    gravity: Vector
    thrust: Vector
    drag: Vector
    throttle: float
    thrust_angle: float
    fuel_used: float
    density: float
    ambient_pressure: float
    engine_efficiency: float


DEFAULT_CONFIG = PhysicsConfig()

SEA_LEVEL_STANDARD_TEMPERATURE_K = 288.15
AIR_SPECIFIC_GAS_CONSTANT = 287.05287
LOWER_ATMOSPHERE_LIMIT_M = 84_852.0
LOWER_ATMOSPHERE_LAYERS = (
    (0.0, 288.15, 1.0, -0.0065),
    (11_000.0, 216.65, 0.2233611050922, 0.0),
    (20_000.0, 216.65, 0.0540329501078, 0.0010),
    (32_000.0, 228.65, 0.0085666783593, 0.0028),
    (47_000.0, 270.65, 0.0010945601338, 0.0),
    (51_000.0, 270.65, 0.0006606353133, -0.0028),
    (71_000.0, 214.65, 0.0000390468337, -0.0020),
)
UPPER_ATMOSPHERE_TABLE = (
    (86_000.0, 3.732e-1, 6.955e-6),
    (90_000.0, 1.8435e-1, 3.4400e-6),
    (95_000.0, 7.5775e-2, 1.3873e-6),
    (100_000.0, 3.2012e-2, 5.6044e-7),
    (110_000.0, 7.1493e-3, 9.6734e-8),
    (120_000.0, 2.5366e-3, 2.2199e-8),
    (130_000.0, 1.2503e-3, 8.1494e-9),
    (140_000.0, 7.2029e-4, 3.8313e-9),
    (150_000.0, 4.5422e-4, 2.0752e-9),
    (160_000.0, 3.0394e-4, 1.2336e-9),
    (170_000.0, 2.1210e-4, 7.8155e-10),
    (180_000.0, 1.5272e-4, 5.1940e-10),
    (190_000.0, 1.1265e-4, 3.5807e-10),
    (200_000.0, 8.4736e-5, 2.5407e-10),
    (250_000.0, 2.4762e-5, 6.0706e-11),
    (300_000.0, 8.7704e-6, 1.9159e-11),
    (350_000.0, 3.4446e-6, 7.0011e-12),
    (400_000.0, 1.4518e-6, 2.8028e-12),
)


def as_vector(value: Vector) -> Vector:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (2,):
        raise ValueError("Expected a 2D vector with shape (2,).")
    return vector


def vector_norm(value: Vector) -> float:
    return float(np.linalg.norm(as_vector(value)))


def unit_vector(value: Vector, fallback: Optional[Vector] = None) -> Vector:
    vector = as_vector(value)
    norm = vector_norm(vector)
    if norm <= 0.0:
        if fallback is None:
            return np.zeros(2, dtype=float)
        return as_vector(fallback).copy()
    return vector / norm


def all_finite(*values: object) -> bool:
    for value in values:
        array = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(array)):
            return False
    return True


def normalize_angle_degrees(angle_degrees: float) -> float:
    """Normalize an angle to the [-180, 180] degree convention."""
    normalized = (float(angle_degrees) + 180.0) % 360.0 - 180.0
    if normalized == -180.0 and float(angle_degrees) > 0.0:
        return 180.0
    return float(normalized)


def shortest_angle_delta_degrees(target_degrees: float, current_degrees: float) -> float:
    return normalize_angle_degrees(float(target_degrees) - float(current_degrees))


def altitude(position: Vector, config: PhysicsConfig = DEFAULT_CONFIG) -> float:
    return vector_norm(position) - config.planet_radius


def radial_unit(position: Vector) -> Vector:
    return unit_vector(position, fallback=np.array([1.0, 0.0]))


def tangential_unit(position: Vector) -> Vector:
    radial = radial_unit(position)
    return np.array([-radial[1], radial[0]], dtype=float)


def gravity_acceleration(
    position: Vector,
    config: PhysicsConfig = DEFAULT_CONFIG,
) -> Vector:
    radius_vector = as_vector(position)
    radius = max(vector_norm(radius_vector), config.min_radius)
    return -config.mu * radius_vector / radius**3


def circular_orbit_velocity(
    radius: float,
    config: PhysicsConfig = DEFAULT_CONFIG,
) -> float:
    if radius <= 0.0:
        raise ValueError("Orbit radius must be positive.")
    return sqrt(config.mu / radius)


def decompose_velocity(position: Vector, velocity: Vector) -> Tuple[float, float]:
    velocity_vector = as_vector(velocity)
    radial = radial_unit(position)
    tangential = tangential_unit(position)
    vertical_speed = float(np.dot(velocity_vector, radial))
    tangential_velocity = float(np.dot(velocity_vector, tangential))
    return vertical_speed, tangential_velocity


def map_bounded_action(
    action: Vector,
    config: PhysicsConfig = DEFAULT_CONFIG,
) -> Tuple[float, float]:
    command = as_vector(action)
    throttle = float(np.clip(command[0], 0.0, 1.0))
    max_angle_degrees = degrees(config.max_thrust_angle)
    angle_degrees = normalize_angle_degrees(command[1])
    if max_angle_degrees < 180.0:
        angle_degrees = float(np.clip(angle_degrees, -max_angle_degrees, max_angle_degrees))
    thrust_angle = float(radians(angle_degrees))
    return throttle, thrust_angle


def map_normalized_action(
    action: Vector,
    config: PhysicsConfig = DEFAULT_CONFIG,
) -> Tuple[float, float]:
    return map_bounded_action(action, config)


def mass_from_fuel(
    dry_mass: float,
    payload_mass: float,
    remaining_fuel: float,
) -> float:
    return float(dry_mass + payload_mass + max(0.0, remaining_fuel))


def burn_fuel(
    remaining_fuel: float,
    throttle: float,
    dt: float,
    config: PhysicsConfig = DEFAULT_CONFIG,
) -> Tuple[float, float, float]:
    if remaining_fuel <= 0.0 or throttle <= 0.0 or dt <= 0.0:
        return max(0.0, remaining_fuel), 0.0, 0.0

    requested = max(0.0, throttle) * config.max_fuel_burn_rate * dt
    fuel_used = min(remaining_fuel, requested)
    new_remaining = max(0.0, remaining_fuel - fuel_used)
    if requested <= 0.0:
        effective_throttle = 0.0
    else:
        effective_throttle = throttle * (fuel_used / requested)
    return float(new_remaining), float(fuel_used), float(effective_throttle)


def geopotential_altitude(geometric_altitude_m: float, config: PhysicsConfig) -> float:
    altitude_m = max(0.0, float(geometric_altitude_m))
    return config.planet_radius * altitude_m / (config.planet_radius + altitude_m)


def lower_standard_atmosphere(
    altitude_m: float,
    config: PhysicsConfig = DEFAULT_CONFIG,
) -> Tuple[float, float]:
    h = min(geopotential_altitude(altitude_m, config), LOWER_ATMOSPHERE_LIMIT_M)
    base_height, base_temperature, base_pressure_ratio, lapse_rate = (
        LOWER_ATMOSPHERE_LAYERS[0]
    )
    for layer in LOWER_ATMOSPHERE_LAYERS:
        if layer[0] <= h:
            base_height, base_temperature, base_pressure_ratio, lapse_rate = layer
        else:
            break

    height_delta = h - base_height
    temperature = base_temperature + lapse_rate * height_delta
    if lapse_rate == 0.0:
        pressure_ratio = base_pressure_ratio * exp(
            -config.g0 * height_delta / (AIR_SPECIFIC_GAS_CONSTANT * base_temperature)
        )
    else:
        pressure_ratio = base_pressure_ratio * (
            temperature / base_temperature
        ) ** (-config.g0 / (AIR_SPECIFIC_GAS_CONSTANT * lapse_rate))

    density_ratio = pressure_ratio * SEA_LEVEL_STANDARD_TEMPERATURE_K / temperature
    return (
        float(max(0.0, config.sea_level_pressure * pressure_ratio)),
        float(max(0.0, config.rho0 * density_ratio)),
    )


def log_interpolated_atmosphere(
    altitude_m: float,
    table: Tuple[Tuple[float, float, float], ...] = UPPER_ATMOSPHERE_TABLE,
) -> Tuple[float, float]:
    altitude_m = max(0.0, float(altitude_m))
    lower = table[0]
    upper = table[-1]
    for index in range(len(table) - 1):
        current = table[index]
        candidate = table[index + 1]
        if current[0] <= altitude_m <= candidate[0]:
            lower = current
            upper = candidate
            break
    else:
        if altitude_m < table[0][0]:
            lower, upper = table[0], table[1]
        else:
            lower, upper = table[-2], table[-1]

    lower_altitude, lower_pressure, lower_density = lower
    upper_altitude, upper_pressure, upper_density = upper
    span = upper_altitude - lower_altitude
    if span <= 0.0:
        return float(lower_pressure), float(lower_density)
    fraction = (altitude_m - lower_altitude) / span
    pressure = exp(log(lower_pressure) + fraction * log(upper_pressure / lower_pressure))
    density = exp(log(lower_density) + fraction * log(upper_density / lower_density))
    return float(max(0.0, pressure)), float(max(0.0, density))


def standard_atmosphere(
    altitude_m: float,
    config: PhysicsConfig = DEFAULT_CONFIG,
) -> Tuple[float, float]:
    if altitude_m <= 86_000.0:
        return lower_standard_atmosphere(altitude_m, config)
    return log_interpolated_atmosphere(altitude_m)


def ambient_pressure(
    altitude_m: float,
    config: PhysicsConfig = DEFAULT_CONFIG,
) -> float:
    if not config.enable_atmosphere:
        return 0.0
    pressure, _density = standard_atmosphere(max(0.0, float(altitude_m)), config)
    return pressure


def sea_level_engine_vacuum_efficiency(
    ambient_pressure: float,
    sea_level_pressure: float = 101_325.0,
    sea_level_relative_efficiency: float = 1.0,
    vacuum_relative_efficiency: float = 0.88,
) -> float:
    if sea_level_pressure <= 0.0:
        return float(vacuum_relative_efficiency)
    pressure_fraction = ambient_pressure / sea_level_pressure
    pressure_fraction = max(0.0, min(1.0, pressure_fraction))
    efficiency = vacuum_relative_efficiency + (
        sea_level_relative_efficiency - vacuum_relative_efficiency
    ) * pressure_fraction
    return float(efficiency)


def atmosphere_density(
    altitude_m: float,
    config: PhysicsConfig = DEFAULT_CONFIG,
) -> float:
    if not config.enable_atmosphere:
        return 0.0
    _pressure, density = standard_atmosphere(max(0.0, float(altitude_m)), config)
    return density


def drag_acceleration(
    velocity: Vector,
    mass: float,
    density: float,
    config: PhysicsConfig = DEFAULT_CONFIG,
) -> Vector:
    velocity_vector = as_vector(velocity)
    speed = vector_norm(velocity_vector)
    if not config.enable_drag or density <= 0.0 or speed <= 0.0 or mass <= 0.0:
        return np.zeros(2, dtype=float)

    drag_force = 0.5 * density * speed**2 * config.drag_coefficient * config.reference_area
    return -unit_vector(velocity_vector) * (drag_force / mass)


def thrust_direction(position: Vector, thrust_angle: float) -> Vector:
    radial = radial_unit(position)
    tangential = tangential_unit(position)
    direction = tangential * cos(thrust_angle) + radial * sin(thrust_angle)
    return unit_vector(direction, fallback=tangential)


def thrust_acceleration(
    position: Vector,
    mass: float,
    throttle: float,
    thrust_angle: float,
    config: PhysicsConfig = DEFAULT_CONFIG,
) -> Vector:
    if throttle <= 0.0 or mass <= 0.0:
        return np.zeros(2, dtype=float)
    pressure = ambient_pressure(altitude(position, config), config)
    engine_efficiency = sea_level_engine_vacuum_efficiency(
        pressure,
        config.sea_level_pressure,
        config.sea_level_relative_efficiency,
        config.vacuum_relative_efficiency,
    )
    force = throttle * config.max_thrust * engine_efficiency
    return thrust_direction(position, thrust_angle) * (force / mass)


def step_dynamics(
    state: RocketState,
    action: Vector,
    config: PhysicsConfig = DEFAULT_CONFIG,
) -> PhysicsStep:
    position = as_vector(state.position)
    velocity = as_vector(state.velocity)
    throttle_command, thrust_angle = map_bounded_action(action, config)
    remaining_fuel, fuel_used, throttle = burn_fuel(
        state.remaining_fuel,
        throttle_command,
        config.dt,
        config,
    )

    mass_before = mass_from_fuel(state.dry_mass, state.payload_mass, state.remaining_fuel)
    altitude_m = altitude(position, config)
    density = atmosphere_density(altitude_m, config)
    pressure = ambient_pressure(altitude_m, config)
    engine_efficiency = sea_level_engine_vacuum_efficiency(
        pressure,
        config.sea_level_pressure,
        config.sea_level_relative_efficiency,
        config.vacuum_relative_efficiency,
    )
    gravity = gravity_acceleration(position, config)
    thrust = thrust_acceleration(position, mass_before, throttle, thrust_angle, config)
    drag = drag_acceleration(velocity, mass_before, density, config)
    acceleration = gravity + thrust + drag

    next_velocity = velocity + acceleration * config.dt
    next_position = position + next_velocity * config.dt
    next_state = RocketState(
        position=next_position,
        velocity=next_velocity,
        dry_mass=state.dry_mass,
        payload_mass=state.payload_mass,
        remaining_fuel=remaining_fuel,
        time=state.time + config.dt,
    )

    if not all_finite(next_state.position, next_state.velocity, next_state.remaining_fuel):
        raise FloatingPointError("Physics step produced a non-finite state.")

    return PhysicsStep(
        state=next_state,
        acceleration=acceleration,
        gravity=gravity,
        thrust=thrust,
        drag=drag,
        throttle=throttle,
        thrust_angle=thrust_angle,
        fuel_used=fuel_used,
        density=density,
        ambient_pressure=pressure,
        engine_efficiency=engine_efficiency,
    )
