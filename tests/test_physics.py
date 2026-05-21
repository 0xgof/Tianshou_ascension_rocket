from math import isclose, pi, sqrt

import numpy as np

from rocket_env.physics import (
    DEFAULT_CONFIG,
    PhysicsConfig,
    RocketState,
    ambient_pressure,
    altitude,
    atmosphere_density,
    burn_fuel,
    circular_orbit_velocity,
    decompose_velocity,
    drag_acceleration,
    gravity_acceleration,
    map_bounded_action,
    mass_from_fuel,
    normalize_angle_degrees,
    sea_level_engine_vacuum_efficiency,
    shortest_angle_delta_degrees,
    step_dynamics,
    thrust_acceleration,
)


def test_altitude_at_surface_is_zero():
    position = np.array([DEFAULT_CONFIG.planet_radius, 0.0])
    assert isclose(altitude(position), 0.0, abs_tol=1e-9)


def test_gravity_points_toward_planet_center():
    position = np.array([DEFAULT_CONFIG.planet_radius + 1_000.0, 0.0])
    gravity = gravity_acceleration(position)
    assert gravity[0] < 0.0
    assert isclose(gravity[1], 0.0, abs_tol=1e-12)


def test_gravity_magnitude_decreases_with_radius():
    low = np.array([DEFAULT_CONFIG.planet_radius + 100_000.0, 0.0])
    high = np.array([DEFAULT_CONFIG.planet_radius + 500_000.0, 0.0])
    assert np.linalg.norm(gravity_acceleration(low)) > np.linalg.norm(
        gravity_acceleration(high)
    )


def test_circular_orbit_velocity_matches_formula():
    radius = DEFAULT_CONFIG.planet_radius + 250_000.0
    expected = sqrt(DEFAULT_CONFIG.mu / radius)
    assert isclose(circular_orbit_velocity(radius), expected, rel_tol=1e-12)


def test_radial_and_tangential_velocity_decomposition():
    position = np.array([DEFAULT_CONFIG.planet_radius + 100_000.0, 0.0])
    velocity = np.array([0.0, 1_500.0])
    vertical_speed, tangential_velocity = decompose_velocity(position, velocity)
    assert isclose(vertical_speed, 0.0, abs_tol=1e-12)
    assert isclose(tangential_velocity, 1_500.0, rel_tol=1e-12)


def test_action_mapping_bounds():
    low_throttle, low_angle = map_bounded_action(np.array([0.0, -180.0]))
    high_throttle, high_angle = map_bounded_action(np.array([1.0, 180.0]))
    assert low_throttle == 0.0
    assert high_throttle == 1.0
    assert low_angle == -pi
    assert high_angle == pi


def test_action_angle_zero_is_horizontal_and_ninety_is_vertical():
    _throttle, horizontal_angle = map_bounded_action(np.array([1.0, 0.0]))
    _throttle, vertical_angle = map_bounded_action(np.array([1.0, 90.0]))
    assert horizontal_angle == 0.0
    assert isclose(vertical_angle, pi / 2.0)


def test_action_angle_supports_full_circle_wrapping():
    _throttle, down_angle = map_bounded_action(np.array([1.0, 270.0]))
    _throttle, wrapped_horizontal = map_bounded_action(np.array([1.0, 360.0]))
    assert isclose(down_angle, -pi / 2.0)
    assert isclose(wrapped_horizontal, 0.0)
    assert normalize_angle_degrees(181.0) == -179.0
    assert shortest_angle_delta_degrees(-179.0, 179.0) == 2.0


def test_default_engine_specific_impulse_is_plausible():
    isp = DEFAULT_CONFIG.max_thrust / (
        DEFAULT_CONFIG.max_fuel_burn_rate * DEFAULT_CONFIG.g0
    )
    assert 500.0 <= isp <= 550.0


def test_ambient_pressure_decreases_with_altitude():
    sea_level = ambient_pressure(0.0)
    ten_km = ambient_pressure(10_000.0)
    fifty_km = ambient_pressure(50_000.0)
    assert sea_level == DEFAULT_CONFIG.sea_level_pressure
    assert sea_level > ten_km > fifty_km >= 0.0


def test_ambient_pressure_matches_standard_atmosphere_checkpoints():
    assert isclose(ambient_pressure(10_000.0), 26_500.0, rel_tol=1e-3)
    assert isclose(ambient_pressure(50_000.0), 79.779, rel_tol=1e-3)
    assert isclose(ambient_pressure(100_000.0), 0.032012, rel_tol=1e-3)


def test_sea_level_engine_vacuum_efficiency_drops_as_pressure_drops():
    sea_level = sea_level_engine_vacuum_efficiency(DEFAULT_CONFIG.sea_level_pressure)
    high_altitude = sea_level_engine_vacuum_efficiency(1_000.0)
    vacuum = sea_level_engine_vacuum_efficiency(0.0)
    overpressure = sea_level_engine_vacuum_efficiency(
        DEFAULT_CONFIG.sea_level_pressure * 2.0
    )
    assert sea_level == DEFAULT_CONFIG.sea_level_relative_efficiency
    assert vacuum < high_altitude < sea_level
    assert vacuum == DEFAULT_CONFIG.vacuum_relative_efficiency
    assert overpressure == DEFAULT_CONFIG.sea_level_relative_efficiency


def test_sea_level_engine_vacuum_efficiency_reduces_vacuum_thrust():
    position = np.array([DEFAULT_CONFIG.planet_radius, 0.0])
    sea_level = thrust_acceleration(
        position,
        mass=100_000.0,
        throttle=1.0,
        thrust_angle=np.pi / 2.0,
    )
    vacuum_config = PhysicsConfig(enable_atmosphere=False)
    vacuum = thrust_acceleration(
        position,
        mass=100_000.0,
        throttle=1.0,
        thrust_angle=np.pi / 2.0,
        config=vacuum_config,
    )
    ratio = np.linalg.norm(sea_level) / np.linalg.norm(vacuum)
    assert ratio == 1.0 / DEFAULT_CONFIG.vacuum_relative_efficiency


def test_negative_throttle_command_is_clamped_to_zero():
    throttle, _angle = map_bounded_action(np.array([-1.0, 0.0]))
    assert throttle == 0.0


def test_fuel_burn_is_monotonic_and_clamped():
    remaining, used, effective_throttle = burn_fuel(100.0, 1.0, 1.0)
    assert remaining < 100.0
    assert used > 0.0
    assert effective_throttle > 0.0

    remaining, used, effective_throttle = burn_fuel(10.0, 1.0, 10.0)
    assert remaining == 0.0
    assert used == 10.0
    assert 0.0 < effective_throttle <= 1.0


def test_mass_respects_dry_mass_and_payload_floor():
    dry_mass = 20_000.0
    payload_mass = 1_000.0
    assert mass_from_fuel(dry_mass, payload_mass, -100.0) == dry_mass + payload_mass


def test_atmosphere_density_decreases_with_altitude():
    sea_level = atmosphere_density(0.0)
    ten_km = atmosphere_density(10_000.0)
    fifty_km = atmosphere_density(50_000.0)
    assert sea_level > ten_km > fifty_km >= 0.0


def test_atmosphere_density_matches_standard_atmosphere_checkpoints():
    assert isclose(atmosphere_density(10_000.0), 0.41351, rel_tol=1e-3)
    assert isclose(atmosphere_density(50_000.0), 0.0010269, rel_tol=1e-3)
    assert isclose(atmosphere_density(100_000.0), 5.6044e-7, rel_tol=1e-3)


def test_drag_opposes_velocity():
    velocity = np.array([100.0, 10.0])
    drag = drag_acceleration(velocity, mass=10_000.0, density=1.0)
    assert np.dot(drag, velocity) < 0.0


def test_drag_is_zero_at_zero_velocity():
    drag = drag_acceleration(np.zeros(2), mass=10_000.0, density=1.0)
    np.testing.assert_allclose(drag, np.zeros(2))


def test_physics_step_returns_finite_values():
    config = PhysicsConfig(enable_drag=True, enable_atmosphere=True, dt=0.5)
    state = RocketState(
        position=np.array([config.planet_radius, 0.0]),
        velocity=np.array([0.0, 0.0]),
        dry_mass=20_000.0,
        payload_mass=2_000.0,
        remaining_fuel=100_000.0,
        time=0.0,
    )
    result = step_dynamics(state, np.array([1.0, 90.0]), config)
    assert np.all(np.isfinite(result.state.position))
    assert np.all(np.isfinite(result.state.velocity))
    assert np.isfinite(result.state.remaining_fuel)
    assert np.isfinite(result.fuel_used)
