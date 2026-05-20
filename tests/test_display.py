import numpy as np
import pytest

from rocket_env.env import EnvConfig, RocketAscentEnv
from rocket_env.live_display import (
    DisplayConfig,
    PygameFlightApp,
    density_indicator_ratio,
    engine_efficiency_indicator_ratio,
    hud_lines,
    manual_step_period,
    overview_camera_limits,
    panel_to_screen,
    rocket_body_points,
    reward_plot_series,
    screen_absolute_vector,
    screen_thrust_direction,
    screen_velocity_vector,
    world_to_local,
)
from rocket_env.manual_control import ManualControlState
from rocket_env.plots import plot_trajectory
from rocket_env.render import collect_display_trace
from rocket_env.render import world_to_screen as render_world_to_screen
from rocket_env.render import Viewport
from rocket_env.rollouts import radial_thrust_policy


def test_manual_control_initial_action_is_valid():
    control = ManualControlState()
    action = control.action()
    assert action.shape == (2,)
    assert action[0] == 0.0
    assert action[1] == 0.0
    assert control.throttle_step == 0.01
    assert control.angle_step == 1.0


def test_manual_control_throttle_step_is_fine_grained():
    control = ManualControlState()
    control.apply_key("up")
    assert control.throttle_command == 0.01
    assert control.action(np.array([0.0, 90.0]))[0] == 1.0


def test_manual_control_throttle_increase_is_clamped():
    control = ManualControlState()
    for _ in range(100):
        control.apply_key("up")
    assert control.throttle_command == 1.0


def test_manual_control_throttle_cut_sets_minimum_throttle_command():
    control = ManualControlState()
    control.apply_key("up")
    control.apply_key(" ")
    assert control.throttle_command == 0.0
    assert control.action(np.array([0.01, 90.0]))[0] == -1.0


def test_manual_control_throttle_never_goes_negative():
    control = ManualControlState()
    for _ in range(100):
        control.apply_key("down")
    assert control.throttle_command == 0.0


def test_manual_control_angle_wraps_full_circle():
    control = ManualControlState()
    for _ in range(181):
        control.apply_key("right")
    assert control.angle_command == -91.0
    for _ in range(271):
        control.apply_key("left")
    assert control.angle_command == 180.0
    control.apply_key("left")
    assert control.angle_command == -179.0


def test_manual_control_angle_zero_is_horizontal_and_allows_negative():
    control = ManualControlState()
    for _ in range(90):
        control.apply_key("right")
    assert control.angle_command == 0.0
    control.apply_key("right")
    assert control.angle_command == -1.0


def test_manual_control_uses_shortest_angle_delta_across_wrap():
    control = ManualControlState(angle_command=-179.0)
    action = control.action(np.array([0.0, 179.0]))
    assert action[1] == 1.0


def test_manual_control_angle_alias_keys_match_a_and_d():
    left_control = ManualControlState(angle_command=0.0)
    alias_left_control = ManualControlState(angle_command=0.0)
    right_control = ManualControlState(angle_command=0.0)
    alias_right_control = ManualControlState(angle_command=0.0)

    left_control.apply_key("a")
    alias_left_control.apply_key("<")
    right_control.apply_key("d")
    alias_right_control.apply_key(">")

    current = np.array([0.0, 0.0])
    assert left_control.action(current)[1] == alias_left_control.action(current)[1] == 1.0
    assert right_control.action(current)[1] == alias_right_control.action(current)[1] == -1.0


def test_manual_control_reset_commands_returns_to_neutral_rate():
    control = ManualControlState(throttle_command=0.7, angle_command=-20.0)
    applied = np.array([0.0, 90.0])
    control.reset_commands(applied)
    np.testing.assert_allclose(control.action(applied), np.array([0.0, 0.0]))
    assert control.throttle_command == 0.0
    assert control.angle_command == 90.0


def test_plot_trajectory_creates_figure():
    env = RocketAscentEnv(EnvConfig(max_steps=5))
    trace, _info = collect_display_trace(env, radial_thrust_policy, max_steps=5)
    fig = plot_trajectory(trace, env)
    assert fig is not None


def test_plots_module_does_not_force_agg_backend():
    source_path = __import__("rocket_env.plots").plots.__file__
    with open(source_path, "r", encoding="utf-8") as handle:
        source = handle.read()
    assert 'matplotlib.use("Agg")' not in source
    assert "matplotlib.use('Agg')" not in source


def test_plot_trajectory_handles_empty_or_single_point_trace():
    env = RocketAscentEnv()
    with pytest.raises(ValueError):
        plot_trajectory([], env)

    trace, _info = collect_display_trace(env, radial_thrust_policy, max_steps=1)
    fig = plot_trajectory(trace, env)
    assert fig is not None


def test_render_projection_returns_finite_screen_coordinates():
    x, y = render_world_to_screen(np.array([1_000.0, -2_000.0]), Viewport())
    assert np.isfinite(x)
    assert np.isfinite(y)


def test_display_does_not_require_policy_object_for_scripted_rollout():
    env = RocketAscentEnv(EnvConfig(max_steps=5))
    trace, info = collect_display_trace(env, radial_thrust_policy, max_steps=5)
    assert len(trace) > 0
    assert "altitude" in info


def test_pygame_display_local_coordinates_use_altitude_and_tangent():
    origin = np.array([6_371_000.0, 0.0])
    position = np.array([6_371_100.0, 20.0])
    local = world_to_local(position, origin, 6_371_000.0)
    assert local[0] == 20.0
    assert local[1] > 99.0


def test_panel_projection_maps_local_square_to_screen():
    rect = (0, 0, 800, 600)
    x, y = panel_to_screen(np.array([0.0, 0.0]), rect, (-100.0, 100.0), (-100.0, 100.0))
    assert x == 400
    assert y == 300


def test_rocket_body_points_rotate_with_forward_vector():
    center = np.array([100.0, 100.0])
    upright = rocket_body_points(center, np.array([0.0, -1.0]), 40.0, 10.0)
    right = rocket_body_points(center, np.array([1.0, 0.0]), 40.0, 10.0)
    assert len(upright) == 5
    assert upright[0][1] < center[1]
    assert right[0][0] > center[0]


def test_screen_thrust_direction_maps_radial_up_at_launch():
    position = np.array([6_371_000.0, 0.0])
    horizontal = screen_thrust_direction(position, 0.0)
    vertical = screen_thrust_direction(position, np.pi / 2.0)
    np.testing.assert_allclose(horizontal, np.array([1.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(vertical, np.array([0.0, -1.0]), atol=1e-12)


def test_screen_velocity_vector_uses_absolute_local_frame():
    position = np.array([6_371_000.0, 0.0])
    vertical_speed = screen_velocity_vector(position, np.array([10.0, 0.0]))
    tangential_velocity = screen_velocity_vector(position, np.array([0.0, 10.0]))
    np.testing.assert_allclose(vertical_speed, np.array([0.0, -10.0]))
    np.testing.assert_allclose(tangential_velocity, np.array([10.0, 0.0]))


def test_screen_absolute_vector_projects_drag_like_velocity():
    position = np.array([6_371_000.0, 0.0])
    drag = np.array([0.0, -2.0])
    projected = screen_absolute_vector(position, drag)
    np.testing.assert_allclose(projected, np.array([-2.0, 0.0]))


def test_screen_absolute_vector_projects_gravity_downward_at_launch():
    position = np.array([6_371_000.0, 0.0])
    gravity = np.array([-9.8, 0.0])
    projected = screen_absolute_vector(position, gravity)
    np.testing.assert_allclose(projected, np.array([0.0, 9.8]))


def test_density_indicator_ratio_clamps_to_display_range():
    assert density_indicator_ratio(1.225, 1.225) == 1.0
    assert density_indicator_ratio(0.0, 1.225) == 0.0
    assert density_indicator_ratio(2.45, 1.225) == 1.0
    assert density_indicator_ratio(1.0, 0.0) == 0.0


def test_engine_efficiency_indicator_ratio_clamps_to_display_range():
    assert engine_efficiency_indicator_ratio(1.0) == 1.0
    assert engine_efficiency_indicator_ratio(0.88) == 0.88
    assert engine_efficiency_indicator_ratio(1.2) == 1.0
    assert engine_efficiency_indicator_ratio(-0.2) == 0.0


def test_hud_lines_include_vertical_and_horizontal_speed():
    info = {
        "altitude": 100.0,
        "speed": 12.0,
        "vertical_speed": 7.0,
        "tangential_velocity": 9.0,
        "fuel_remaining": 500.0,
        "target_altitude": 200_000.0,
        "target_orbital_velocity": 7_788.0,
        "altitude_error": 42.0,
        "vertical_speed_error": 2.0,
        "tangential_velocity_error": 100.0,
        "density": 1.0,
        "engine_efficiency": 0.88,
        "applied_action": np.array([0.5, 45.0]),
        "crashed": False,
        "success": False,
    }
    lines = hud_lines(info, np.array([0.5, 45.0]), "RUNNING")
    speed_line = next(line for line in lines if line.startswith("altitude="))
    assert "speed=12.0 m/s" in speed_line
    assert "vertical=7.0 m/s" in speed_line
    assert "horizontal=9.0 m/s" in speed_line
    assert "engine_efficiency=0.88" in lines


def test_reward_plot_series_aligns_total_reward_and_components():
    trace = [
        {"reward": 10.0,
         "info": {"reward_components": {"altitude_gain_bonus": 12.0,
                                        "fuel_penalty": -2.0}}},
        {"reward": -5.0,
         "info": {"reward_components": {"fuel_penalty": -5.0}}},
    ]

    series = reward_plot_series(trace)

    assert series["total_reward"] == [10.0, -5.0]
    assert series["altitude_gain_bonus"] == [12.0, 0.0]
    assert series["fuel_penalty"] == [-2.0, -5.0]


def test_manual_step_period_uses_display_rate_only():
    config = DisplayConfig(manual_simulation_rate_hz=5.0)
    assert manual_step_period(config) == 0.2


def test_overview_camera_follows_sideways_without_vertical_scroll():
    config = DisplayConfig(overview_size_m=200_000.0)
    rocket_local = np.array([80_000.0, 250_000.0])
    x_limits, y_limits = overview_camera_limits(rocket_local, config)
    rect = (0, 0, 800, 600)
    rocket_screen = panel_to_screen(rocket_local, rect, x_limits, y_limits)
    ground_screen = panel_to_screen(np.array([rocket_local[0], 0.0]), rect, x_limits, y_limits)
    assert rocket_screen[0] == 400
    assert y_limits == (0.0, 200_000.0)
    assert ground_screen[1] == 564


def test_pygame_flight_app_starts_paused():
    app = PygameFlightApp(DisplayConfig(max_steps=5))
    assert app.control.paused is True
    assert app.done is False
    assert app.trace == []


def test_pygame_flight_app_has_bottom_reward_panel():
    app = PygameFlightApp(DisplayConfig(width=1000, height=800, max_steps=5))
    left_rect, right_rect, reward_rect = app.panel_rects()

    assert left_rect == (0, 0, 500, 600)
    assert right_rect == (500, 0, 500, 600)
    assert reward_rect == (0, 600, 1000, 200)


def test_pygame_flight_app_reset_clears_manual_commands():
    app = PygameFlightApp(DisplayConfig(max_steps=5))
    app.control.apply_key("up")
    app.control.apply_key("right")
    assert not np.allclose(
        app.current_control_action(),
        np.array([0.0, 0.0]),
    )
    app.reset()
    np.testing.assert_allclose(
        app.current_control_action(),
        np.array([0.0, 0.0]),
    )
    np.testing.assert_allclose(app.info["applied_action"], np.array([0.0, 90.0]))


def test_pygame_flight_app_ignores_time_limit_truncation():
    app = PygameFlightApp(
        DisplayConfig(max_steps=1),
        env_config=EnvConfig(max_steps=1),
    )
    app.control.paused = False
    app.step()
    assert app.trace[-1]["truncated"] is True
    assert app.done is False
    assert app.control.paused is False


def test_pygame_flight_app_rate_limits_manual_steps():
    app = PygameFlightApp(
        DisplayConfig(max_steps=5, manual_simulation_rate_hz=5.0),
        env_config=EnvConfig(max_steps=5),
    )
    app.control.paused = False
    app.advance_manual_time(0.1)
    assert app.trace == []
    app.advance_manual_time(0.1)
    assert len(app.trace) == 1
