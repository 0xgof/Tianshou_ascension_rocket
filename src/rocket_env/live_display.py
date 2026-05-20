"""Pygame live display for manual rocket flight."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rocket_env.env import EnvConfig, RocketAscentEnv
from rocket_env.manual_control import ManualControlState
from rocket_env.plots import plot_trajectory


@dataclass(frozen=True)
class DisplayConfig:
    width: int = 1100
    height: int = 820
    fps: int = 30
    manual_simulation_rate_hz: float = 5.0
    overview_size_m: float = 200_000.0
    zoom_size_m: float = 300.0
    rocket_height_m: float = 50.0
    karman_line_m: float = 100_000.0
    output_path: str = "results/figures/manual_flight.png"
    max_steps: int = 5_000


KEY_BINDINGS = {"up": "up",
                "w": "w",
                "down": "down",
                "s": "s",
                "left": "left",
                "a": "a",
                "<": "<",
                "less": "<",
                "comma": "<",
                "right": "right",
                "d": "d",
                ">": ">",
                "greater": ">",
                "period": ">",
                "space": " ",
                "r": "r",
                "p": "p",
                "escape": "escape"}

REWARD_PLOT_COLORS = {"total_reward": (245, 245, 245),
                      "altitude_gain_bonus": (70, 170, 240),
                      "success_bonus": (70, 210, 130),
                      "target_altitude_bonus": (85, 190, 255),
                      "tangential_velocity_bonus": (180, 150, 255),
                      "leaving_pad_bonus": (240, 210, 90),
                      "altitude_reached_bonus": (80, 220, 180),
                      "fuel_penalty": (245, 150, 45),
                      "crash_penalty": (235, 68, 52)}


def pygame_key_to_control(pygame,
                          key_code: int) -> str:
    name = pygame.key.name(key_code).lower()
    control_key = KEY_BINDINGS.get(name, "")

    return control_key


def world_to_local(position: np.ndarray,
                   origin: np.ndarray,
                   planet_radius: float) -> np.ndarray:
    origin_radial = origin / max(np.linalg.norm(origin), 1.0)
    origin_tangent = np.array([-origin_radial[1], origin_radial[0]])

    delta = np.asarray(position, dtype=float) - origin
    local_x = float(np.dot(delta, origin_tangent))
    local_y = float(np.linalg.norm(position) - planet_radius)

    local_position = np.array([local_x, local_y], dtype=float)

    return local_position


def panel_to_screen(local_xy: np.ndarray,
                    rect: tuple[int, int, int, int],
                    x_limits: tuple[float, float],
                    y_limits: tuple[float, float],
                    padding: int = 36) -> tuple[int, int]:
    left, top, width, height = rect

    inner_width = max(width - padding * 2, 1)
    inner_height = max(height - padding * 2, 1)

    x_ratio = (float(local_xy[0]) - x_limits[0]) / max(x_limits[1] - x_limits[0], 1.0)
    y_ratio = (float(local_xy[1]) - y_limits[0]) / max(y_limits[1] - y_limits[0], 1.0)

    x = int(left + padding + x_ratio * inner_width)
    y = int(top + height - padding - y_ratio * inner_height)

    screen_point = x, y

    return screen_point


def overview_camera_limits(rocket_local: np.ndarray,
                           config: DisplayConfig) -> tuple[tuple[float, float],
                                                           tuple[float, float]]:
    half_width = config.overview_size_m / 2.0
    center_x = float(rocket_local[0])

    camera_limits = ((center_x - half_width, center_x + half_width),
                     (0.0, config.overview_size_m))

    return camera_limits


def world_to_screen(position: np.ndarray,
                    center: np.ndarray,
                    config: DisplayConfig) -> tuple[int, int]:
    rect = (0, 0, config.width, config.height)
    half_span = config.overview_size_m / 2.0
    local = world_to_local(position, center, 0.0)

    screen_point = panel_to_screen(local,
                                   rect,
                                   (-half_span, half_span),
                                   (-half_span, half_span))

    return screen_point


def screen_thrust_direction(position: np.ndarray,
                            thrust_angle: float) -> np.ndarray:
    radial = np.asarray(position, dtype=float) / max(np.linalg.norm(position), 1.0)
    tangential = np.array([-radial[1], radial[0]])

    thrust_direction = tangential * np.cos(thrust_angle) + radial * np.sin(thrust_angle)
    screen_direction = np.array([thrust_direction[1], -thrust_direction[0]], dtype=float)

    return screen_direction


def screen_velocity_vector(position: np.ndarray,
                           velocity: np.ndarray) -> np.ndarray:
    screen_vector = screen_absolute_vector(position, velocity)

    return screen_vector


def screen_absolute_vector(position: np.ndarray,
                           vector: np.ndarray) -> np.ndarray:
    radial = np.asarray(position, dtype=float) / max(np.linalg.norm(position), 1.0)
    tangential = np.array([-radial[1], radial[0]], dtype=float)
    vector = np.asarray(vector, dtype=float)

    tangential_component = float(np.dot(vector, tangential))
    radial_component = float(np.dot(vector, radial))

    screen_vector = np.array([tangential_component, -radial_component], dtype=float)

    return screen_vector


def density_indicator_ratio(density: float,
                            sea_level_density: float) -> float:
    if sea_level_density <= 0.0:
        return 0.0

    ratio = float(np.clip(density / sea_level_density, 0.0, 1.0))

    return ratio


def engine_efficiency_indicator_ratio(efficiency: float) -> float:
    ratio = float(np.clip(efficiency, 0.0, 1.0))

    return ratio


def manual_step_period(config: DisplayConfig) -> float:
    period = 1.0 / max(float(config.manual_simulation_rate_hz), 1e-9)

    return period


def rocket_body_points(center: np.ndarray,
                       forward: np.ndarray,
                       height: float,
                       width: float) -> list[tuple[int, int]]:
    body_forward, right = rocket_axes(forward)
    center = np.asarray(center, dtype=float)

    nose = center + body_forward * (height * 0.52)
    shoulder = center + body_forward * (height * 0.28)
    tail = center - body_forward * (height * 0.38)

    left_shoulder = shoulder - right * (width * 0.30)
    right_shoulder = shoulder + right * (width * 0.30)
    left_tail = tail - right * (width * 0.46)
    right_tail = tail + right * (width * 0.46)

    body_points = [tuple(nose.astype(int)),
                   tuple(left_shoulder.astype(int)),
                   tuple(left_tail.astype(int)),
                   tuple(right_tail.astype(int)),
                   tuple(right_shoulder.astype(int))]

    return body_points


def rocket_axes(forward: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    body_forward = np.asarray(forward, dtype=float)
    norm = max(float(np.linalg.norm(body_forward)), 1e-9)

    body_forward = body_forward / norm
    right = np.array([body_forward[1], -body_forward[0]], dtype=float)

    axes = body_forward, right

    return axes


def rocket_fin_points(center: np.ndarray,
                      forward: np.ndarray,
                      height: float,
                      width: float) -> tuple[list[tuple[int, int]],
                                             list[tuple[int, int]]]:
    body_forward, right = rocket_axes(forward)
    center = np.asarray(center, dtype=float)

    tail = center - body_forward * (height * 0.38)
    fin_root = center - body_forward * (height * 0.20)

    left_fin = [fin_root - right * (width * 0.36),
                tail - right * (width * 0.46),
                tail - body_forward * (height * 0.18) - right * (width * 0.95)]

    right_fin = [fin_root + right * (width * 0.36),
                 tail + right * (width * 0.46),
                 tail - body_forward * (height * 0.18) + right * (width * 0.95)]

    fin_points = ([tuple(point.astype(int)) for point in left_fin],
                  [tuple(point.astype(int)) for point in right_fin])

    return fin_points


def trace_entry(obs,
                action,
                reward,
                terminated,
                truncated,
                info) -> dict[str, object]:
    applied_action = np.asarray(info.get("applied_action", action)).copy()

    entry = {"observation": obs,
             "action": action.copy(),
             "applied_action": applied_action,
             "reward": float(reward),
             "terminated": bool(terminated),
             "truncated": bool(truncated),
             "info": dict(info)}

    return entry


def hud_lines(info: dict[str, object],
              action: np.ndarray,
              status: str) -> list[str]:
    density = float(info.get("density", 0.0))
    engine_efficiency = float(info.get("engine_efficiency", 1.0))

    status_line = (f"status={status} throttle={info['applied_action'][0]:.2f} "
                   f"angle={info['applied_action'][1]:.1f} deg")

    velocity_line = (f"altitude={info['altitude']:.0f} m speed={info['speed']:.1f} m/s "
                     f"vertical={info['vertical_speed']:.1f} m/s "
                     f"horizontal={info['tangential_velocity']:.1f} m/s")

    hud_text = ["P start/pause | W/S throttle | A/D angle | Space cut | R reset | Esc exit",
                status_line,
                velocity_line,
                f"fuel={info['fuel_remaining']:.0f} "
                f"alt_err={info['altitude_error']:.1f} m",
                f"vertical_err={info['vertical_speed_error']:.1f} m/s "
                f"horizontal_err={info['tangential_velocity_error']:.1f} m/s",
                f"density={density:.4f} kg/m^3",
                f"engine_efficiency={engine_efficiency:.2f}",
                f"crashed={info['crashed']} success={info['success']}"]

    return hud_text


def reward_plot_series(trace: list[dict[str, object]]) -> dict[str, list[float]]:
    series: dict[str, list[float]] = {}

    for step_index, step in enumerate(trace):
        total_reward = float(step.get("reward", 0.0))
        series.setdefault("total_reward", []).append(total_reward)

        info = step.get("info", {})
        if not isinstance(info, dict):
            continue

        components = info.get("reward_components", {})
        if not isinstance(components, dict):
            continue

        for name, value in components.items():
            component_name = str(name)
            if component_name not in series:
                series[component_name] = [0.0] * step_index
            series[component_name].append(float(value))

        for name, values in series.items():
            if name not in components and len(values) < step_index + 1:
                values.append(0.0)

    return series


class PygameFlightApp:
    def __init__(self,
                 display_config: DisplayConfig = DisplayConfig(),
                 env_config: EnvConfig | None = None,
                 throttle_step: float = 0.01,
                 angle_step: float = 1.0) -> None:
        self.display_config = display_config
        self.env = RocketAscentEnv(env_config or EnvConfig(max_steps=display_config.max_steps))
        self.control = ManualControlState(throttle_step=throttle_step,
                                          angle_step=angle_step,
                                          paused=True)

        self.obs, self.info = self.env.reset(seed=0)
        self.initial_position = np.asarray(self.obs["state"][:2], dtype=float)
        self.trace = []
        self.done = False
        self.step_count = 0
        self.step_accumulator = 0.0

    def run(self) -> None:
        import pygame

        pygame.init()
        pygame.display.set_caption("RL Ascension Rocket - Manual Flight")

        screen = pygame.display.set_mode((self.display_config.width,
                                          self.display_config.height))
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("consolas", 18)

        running = True

        while running:
            elapsed_seconds = min(clock.tick(self.display_config.fps) / 1000.0, 0.25)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self.handle_key(pygame, event.key)

            self.advance_manual_time(elapsed_seconds)

            self.draw(pygame, screen, font)
            pygame.display.flip()

        self.save_trace()
        pygame.quit()

    def handle_key(self,
                   pygame,
                   key_code: int) -> bool:
        key = pygame_key_to_control(pygame, key_code)

        if not key:
            return True

        self.control.apply_key(key)

        if self.control.consume_reset():
            self.reset()

        keep_running = not self.control.exit_requested

        return keep_running

    def reset(self) -> None:
        self.obs, self.info = self.env.reset(seed=0)

        self.control.reset_commands(self.info.get("applied_action"))
        self.trace.clear()

        self.done = False
        self.step_count = 0
        self.step_accumulator = 0.0

    def advance_manual_time(self,
                            elapsed_seconds: float) -> None:
        if self.control.paused or self.done:
            self.step_accumulator = 0.0
            return

        self.step_accumulator += max(0.0, float(elapsed_seconds))

        period = manual_step_period(self.display_config)

        if self.step_accumulator < period:
            return

        self.step()
        self.step_accumulator = min(self.step_accumulator - period, period)

    def step(self) -> None:
        action = self.current_control_action()
        self.obs, reward, terminated, truncated, self.info = self.env.step(action)

        entry = trace_entry(self.obs,
                            action,
                            reward,
                            terminated,
                            truncated,
                            self.info)

        self.trace.append(entry)
        self.step_count += 1
        self.done = bool(terminated and self.info.get("crashed", False))

        if self.done:
            self.control.paused = True

    def current_control_action(self) -> np.ndarray:
        dt = self.env.config.physics.dt

        throttle_limit = self.env.config.throttle_change_per_second * dt
        angle_limit = self.env.config.angle_change_per_second * dt

        action = self.control.action(self.info.get("applied_action"),
                                     throttle_change_limit=throttle_limit,
                                     angle_change_limit=angle_limit)

        return action

    def draw(self,
             pygame,
             screen,
             font) -> None:
        screen.fill((14, 18, 24))

        left_rect, right_rect, reward_rect = self.panel_rects()

        self.draw_panel_background(pygame, screen, left_rect, "ASCENT OVERVIEW")
        self.draw_panel_background(pygame, screen, right_rect, "ROCKET ZOOM")
        self.draw_panel_background(pygame, screen, reward_rect, "REWARD COMPONENTS")

        previous_clip = screen.get_clip()

        screen.set_clip(left_rect)
        self.draw_overview(pygame, screen, left_rect)

        screen.set_clip(right_rect)
        self.draw_zoom(pygame, screen, right_rect)

        screen.set_clip(reward_rect)
        self.draw_reward_plot(pygame, screen, reward_rect)

        screen.set_clip(previous_clip)

        self.draw_hud(pygame, screen, font)

    def panel_rects(self) -> tuple[tuple[int, int, int, int],
                                  tuple[int, int, int, int],
                                  tuple[int, int, int, int]]:
        half_width = self.display_config.width // 2
        reward_height = max(self.display_config.height // 4, 180)
        top_height = self.display_config.height - reward_height

        panel_rects = ((0, 0, half_width, top_height),
                       (half_width,
                        0,
                        self.display_config.width - half_width,
                        top_height),
                       (0,
                        top_height,
                        self.display_config.width,
                        reward_height))

        return panel_rects

    def draw_panel_background(self,
                              pygame,
                              screen,
                              rect,
                              title: str) -> None:
        left, top, width, _height = rect

        pygame.draw.rect(screen, (20, 26, 36), rect)
        pygame.draw.rect(screen, (70, 80, 94), rect, width=1)

        font = pygame.font.SysFont("consolas", 18)
        title_surface = font.render(title, True, (220, 230, 240))

        screen.blit(title_surface, (left + 14, top + 12))

    def draw_overview(self,
                      pygame,
                      screen,
                      rect) -> None:
        rocket_local = world_to_local(self.obs["state"][:2],
                                      self.initial_position,
                                      self.env.config.physics.planet_radius)

        x_limits, y_limits = overview_camera_limits(rocket_local, self.display_config)

        self.draw_atmosphere(pygame, screen, rect, x_limits, y_limits)
        self.draw_ground(pygame, screen, rect, x_limits, y_limits)
        self.draw_karman_line(pygame, screen, rect, x_limits, y_limits)
        self.draw_overview_trace(pygame, screen, rect, x_limits, y_limits)

    def draw_atmosphere(self,
                        pygame,
                        screen,
                        rect,
                        _x_limits,
                        _y_limits) -> None:
        left, top, width, height = rect

        for i in range(height):
            t = i / max(height - 1, 1)

            color = (int(22 + 18 * (1 - t)),
                     int(34 + 50 * (1 - t)),
                     int(54 + 90 * (1 - t)))

            pygame.draw.line(screen,
                             color,
                             (left + 1, top + i),
                             (left + width - 2, top + i))

    def draw_ground(self,
                    pygame,
                    screen,
                    rect,
                    x_limits,
                    y_limits) -> None:
        y = panel_to_screen(np.array([0.0, 0.0]), rect, x_limits, y_limits)[1]

        pygame.draw.line(screen,
                         (100, 170, 95),
                         (rect[0] + 36, y),
                         (rect[0] + rect[2] - 36, y),
                         4)

    def draw_karman_line(self,
                         pygame,
                         screen,
                         rect,
                         x_limits,
                         y_limits) -> None:
        karman_position = np.array([0.0, self.display_config.karman_line_m])
        y = panel_to_screen(karman_position, rect, x_limits, y_limits)[1]

        pygame.draw.line(screen,
                         (235, 190, 85),
                         (rect[0] + 36, y),
                         (rect[0] + rect[2] - 36, y),
                         2)

        font = pygame.font.SysFont("consolas", 16)
        label = font.render("Karman line 100 km", True, (235, 210, 130))

        screen.blit(label, (rect[0] + 42, y - 22))

    def draw_overview_trace(self,
                            pygame,
                            screen,
                            rect,
                            x_limits,
                            y_limits) -> None:
        points = [panel_to_screen(world_to_local(step["observation"]["state"][:2],
                                                self.initial_position,
                                                self.env.config.physics.planet_radius),
                                  rect,
                                  x_limits,
                                  y_limits)
                  for step in self.trace]

        if len(points) > 1:
            pygame.draw.lines(screen, (235, 235, 235), False, points, width=2)

        rocket_local = world_to_local(self.obs["state"][:2],
                                      self.initial_position,
                                      self.env.config.physics.planet_radius)
        rocket_position = panel_to_screen(rocket_local, rect, x_limits, y_limits)

        pygame.draw.circle(screen, (235, 68, 52), rocket_position, 6)

    def draw_zoom(self,
                  pygame,
                  screen,
                  rect) -> None:
        position = np.asarray(self.obs["state"][:2], dtype=float)
        rocket_local = world_to_local(position,
                                      self.initial_position,
                                      self.env.config.physics.planet_radius)

        half = self.display_config.zoom_size_m / 2.0
        x_limits = (rocket_local[0] - half, rocket_local[0] + half)
        y_limits = (rocket_local[1] - half, rocket_local[1] + half)

        self.draw_ground(pygame, screen, rect, x_limits, y_limits)

        rocket = np.array(panel_to_screen(rocket_local, rect, x_limits, y_limits),
                          dtype=float)
        action = np.asarray(self.info.get("applied_action", [0.0, 90.0]), dtype=float)
        thrust_angle = np.deg2rad(action[1])
        screen_thrust = screen_thrust_direction(position, thrust_angle)

        self.draw_rocket(pygame, screen, rocket, rect, screen_thrust)

        velocity = np.asarray(self.obs["state"][2:4], dtype=float)

        if np.linalg.norm(velocity) > 0.0:
            screen_velocity = screen_velocity_vector(position, velocity)
            self.draw_arrow(pygame, screen, rocket, screen_velocity, (55, 188, 135), 0.08)

        drag = np.asarray(self.info.get("drag", np.zeros(2)), dtype=float)

        if np.linalg.norm(drag) > 0.0:
            screen_drag = screen_absolute_vector(position, drag)
            self.draw_arrow(pygame, screen, rocket, screen_drag, (230, 70, 70), 8.0)

        gravity = np.asarray(self.info.get("gravity", np.zeros(2)), dtype=float)

        if np.linalg.norm(gravity) > 0.0:
            screen_gravity = screen_absolute_vector(position, gravity)
            self.draw_arrow(pygame, screen, rocket, screen_gravity, (245, 150, 45), 8.0)

        self.draw_arrow(pygame, screen, rocket, screen_thrust, (130, 110, 220), 80.0)

    def draw_reward_plot(self,
                         pygame,
                         screen,
                         rect) -> None:
        series = reward_plot_series(self.trace)
        if not series:
            return

        left, top, width, height = rect
        plot_rect = (left + 48,
                     top + 42,
                     max(width - 280, 1),
                     max(height - 70, 1))
        values = [value
                  for values in series.values()
                  for value in values]
        y_min = min(values)
        y_max = max(values)

        if abs(y_max - y_min) < 1e-9:
            y_min -= 1.0
            y_max += 1.0

        zero_y = self.reward_plot_point(0,
                                        0.0,
                                        1,
                                        plot_rect,
                                        y_min,
                                        y_max)[1]
        pygame.draw.line(screen,
                         (70, 80, 94),
                         (plot_rect[0], zero_y),
                         (plot_rect[0] + plot_rect[2], zero_y),
                         1)

        for name, values in series.items():
            if len(values) < 2:
                continue

            color = REWARD_PLOT_COLORS.get(name, (190, 190, 190))
            points = [self.reward_plot_point(index,
                                             value,
                                             len(values),
                                             plot_rect,
                                             y_min,
                                             y_max)
                      for index, value in enumerate(values)]
            pygame.draw.lines(screen, color, False, points, width=2)

        self.draw_reward_plot_legend(pygame,
                                     screen,
                                     series,
                                     plot_rect,
                                     y_min,
                                     y_max)

    def reward_plot_point(self,
                          index: int,
                          value: float,
                          count: int,
                          rect: tuple[int, int, int, int],
                          y_min: float,
                          y_max: float) -> tuple[int, int]:
        left, top, width, height = rect
        x_ratio = index / max(count - 1, 1)
        y_ratio = (float(value) - y_min) / max(y_max - y_min, 1e-9)

        point = (int(left + x_ratio * width),
                 int(top + height - y_ratio * height))

        return point

    def draw_reward_plot_legend(self,
                                pygame,
                                screen,
                                series: dict[str, list[float]],
                                plot_rect: tuple[int, int, int, int],
                                y_min: float,
                                y_max: float) -> None:
        font = pygame.font.SysFont("consolas", 14)
        legend_x = plot_rect[0] + plot_rect[2] + 18
        legend_y = plot_rect[1]

        scale_text = f"range {y_min:.1f}..{y_max:.1f}"
        scale_surface = font.render(scale_text, True, (210, 220, 230))
        screen.blit(scale_surface, (plot_rect[0], plot_rect[1] - 24))

        for index, name in enumerate(series):
            color = REWARD_PLOT_COLORS.get(name, (190, 190, 190))
            y = legend_y + index * 18
            pygame.draw.line(screen,
                             color,
                             (legend_x, y + 8),
                             (legend_x + 18, y + 8),
                             3)
            label = font.render(name, True, (220, 230, 240))
            screen.blit(label, (legend_x + 26, y))

    def draw_rocket(self,
                    pygame,
                    screen,
                    rocket,
                    rect,
                    forward) -> None:
        left, top, width, height = rect

        pixels_per_meter = (height - 72) / self.display_config.zoom_size_m
        rocket_height = max(int(self.display_config.rocket_height_m * pixels_per_meter), 20)
        rocket_width = max(int(rocket_height * 0.28), 8)

        body = rocket_body_points(rocket,
                                  forward,
                                  float(rocket_height),
                                  float(rocket_width))
        left_fin, right_fin = rocket_fin_points(rocket,
                                                forward,
                                                float(rocket_height),
                                                float(rocket_width))
        body_forward, _right = rocket_axes(forward)

        engine = rocket - body_forward * (rocket_height * 0.56)
        tail = rocket - body_forward * (rocket_height * 0.38)

        pygame.draw.polygon(screen, (205, 210, 220), left_fin)
        pygame.draw.polygon(screen, (205, 210, 220), right_fin)
        pygame.draw.polygon(screen, (70, 80, 94), left_fin, width=2)
        pygame.draw.polygon(screen, (70, 80, 94), right_fin, width=2)
        pygame.draw.polygon(screen, (230, 230, 235), body)
        pygame.draw.polygon(screen, (70, 80, 94), body, width=2)
        pygame.draw.line(screen, (170, 175, 185), tail.astype(int), body[0], width=1)
        pygame.draw.circle(screen, (60, 68, 78), engine.astype(int), max(2, rocket_width // 6))
        pygame.draw.line(screen, (70, 80, 94), (left, top), (left + width, top), 1)

    def draw_arrow(self,
                   pygame,
                   screen,
                   start,
                   vector,
                   color,
                   scale) -> None:
        end = start + np.asarray(vector, dtype=float) * scale

        pygame.draw.line(screen, color, start.astype(int), end.astype(int), width=3)
        pygame.draw.circle(screen, color, end.astype(int), 4)

    def draw_hud(self,
                 pygame,
                 screen,
                 font) -> None:
        action = self.current_control_action()
        status = "CRASHED" if self.done else ("PAUSED" if self.control.paused else "RUNNING")
        lines = hud_lines(self.info, action, status)

        for index, line in enumerate(lines):
            surface = font.render(line, True, (240, 240, 240))
            screen.blit(surface, (18, 18 + index * 24))

        bar_y = 18 + len(lines) * 24 + 4

        self.draw_density_bar(pygame, screen, 18, bar_y)
        self.draw_engine_efficiency_bar(pygame, screen, 18, bar_y + 16)

    def draw_density_bar(self,
                         pygame,
                         screen,
                         x: int,
                         y: int) -> None:
        density = float(self.info.get("density", 0.0))
        ratio = density_indicator_ratio(density, self.env.config.physics.rho0)

        width = 180
        height = 10

        pygame.draw.rect(screen, (50, 60, 74), (x, y, width, height), width=1)
        pygame.draw.rect(screen,
                         (90, 170, 230),
                         (x + 1, y + 1, int((width - 2) * ratio), height - 2))

    def draw_engine_efficiency_bar(self,
                                   pygame,
                                   screen,
                                   x: int,
                                   y: int) -> None:
        efficiency = float(self.info.get("engine_efficiency", 1.0))
        ratio = engine_efficiency_indicator_ratio(efficiency)

        width = 180
        height = 10

        pygame.draw.rect(screen, (50, 60, 74), (x, y, width, height), width=1)
        pygame.draw.rect(screen,
                         (130, 110, 220),
                         (x + 1, y + 1, int((width - 2) * ratio), height - 2))

    def save_trace(self) -> None:
        if not self.trace:
            return

        output_path = Path(self.display_config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plot_trajectory(self.trace,
                        self.env,
                        title="Manual flight",
                        output_path=self.display_config.output_path)

        print("saved:", self.display_config.output_path)
