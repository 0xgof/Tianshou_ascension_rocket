"""Pydantic settings schema for the rocket DRL project."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rocket_env.env import EnvConfig
from rocket_env.live_display import DisplayConfig
from rocket_env.physics import PhysicsConfig


class PhysicsSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    planet_radius: float = Field(default=6_371_000.0, gt=0.0)
    mu: float = Field(default=3.986004418e14, gt=0.0)
    g0: float = Field(default=9.80665, gt=0.0)
    rho0: float = Field(default=1.225, ge=0.0)
    sea_level_pressure: float = Field(default=101_325.0, gt=0.0)
    scale_height: float = Field(default=8_500.0, gt=0.0)
    max_thrust: float = Field(default=7_800_000.0, gt=0.0)
    max_fuel_burn_rate: float = Field(default=1_508.2420932599366, ge=0.0)
    sea_level_relative_efficiency: float = Field(default=1.0, gt=0.0, le=1.0)
    vacuum_relative_efficiency: float = Field(default=0.88, gt=0.0, le=1.0)
    max_thrust_angle: float = Field(default=3.141592653589793, gt=0.0)
    drag_coefficient: float = Field(default=0.5, ge=0.0)
    reference_area: float = Field(default=10.0, gt=0.0)
    dt: float = Field(default=1.0, gt=0.0)
    enable_atmosphere: bool = True
    enable_drag: bool = True
    min_radius: float = Field(default=1.0, gt=0.0)

    def to_physics_config(self) -> PhysicsConfig:
        return PhysicsConfig(**self.model_dump())


class EnvironmentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dry_mass: float = Field(default=110_000.0, gt=0.0)
    payload_mass: float = Field(default=15_000.0, gt=0.0)
    initial_fuel: float = Field(default=600_000.0, ge=0.0)
    initial_altitude: float = Field(default=0.0, ge=0.0)
    target_altitude: float = Field(default=200_000.0, ge=0.0)
    max_steps: int = Field(default=1_000, gt=0)
    altitude_tolerance: float = Field(default=5_000.0, gt=0.0)
    vertical_speed_tolerance: float = Field(default=50.0, gt=0.0)
    tangential_velocity_tolerance: float = Field(default=100.0, gt=0.0)
    throttle_change_per_second: float = Field(default=0.5, gt=0.0, le=1.0)
    angle_change_per_second: float = Field(default=5.0, gt=0.0, le=180.0)
    terminate_on_fuel_empty: bool = False

    def to_env_config(self,
                      physics: PhysicsConfig,
                      reward_name: str) -> EnvConfig:
        return EnvConfig(physics=physics,
                         reward_name=reward_name,
                         **self.model_dump())


class DisplaySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_steps: int = Field(default=300, gt=0)
    width: int = Field(default=1100, gt=0)
    height: int = Field(default=820, gt=0)
    fps: int = Field(default=30, gt=0)
    manual_simulation_rate_hz: float = Field(default=5.0, gt=0.0)
    figure_width: float = Field(default=10.0, gt=0.0)
    figure_height: float = Field(default=6.0, gt=0.0)
    output_dir: str = "results/figures"
    manual_flight_output_name: str = "manual_flight.png"
    random_output_name: str = "random_rollout.png"
    scripted_output_name: str = "scripted_rollout.png"
    overview_size_m: float = Field(default=200_000.0, gt=0.0)
    zoom_size_m: float = Field(default=300.0, gt=0.0)
    rocket_height_m: float = Field(default=50.0, gt=0.0)
    karman_line_m: float = Field(default=100_000.0, ge=0.0)
    manual_control_throttle_step: float = Field(default=0.01, gt=0.0, le=1.0)
    manual_control_angle_step: float = Field(default=1.0, gt=0.0, le=90.0)

    @field_validator(
        "output_dir",
        "manual_flight_output_name",
        "random_output_name",
        "scripted_output_name",
    )
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Value must not be empty.")
        return value

    def output_path(self, output_name: str) -> str:
        return str(Path(self.output_dir) / output_name)

    def manual_flight_output_path(self) -> str:
        return self.output_path(self.manual_flight_output_name)

    def random_output_path(self) -> str:
        return self.output_path(self.random_output_name)

    def scripted_output_path(self) -> str:
        return self.output_path(self.scripted_output_name)

    def to_display_config(
        self,
        output_path: Optional[str] = None,
        max_steps: Optional[int] = None,
    ) -> DisplayConfig:
        return DisplayConfig(
            width=self.width,
            height=self.height,
            fps=self.fps,
            manual_simulation_rate_hz=self.manual_simulation_rate_hz,
            overview_size_m=self.overview_size_m,
            zoom_size_m=self.zoom_size_m,
            rocket_height_m=self.rocket_height_m,
            karman_line_m=self.karman_line_m,
            output_path=output_path or self.manual_flight_output_path(),
            max_steps=max_steps or self.default_steps,
        )


class ObservationScalingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference_initial_fuel: float = Field(default=400_000.0, gt=0.0)
    reference_target_altitude: float = Field(default=200_000.0, gt=0.0)


class RewardSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected_reward: str = "altitude_gain_v1"

    @field_validator("selected_reward")
    @classmethod
    def non_empty_reward_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Reward name must not be empty.")
        return value


class LearningSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation_scaling: ObservationScalingSettings = Field(
        default_factory=ObservationScalingSettings
    )


class ProjectSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    physics: PhysicsSettings = Field(default_factory=PhysicsSettings)
    environment: EnvironmentSettings = Field(default_factory=EnvironmentSettings)
    reward: RewardSettings = Field(default_factory=RewardSettings)
    display: DisplaySettings = Field(default_factory=DisplaySettings)
    learning: LearningSettings = Field(default_factory=LearningSettings)

    def to_physics_config(self) -> PhysicsConfig:
        return self.physics.to_physics_config()

    def to_env_config(self, max_steps: Optional[int] = None) -> EnvConfig:
        env_settings = self.environment
        if max_steps is not None:
            env_settings = env_settings.model_copy(update={"max_steps": max_steps})
        env_config = env_settings.to_env_config(
            physics=self.to_physics_config(),
            reward_name=self.reward.selected_reward,
        )

        return env_config

    def to_display_config(
        self,
        output_path: Optional[str] = None,
        max_steps: Optional[int] = None,
    ) -> DisplayConfig:
        return self.display.to_display_config(output_path=output_path, max_steps=max_steps)
