import argparse
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rocket_env.live_display import PygameFlightApp
from settings import MANUAL_LOG_DIR, load_settings_with_audit, write_run_log
from settings.run_logging import timestamped_run_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--output")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--overview-size-m", type=float)
    parser.add_argument("--zoom-size-m", type=float)
    parser.add_argument("--log-name")
    args = parser.parse_args()

    overrides = {"display": {}}
    for key in ["width", "height", "fps", "overview_size_m", "zoom_size_m"]:
        value = getattr(args, key)
        if value is not None:
            overrides["display"][key] = value
    loaded_settings = load_settings_with_audit(args.config, overrides=overrides)
    settings = loaded_settings.settings
    steps = args.steps or settings.display.default_steps
    output = args.output or settings.display.manual_flight_output_path()
    display_config = settings.to_display_config(output_path=output, max_steps=steps)
    env_config = settings.to_env_config(max_steps=steps)
    log_name = args.log_name or timestamped_run_name("manual")
    log_payload = {"run_type": "manual",
                   "run_name": log_name,
                   "settings_input": loaded_settings.audit,
                   "cli_args": vars(args),
                   "display_config": asdict(display_config),
                   "env_config": asdict(env_config)}
    log_path = write_run_log(MANUAL_LOG_DIR, log_name, log_payload)
    print(f"Manual run log: {log_path}")

    PygameFlightApp(
        display_config,
        env_config=env_config,
        throttle_step=settings.display.manual_control_throttle_step,
        angle_step=settings.display.manual_control_angle_step,
    ).run()


if __name__ == "__main__":
    main()
