import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rocket_env.env import RocketAscentEnv
from rocket_env.render import print_rollout_summary, save_rollout_plot
from rocket_env.rollouts import no_thrust_policy, prograde_thrust_policy, radial_thrust_policy
from settings import load_settings


POLICIES = {
    "no-thrust": no_thrust_policy,
    "radial": radial_thrust_policy,
    "prograde": prograde_thrust_policy,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--policy", choices=sorted(POLICIES), default="radial")
    parser.add_argument("--output")
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()

    settings = load_settings(args.config)
    steps = args.steps or settings.display.default_steps
    output = args.output or settings.display.scripted_output_path()
    env = RocketAscentEnv(settings.to_env_config(max_steps=steps))
    fig, trace, info = save_rollout_plot(
        env,
        POLICIES[args.policy],
        output,
        f"{args.policy} scripted rollout",
        max_steps=steps,
    )
    print_rollout_summary(trace, info)
    print("saved:", output)
    fig.clear()


if __name__ == "__main__":
    main()
