import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rocket_env.env import EnvConfig, RocketAscentEnv
from rocket_env.plots import plot_trajectory
from rocket_env.rollouts import radial_thrust_policy, run_episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/figures/episode.png")
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()

    env = RocketAscentEnv(EnvConfig(max_steps=args.steps))
    trace, _info = run_episode(env, radial_thrust_policy, max_steps=args.steps)
    plot_trajectory(trace, env, title="Episode trajectory", output_path=args.output)
    print("saved:", args.output)


if __name__ == "__main__":
    main()
