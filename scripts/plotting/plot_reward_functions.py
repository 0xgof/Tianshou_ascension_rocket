"""Plot reward and penalty curves used by the rocket environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rocket_env.reward_plots import plot_target_orbit_reward_functions


def main() -> None:
    args = _parse_args()
    output_path = Path(args.output)

    plot_target_orbit_reward_functions(output_path)

    print(output_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",
                        default="results/figures/reward_functions/target_orbit_v1.png",
                        help="Output PNG path.")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
