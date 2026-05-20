"""Run inference from a saved Tianshou training checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from learning.inference import run_tianshou_checkpoint_episode


def main() -> None:
    args = _parse_args()
    result = run_tianshou_checkpoint_episode(
        checkpoint_path=args.checkpoint,
        settings_path=args.settings,
        output_dir=args.output_dir,
        run_name=args.run_name,
        deterministic=not args.sample,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    print({"metrics": result.metrics,
           "trace_path": result.trace_path,
           "metrics_path": result.metrics_path})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",
                        required=True,
                        help="Path to a saved Tianshou checkpoint .pt file.")
    parser.add_argument("--settings",
                        help="Optional settings artifact JSON. If omitted, checkpoint settings are used.")
    parser.add_argument("--output-dir",
                        default="results/inference",
                        help="Folder for inference trace and metrics JSON.")
    parser.add_argument("--run-name",
                        default="latest",
                        help="File stem for inference artifacts.")
    parser.add_argument("--max-steps",
                        type=int,
                        help="Optional episode step cap. Defaults to the checkpoint training max_steps.")
    parser.add_argument("--seed",
                        type=int,
                        default=0,
                        help="Environment reset seed.")
    parser.add_argument("--sample",
                        action="store_true",
                        help="Sample from the Gaussian policy instead of using the mean action.")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
