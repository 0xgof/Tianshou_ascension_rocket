"""Compatibility wrapper for checkpoint inference."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.evaluation.run_tianshou_policy import main


if __name__ == "__main__":
    main()
