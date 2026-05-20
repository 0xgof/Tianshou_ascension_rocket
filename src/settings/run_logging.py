"""JSON run logging helpers for manual and training sessions."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGS_ROOT = PROJECT_ROOT / "logs"
MANUAL_LOG_DIR = LOGS_ROOT / "manual"
TRAINING_LOG_DIR = LOGS_ROOT / "training"


def timestamped_run_name(prefix: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{prefix}_{timestamp}"

    return run_name


def write_run_log(log_dir: str | Path,
                  run_name: str,
                  payload: dict[str, Any]) -> Path:
    target_dir = Path(log_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    log_path = target_dir / f"{run_name}.json"
    log_path.write_text(json.dumps(_json_safe(payload), indent=2),
                        encoding="utf-8")

    return log_path


def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    return value
