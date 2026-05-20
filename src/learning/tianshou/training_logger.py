"""Persistent diagnostics capture for Tianshou training runs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tianshou.utils.logger.base import BaseLogger


class TrainingDiagnosticsLogger(BaseLogger):
    """Capture train, test, and PPO update metrics emitted by Tianshou."""

    def __init__(self) -> None:
        super().__init__(train_interval=1,
                         test_interval=1,
                         update_interval=1)

        self.records: list[dict[str, Any]] = []
        self.checkpoints: list[dict[str, int | str | None]] = []

    def write(self,
              step_type: str,
              step: int,
              data: dict[str, Any]) -> None:
        record = {"step_type": step_type,
                  "step": int(step),
                  "data": data}

        self.records.append(record)

    def save_data(self,
                  epoch: int,
                  env_step: int,
                  gradient_step: int,
                  save_checkpoint_fn: Callable[[int, int, int], str] | None = None
                  ) -> None:
        checkpoint_path = None
        if save_checkpoint_fn is not None:
            checkpoint_path = save_checkpoint_fn(epoch, env_step, gradient_step)

        checkpoint_record = {"epoch": int(epoch),
                             "env_step": int(env_step),
                             "gradient_step": int(gradient_step),
                             "checkpoint_path": checkpoint_path}

        self.checkpoints.append(checkpoint_record)

    def restore_data(self) -> tuple[int, int, int]:
        return 0, 0, 0

    def payload(self) -> dict[str, Any]:
        diagnostics = {"records": self.records,
                       "checkpoints": self.checkpoints}

        return diagnostics
