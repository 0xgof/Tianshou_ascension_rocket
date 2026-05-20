import importlib.util
import json
from pathlib import Path

import pytest

from learning.inference import load_tianshou_policy, run_tianshou_checkpoint_episode
from learning.tianshou import MODEL_C


@pytest.mark.skipif(importlib.util.find_spec("tianshou") is None,
                    reason="Tianshou is not installed in this environment.")
def test_tianshou_checkpoint_inference_round_trip(tmp_path):
    from learning.tianshou import TianshouPPOConfig, run_tianshou_ppo_smoke

    training_config = TianshouPPOConfig(max_steps=4,
                                        step_per_epoch=4,
                                        step_per_collect=4,
                                        batch_size=4,
                                        model_name=MODEL_C,
                                        output_dir=str(tmp_path),
                                        run_name="inference_unit")
    training_metrics = run_tianshou_ppo_smoke(training_config)
    checkpoint_path = training_metrics["checkpoint_path"]
    settings_path = training_metrics["settings_path"]

    loaded_policy = load_tianshou_policy(checkpoint_path)
    assert loaded_policy.config.model_name == MODEL_C

    result = run_tianshou_checkpoint_episode(
        checkpoint_path=checkpoint_path,
        settings_path=settings_path,
        output_dir=tmp_path / "inference",
        run_name="inference_unit_eval",
        max_steps=4,
    )

    assert result.metrics["steps"] >= 1.0
    assert Path(result.trace_path).is_file()
    assert Path(result.metrics_path).is_file()

    metrics_payload = json.loads(Path(result.metrics_path).read_text(encoding="utf-8"))
    assert metrics_payload["metrics"]["steps"] == result.metrics["steps"]
