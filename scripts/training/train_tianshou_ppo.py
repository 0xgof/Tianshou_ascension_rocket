"""Run a small Tianshou PPO training smoke loop."""

from __future__ import annotations

import argparse
from dataclasses import replace
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from learning.tianshou import MODEL_NAMES, run_tianshou_ppo_smoke
from learning.tianshou.training_settings import (build_tianshou_ppo_config,
                                                build_training_audit,
                                                extract_settings_config,
                                                extract_training_config,
                                                load_training_file,
                                                merge_training_config)
from settings import load_settings_with_audit


def main() -> None:
    args = _parse_args()
    raw_training_file = load_training_file(args.training_config)
    settings_overrides = extract_settings_config(raw_training_file)
    loaded_settings = load_settings_with_audit(args.config,
                                              overrides=settings_overrides)
    project_settings = loaded_settings.settings

    file_training_config = extract_training_config(raw_training_file)
    runtime_overrides = _runtime_training_overrides(args)
    merged_training_config = merge_training_config(file_training_config,
                                                  runtime_overrides)
    config = build_tianshou_ppo_config(merged_training_config,
                                       settings_input=loaded_settings.audit)
    training_audit = build_training_audit(config=config,
                                          config_path=args.training_config,
                                          raw_file=raw_training_file,
                                          file_config=file_training_config,
                                          runtime_overrides=runtime_overrides)
    config = replace(config, training_input=training_audit)

    env_config = project_settings.to_env_config(max_steps=config.max_steps)
    metrics = run_tianshou_ppo_smoke(config, env_config=env_config)

    print(metrics)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",
                        choices=MODEL_NAMES,
                        default=None,
                        help="Named model builder to plug into the Tianshou PPO learner.")
    parser.add_argument("--actor-hidden",
                        default=None,
                        help="Comma-separated actor hidden sizes for model_a.")
    parser.add_argument("--critic-hidden",
                        default=None,
                        help="Comma-separated critic hidden sizes for model_a.")
    parser.add_argument("--actor-activation",
                        default=None,
                        help="Actor activation name for model_a.")
    parser.add_argument("--critic-activation",
                        default=None,
                        help="Critic activation name for model_a.")
    parser.add_argument("--initial-log-std",
                        type=float,
                        default=None,
                        help="Initial Gaussian log standard deviation for model_a.")
    parser.add_argument("--config",
                        help="Optional settings YAML/JSON overlay. configs/app.yaml is always used as the base when present.")
    parser.add_argument("--training-config",
                        help="Optional YAML/JSON file with TianshouPPOConfig training parameters.")
    parser.add_argument("--output-dir",
                        default=None,
                        help="Root folder for metrics and checkpoints.")
    parser.add_argument("--run-name",
                        default=None,
                        help="File stem for saved metrics and checkpoint artifacts.")
    parser.add_argument("--resume-checkpoint-path",
                        default=None,
                        help="Optional training checkpoint to resume actor, critic, and optimizer state from.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train-envs", type=int, default=None)
    parser.add_argument("--test-envs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-epoch", type=int, default=None)
    parser.add_argument("--step-per-epoch", type=int, default=None)
    parser.add_argument("--step-per-collect", type=int, default=None)
    parser.add_argument("--episode-per-test", type=int, default=None)
    parser.add_argument("--repeat-per-collect", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--gae-lambda", type=float, default=None)
    parser.add_argument("--eps-clip", type=float, default=None)
    parser.add_argument("--value-coef", type=float, default=None)
    parser.add_argument("--entropy-coef", type=float, default=None)
    parser.add_argument("--show-training-display",
                        action="store_true",
                        help="Open a Pygame display for the first training environment.")
    parser.add_argument("--training-display-delay",
                        type=float,
                        default=None,
                        help="Seconds to sleep after each displayed training step.")
    parser.add_argument("--training-display-width", type=int, default=None)
    parser.add_argument("--training-display-height", type=int, default=None)
    parser.add_argument("--training-display-fps", type=int, default=None)
    parser.add_argument("--no-save",
                        action="store_true",
                        help="Run training without writing metrics or checkpoints.")

    args = parser.parse_args()

    return args


def _runtime_training_overrides(args: argparse.Namespace) -> dict:
    overrides = {}

    for field_name in [
        "model_name",
        "seed",
        "train_envs",
        "test_envs",
        "max_steps",
        "max_epoch",
        "step_per_epoch",
        "step_per_collect",
        "episode_per_test",
        "repeat_per_collect",
        "batch_size",
        "learning_rate",
        "gamma",
        "gae_lambda",
        "eps_clip",
        "value_coef",
        "entropy_coef",
        "output_dir",
        "run_name",
        "resume_checkpoint_path",
    ]:
        arg_name = "model" if field_name == "model_name" else field_name
        value = getattr(args, arg_name)
        if value is not None:
            overrides[field_name] = value

    actor_network = _network_overrides(args.actor_hidden,
                                       args.actor_activation)
    if actor_network:
        overrides["actor_network"] = actor_network

    critic_network = _network_overrides(args.critic_hidden,
                                        args.critic_activation)
    if critic_network:
        overrides["critic_network"] = critic_network

    if args.initial_log_std is not None:
        overrides["gaussian_policy"] = {"initial_log_std": args.initial_log_std}

    training_display = _training_display_overrides(args)
    if training_display:
        overrides["training_display"] = training_display

    if args.no_save:
        overrides["save_artifacts"] = False

    return overrides


def _network_overrides(hidden_sizes_text: str | None,
                       activation: str | None) -> dict:
    network = {}

    if hidden_sizes_text is not None:
        network["hidden_sizes"] = _parse_hidden_sizes(hidden_sizes_text)

    if activation is not None:
        network["activation"] = activation

    return network


def _training_display_overrides(args: argparse.Namespace) -> dict:
    training_display = {}

    if args.show_training_display:
        training_display["enabled"] = True

    if args.training_display_delay is not None:
        training_display["step_delay_seconds"] = args.training_display_delay

    if args.training_display_width is not None:
        training_display["width"] = args.training_display_width

    if args.training_display_height is not None:
        training_display["height"] = args.training_display_height

    if args.training_display_fps is not None:
        training_display["fps"] = args.training_display_fps

    return training_display


def _parse_hidden_sizes(text: str) -> tuple[int, ...]:
    if not text.strip():
        return ()

    hidden_sizes = tuple(int(value.strip())
                         for value in text.split(",")
                         if value.strip())

    return hidden_sizes


if __name__ == "__main__":
    main()
