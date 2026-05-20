# Tianshou Learning Side Guide

## Purpose

This fork uses Tianshou as the training library and keeps the neural-network
choices visible in this repository.

The learning boundary is:

```text
this repo chooses observations, layers, activations, actor/critic heads, and Gaussian policy settings
Tianshou owns rollout collection, buffers, PPO updates, and trainer orchestration
```

The playable rocket environment remains the same. Manual flight and display
scripts still exercise the normal environment action path.

## Main Files

### Neural Network Input

- [observations.py](./adapters/observations.py) builds the 13-feature input vector.
- [normalization.py](./adapters/normalization.py) contains optional feature scaling helpers.

This is where you change what the NN can see.

### Model Folders

- [model_a](./models/model_a/) is the editable scaffold model.
- [model_b](./models/model_b/) is the empty model shell.
- [model_c](./models/model_c/) is the implemented baseline model.

Each model folder owns its own `mlp.py`, `heads.py`, and `model.py`. This is
where you work on layers, hidden sizes, activation functions, and output heads.

### Tianshou Wiring

- [envs.py](./tianshou/envs.py) wraps the Gymnasium rocket environment into flat NN observations.
- [models.py](./tianshou/models.py) defines shared config/types and tensor helpers.
- [ppo_runner.py](./tianshou/ppo_runner.py) passes the actor and critic into Tianshou PPO.

This is where you choose the NN architecture and give it to Tianshou.

### Action Adapter

- [actions.py](./adapters/actions.py) converts model action outputs into clipped environment command-rate actions.

The policy produces a `model_action`. The environment receives an `env_action`.

## First NN Mental Model

For this fork, read the neural-network path like this:

```text
1. observations.py        -> what the NN sees
2. models/model_x/mlp.py  -> hidden layers
3. models/model_x/heads.py -> Gaussian actor head and value head
4. models/model_x/model.py -> actor and critic modules Tianshou can call
5. tianshou/ppo_runner.py -> Tianshou PPO trains those modules
```

The current input vector is:

```text
obs shape = (batch_size, 13)
```

The 13 input features are:

```text
altitude_scaled
vertical_speed_scaled
tangential_velocity_scaled
fuel_fraction
initial_fuel_scaled
current_throttle_command
current_angle_sin
current_angle_cos
target_altitude_scaled
target_orbital_velocity_scaled
altitude_error_scaled
vertical_speed_error_scaled
tangential_velocity_error_scaled
```

Starter architecture:

```text
actor:
13 -> 64 -> 64 -> Gaussian mean/std for 2 action dimensions

critic:
13 -> 64 -> 64 -> 1 state-value estimate
```

Meaning:

```text
encoded observation -> actor  -> Gaussian action distribution
encoded observation -> critic -> value estimate
```

## Configurable NN Choices

The intended edit point is `NetworkConfig` and `GaussianPolicyConfig` in
`src/learning/tianshou/models.py`.

The named model slots are explicit folders in `src/learning/models/`:

```text
model_a/  scaffold you can edit while keeping the Tianshou interface stable
model_b/  empty shell reserved for your own model
model_c/  implemented 13 -> 64 -> 64 baseline
```

Example:

```python
from learning.tianshou import MODEL_A, GaussianPolicyConfig, NetworkConfig, TianshouPPOConfig

config = TianshouPPOConfig(
    model_name=MODEL_A,
    actor_network=NetworkConfig(hidden_sizes=(128, 128), activation="relu"),
    critic_network=NetworkConfig(hidden_sizes=(128, 64), activation="tanh"),
    gaussian_policy=GaussianPolicyConfig(initial_log_std=-0.5),
)
```

That says:

```text
we choose the NN layers and Gaussian policy setup
Tianshou still performs PPO training
```

## Data Flow

The full learning path is:

```text
environment observation
-> observation adapter
-> Tianshou actor and critic
-> Gaussian action distribution and value estimate
-> Tianshou collector
-> environment step
-> Tianshou buffer
-> Tianshou PPO update
-> optimizer step
```

The important action contract:

```text
model_action != env_action
```

`model_action` is the action produced by the policy distribution.

`env_action` is the clipped physical command passed to `env.step`.

For PPO, log probabilities must describe the model action, not the clipped
environment action.

## Package Map

```text
src/learning/adapters/     observation and action conversion
src/learning/models/       one folder per model, each with mlp, heads, model
src/learning/tianshou/     Tianshou actor, critic, env wrapper, PPO runner
src/learning/checkpoints/  save/load model state
src/learning/evaluation/   run policy episodes without training
```

The old from-scratch training stack is intentionally not part of this fork.
There is no local `learning.algorithms`, `learning.data`, `learning.math`, or
`learning.training` package here. Tianshou owns those responsibilities.

## Useful Commands

From the repo root:

```powershell
cd C:\Git\Tianshou_ascension_rocket
$env:PYTHONPATH = "C:\Git\Tianshou_ascension_rocket\src"
```

Run the Tianshou PPO smoke loop:

```powershell
.\.venv\Scripts\python.exe scripts\training\train_tianshou_ppo.py
```

Run it with an explicit app settings file:

```powershell
.\.venv\Scripts\python.exe scripts\training\train_tianshou_ppo.py --config configs\app.yaml
```

Run it with a training-parameter YAML file:

```powershell
.\.venv\Scripts\python.exe scripts\training\train_tianshou_ppo.py `
  --training-config experiments\model_c\smoke.yaml `
  --run-name yaml_training_smoke
```

The training YAML controls the PPO/model training setup:

```yaml
training:
  model_name: model_c
  max_epoch: 1
  step_per_epoch: 32
  step_per_collect: 32
  batch_size: 16
  learning_rate: 0.0003
  training_display:
    enabled: false
    step_delay_seconds: 0.0
```

CLI flags override the YAML. Saved settings artifacts can be reused as training
input, so a previous run can be repeated or used as the base for a curriculum
step:

```powershell
.\.venv\Scripts\python.exe scripts\training\train_tianshou_ppo.py `
  --training-config results\settings\model_c\yaml_training_smoke.json `
  --run-name repeated_from_saved_settings
```

The training script saves:

```text
results/metrics/<model>/<run-name>.json
results/checkpoints/<model>/<run-name>.pt
results/settings/<model>/<run-name>.json
logs/training/<model>_<run-name>.json
```

Both artifacts include the learner hyperparameters and the settings input used
to build the environment. That means a saved run records not only the model and
PPO values, but also the raw `--config` file contents and the resolved Pydantic
settings.

Use `results/settings/<model>/<run-name>.json` when you want the cleanest record
of the training settings. It contains the raw and parsed app settings file, the
raw and parsed explicit overlay file if one was passed, runtime overrides, and
the final resolved settings passed into the environment. It also contains the
raw training YAML/input, CLI training overrides, and final resolved PPO training
parameters.

Watch the first training environment while Tianshou collects rollouts:

```powershell
.\.venv\Scripts\python.exe scripts\training\train_tianshou_ppo.py `
  --training-config experiments\model_c\smoke.yaml `
  --show-training-display `
  --training-display-delay 0.05
```

This is for inspection only. The delay slows training so the motion is readable.

The training log also records where settings came from:

```text
app_file          values taken from configs/app.yaml
explicit_config   values supplied by --config overlays
runtime_overrides values supplied by CLI/runtime overrides
pydantic_fallback values that came from schema defaults because no file set them
```

Run checkpoint inference:

```powershell
.\.venv\Scripts\python.exe scripts\run_tianshou_policy.py `
  --checkpoint results\checkpoints\model_c\short_test_1_model_c.pt `
  --settings results\settings\model_c\short_test_1_model_c.json `
  --run-name short_test_1_model_c_eval
```

Inference restores the saved actor and critic, uses the same settings artifact
to rebuild the environment, then writes:

```text
results/inference/<run-name>_trace.json
results/inference/<run-name>_metrics.json
```

The default run name is `latest`. Use `--run-name` when you want to keep a
separate experiment result.

Run learning-side tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider `
  tests\test_learning_setup.py `
  tests\test_learning_adapters.py `
  tests\test_learning_models.py `
  tests\test_learning_tianshou.py `
  tests\test_learning_evaluation.py
```

Run the full suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests
```

## Debugging Checklist

Before changing the architecture, check:

- observations are finite
- actor mean and standard deviation are finite
- critic values are finite
- action shapes match the environment action space
- rewards are not accidentally all zero
- Tianshou collector runs at least one episode
- at least one model parameter changes during training

## Stable Shell Rule

Keep the environment and training shell stable while changing one NN variable at
a time.

That makes it possible to tell whether a result came from the observation
vector, the neural network, the Gaussian policy setup, the reward, or PPO.
