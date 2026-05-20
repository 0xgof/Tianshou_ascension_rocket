# Tianshou Ascension Rocket

This fork keeps the playable rocket environment from `RL_ascension_rocket` and
adds a Tianshou-managed training path. The environment, observation adapters,
and PyTorch actor/critic modules remain visible in this repository; Tianshou is
used for the training loop, collector, replay buffer, and PPO update machinery.

## Repository Layout

This project uses a `src/` Python package layout.

```text
src/rocket_env/   physics, environment, display, plotting, rollouts
src/learning/     adapters, PyTorch NN parts, Tianshou training, evaluation
src/settings/     Pydantic settings ingestion
configs/          app, environment, and display configuration files
experiments/      named training profiles and experiment recipes
scripts/          command entry points grouped by purpose
tests/            unit and structure tests
results/          generated artifacts
```

## Launching the Display

Create and use the local virtual environment:

```powershell
cd path\to\Tianshou_ascension_rocket
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run these commands from the repository root:

```powershell
cd path\to\Tianshou_ascension_rocket
$env:PYTHONPATH = "$PWD\src"
```

## Settings

The app uses Pydantic settings from `src/settings/`.

Default launch behavior is:

```text
explicit --config path
else configs/app.yaml if present
else hardcoded Pydantic defaults
```

Edit `configs/app.yaml` to change the default settings used when launching
scripts without `--config`. If that file is missing, the app still runs from
hardcoded defaults in the Pydantic models.

Run logs are written under `logs/`:

```text
logs/manual/    manual display session logs
logs/training/  Tianshou training run logs
```

Each log records the resolved settings plus a source audit:

```text
app_file          values taken from configs/app.yaml
explicit_config   values supplied by --config overlays
runtime_overrides values supplied by CLI/runtime overrides
pydantic_fallback values that came from schema defaults because no file set them
```

Use an explicit config file when you want a separate launch profile:

```powershell
.\.venv\Scripts\python.exe scripts\manual_fly.py --config configs\display_manual.yaml
```

### Manual Keyboard Flight

```powershell
.\.venv\Scripts\python.exe scripts\manual_fly.py
```

This opens a Pygame window for live interaction.

The window is split into two panels:

```text
left  - 200 km x 200 km local ascent view with flat ground and the Karman line
right - 300 m x 300 m rocket-centered zoom view
```

Controls:

```text
P            start, pause, or resume simulation
Up / W       increase throttle
Down / S     decrease throttle
Left / A / <  increase thrust pitch angle
Right / D / > decrease thrust pitch angle
Space        cut throttle
R            reset episode
Esc          exit
```

Throttle changes use one-percentage-point steps by default, so you can build
thrust gradually while the rocket rests on the launch pad.

The app starts paused so it does not immediately fall and crash. Click the Pygame window, press `W` or `Up` a few times to add throttle, then press `P` to start the simulation.

The manual display sends keyboard input through the normal environment action path:

```text
keyboard -> command-rate action -> env.step(action) -> display
```

The action format is:

```text
throttle rate command: [-1, 1]
angle rate command:    [-1, 1]
```

The environment applies those rate commands gradually to the physical throttle
and angle commands.

Angle convention:

```text
0 deg   horizontal/prograde
90 deg  vertical/radial-out
<0 deg  below the local horizontal
```

## Tianshou PPO Smoke Training

The Tianshou fork keeps the NN parts explicit:

```text
src/learning/tianshou/envs.py        wraps raw env observations into NN vectors
src/learning/tianshou/models.py      defines shared config/types only
src/learning/tianshou/ppo_runner.py  wires those modules into Tianshou PPO
src/learning/models/model_a/         editable scaffold model
src/learning/models/model_b/         empty model shell
src/learning/models/model_c/         implemented baseline model
```

The intended learning boundary is:

```text
you choose layers, activations, and Gaussian policy settings
Tianshou runs PPO, collection, replay buffers, and trainer orchestration
```

### How The Model Enters Tianshou

The custom model is passed into Tianshou in
`src/learning/tianshou/ppo_runner.py`.

The wiring is:

```text
NetworkConfig + GaussianPolicyConfig
-> selected model folder
-> actor and critic PyTorch modules
-> PPOPolicy(actor=actor, critic=critic, optim=optimizer, dist_fn=...)
-> Collector and OnpolicyTrainer
```

In other words:

```text
this repo builds the neural networks
Tianshou receives those networks as actor and critic modules
Tianshou trains them with PPO
```

Each model folder defines its own actor, critic, MLP, and heads. The actor
outputs Gaussian distribution parameters. The `_independent_normal(...)`
function in `ppo_runner.py` tells Tianshou how to convert those parameters into
a Torch distribution for sampling, log probabilities, entropy, and PPO updates.

### Model Files

Each model lives in its own folder under `src/learning/models/`.

```text
model_a/
  mlp.py
  heads.py
  model.py

model_b/
  mlp.py
  heads.py
  model.py

model_c/
  mlp.py
  heads.py
  model.py
```

The runner selects a model with `TianshouPPOConfig.model_name`. You can also
pass both a custom actor and a custom critic directly into
`run_tianshou_ppo_smoke(...)`; if you do that, the named model registry is
bypassed.

Example architecture configuration:

```python
from learning.tianshou import MODEL_A, GaussianPolicyConfig, NetworkConfig, TianshouPPOConfig

config = TianshouPPOConfig(
    model_name=MODEL_A,
    actor_network=NetworkConfig(hidden_sizes=(128, 128), activation="relu"),
    critic_network=NetworkConfig(hidden_sizes=(128, 64), activation="tanh"),
    gaussian_policy=GaussianPolicyConfig(initial_log_std=-0.5),
)
```

Run a tiny PPO smoke loop:

```powershell
.\.venv\Scripts\python.exe scripts\train_tianshou_ppo.py
```

Use an explicit settings file for the training environment:

```powershell
.\.venv\Scripts\python.exe scripts\train_tianshou_ppo.py --config configs\app.yaml
```

Select a model explicitly:

```powershell
.\.venv\Scripts\python.exe scripts\train_tianshou_ppo.py --model model_c
.\.venv\Scripts\python.exe scripts\train_tianshou_ppo.py --model model_a --actor-hidden 128,128 --actor-activation relu
```

Use a training YAML file for reproducible PPO parameters:

```powershell
.\.venv\Scripts\python.exe scripts\train_tianshou_ppo.py `
  --training-config experiments\model_c\smoke.yaml `
  --run-name yaml_training_smoke
```

The training YAML uses this shape:

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

CLI flags override the YAML values. A saved settings artifact can also be fed
back into training:

```powershell
.\.venv\Scripts\python.exe scripts\train_tianshou_ppo.py `
  --training-config results\settings\model_c\<run-name>_<timestamp>.json `
  --run-name repeated_from_saved_settings
```

To watch the first training environment while PPO is collecting rollouts:

```powershell
.\.venv\Scripts\python.exe scripts\train_tianshou_ppo.py `
  --training-config experiments\model_c\smoke.yaml `
  --show-training-display `
  --training-display-delay 0.05
```

The display is intentionally off by default. It slows training down and should
be used for inspection, not benchmark runs.

Each training run saves metrics, model state, resolved settings, diagnostics,
and figures by default:

```text
results/metrics/<model>/<run-name>_<timestamp>.json
results/checkpoints/<model>/<run-name>_<timestamp>.pt
results/settings/<model>/<run-name>_<timestamp>.json
results/traces/<model>/<run-name>_<timestamp>_training_diagnostics.json
results/figures/<model>/<run-name>_<timestamp>/
```

The metrics JSON and checkpoint both include:

```text
TianshouPPOConfig hyperparameters
raw settings input from --config
resolved Pydantic settings used to build the environment
trainer metrics returned by Tianshou
figure and diagnostics artifact paths
```

The per-run figure folder can contain episode altitude, throttle, angle, reward,
reward-function, and model-metric plots. Timestamps are included in artifact
names so repeated runs do not overwrite older output.

The settings JSON is the clearest source for reproducing a training run. It
contains the raw text and parsed values from `configs/app.yaml`, the raw text
and parsed values from any explicit `--config` overlay, runtime overrides, and
the final resolved settings passed into the environment. It also contains the
raw training YAML/input, runtime training overrides, and resolved PPO training
parameters.

Use `--run-name` to choose the stable prefix for a run:

```powershell
.\.venv\Scripts\python.exe scripts\train_tianshou_ppo.py --model model_c --run-name first_baseline
```

Use `--no-save` for a temporary run that should not write artifacts.

This is not intended to produce a good policy yet. It verifies that Tianshou can
collect from the rocket environment, call the custom PyTorch actor and critic,
and run one PPO update.

### Checkpoint Inference

Run a saved policy checkpoint without training:

```powershell
.\.venv\Scripts\python.exe scripts\run_tianshou_policy.py `
  --checkpoint results\checkpoints\model_c\<run-name>_<timestamp>.pt `
  --settings results\settings\model_c\<run-name>_<timestamp>.json `
  --run-name policy_eval
```

Inference writes:

```text
results/inference/<run-name>_trace.json
results/inference/<run-name>_metrics.json
```

By default inference uses the actor's Gaussian mean action. Pass `--sample` to
sample from the Gaussian policy instead.

### Generate Display Plots

Matplotlib is still used for static plots and report artifacts.

Random rollout:

```powershell
.\.venv\Scripts\python.exe scripts\watch_random.py --steps 300 --output results\figures\random_rollout.png
```

Radial thrust scripted rollout:

```powershell
.\.venv\Scripts\python.exe scripts\watch_scripted.py --policy radial --steps 300 --output results\figures\radial_rollout.png
```

Pitched/prograde scripted rollout:

```powershell
.\.venv\Scripts\python.exe scripts\watch_scripted.py --policy prograde --steps 300 --output results\figures\prograde_rollout.png
```

Visual manual-check artifact generation:

```powershell
.\.venv\Scripts\python.exe scripts\run_visual_manual_checks.py
```

This writes plots and a report under:

```text
results\visual_manual_checks\
```

## Test Command

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\test_structure.py tests\test_physics.py tests\test_env.py tests\test_display.py
```

Full test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests
```
