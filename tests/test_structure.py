from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_FOLDERS = [
    "src/rocket_env",
    "src/learning",
    "src/learning/models",
    "src/learning/models/model_a",
    "src/learning/models/model_b",
    "src/learning/models/model_c",
    "src/learning/models/model_d",
    "src/learning/tianshou",
    "src/settings",
    "configs",
    "scripts",
    "scripts/display",
    "scripts/plotting",
    "scripts/validation",
    "notebooks",
    "results",
    "results/checkpoints",
    "results/figures",
    "results/metrics",
    "results/traces",
    "results/reports",
    "tests",
]


def test_required_project_folders_exist():
    for folder in REQUIRED_FOLDERS:
        assert (ROOT / folder).is_dir(), folder


def test_importable_learning_and_settings_packages_exist():
    assert (ROOT / "src/learning/__init__.py").is_file()
    assert (ROOT / "src/learning/models/__init__.py").is_file()
    assert (ROOT / "src/learning/models/model_a/__init__.py").is_file()
    assert (ROOT / "src/learning/models/model_b/__init__.py").is_file()
    assert (ROOT / "src/learning/models/model_c/__init__.py").is_file()
    assert (ROOT / "src/learning/models/model_d/__init__.py").is_file()
    assert (ROOT / "src/learning/tianshou/__init__.py").is_file()
    assert (ROOT / "src/settings/__init__.py").is_file()


def test_models_are_self_contained():
    assert not (ROOT / "src/learning/models/mlp.py").exists()
    assert not (ROOT / "src/learning/models/heads.py").exists()

    for model_name in ["model_a", "model_b", "model_c", "model_d"]:
        model_root = ROOT / "src/learning/models" / model_name
        assert (model_root / "mlp.py").is_file()
        assert (model_root / "heads.py").is_file()
        assert (model_root / "model.py").is_file()


def test_runtime_configs_live_at_repo_root():
    assert (ROOT / "configs").is_dir()
    assert not (ROOT / "src/configs").exists()


def test_results_subfolders_exist():
    for folder in ["checkpoints", "figures", "metrics", "traces", "reports"]:
        assert (ROOT / "results" / folder).is_dir()


def test_existing_script_wrappers_remain_available():
    for script in [
        "watch_random.py",
        "watch_scripted.py",
        "manual_fly.py",
        "plot_episode.py",
        "run_visual_manual_checks.py",
    ]:
        assert (ROOT / "scripts" / script).is_file()


def test_script_implementations_are_grouped_by_purpose():
    expected = [
        "scripts/display/watch_random.py",
        "scripts/display/watch_scripted.py",
        "scripts/display/manual_fly.py",
        "scripts/plotting/plot_episode.py",
        "scripts/validation/run_visual_manual_checks.py",
    ]
    for script in expected:
        assert (ROOT / script).is_file(), script
