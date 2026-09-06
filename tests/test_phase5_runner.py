import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runner():
    return load_script("run_phase5")


@pytest.fixture
def config():
    return yaml.safe_load(Path("configs/phase5/core.yaml").read_text())


def result(name, seed, weight=0.05):
    metrics = {
        "prediction": {"improvement_over_persistence": weight, "changed_smooth_l1": 0.2},
        "geometry": {"effective_rank_fraction": 0.5},
    }
    return {
        "run_id": f"{name}_seed{seed}", "seed": seed, "ratio": "1:1",
        "action_weight": weight, "threshold": 0.25,
        "final_train_loss": 0.5, "final_val_loss": 0.6,
        "validation": metrics, "standard_test": {"score": 1000 - weight},
        "policy_transfer": {"score": 1000 - weight},
        "validation_probe": {"macro_f1": 0.6 + weight},
        "standard_probe": {"macro_f1": 0.5}, "policy_transfer_probe": {"macro_f1": 0.4},
    }


@pytest.fixture
def stage2(runner, config, tmp_path):
    rows = [
        result(f"static50_dynamic50_action{weight:g}", seed, weight)
        for weight in config["stage2"]["action_weights"] for seed in config["seeds"]
    ]
    runner._write_json(tmp_path / "stage2_results.json", {"runs": rows})
    return rows


@pytest.fixture
def sweep(runner, config, tmp_path, stage2, monkeypatch):
    manifest = runner._stage3_manifest(config, "static50_dynamic50", 0.05)
    manifest.update(threshold=0.25, shards_dir="data/shards")
    baselines = runner._stage3_baselines(tmp_path, manifest, (1, 1))
    normalizers = {"mean": [0.0], "std": [1.0]}
    builds = []
    trained = []

    def datasets(shards, splits, transfer, core, normalizers=normalizers):
        builds.append(copy.deepcopy(core))
        return {split: SimpleNamespace(core=core, normalizer_stats=normalizers)
                for split in ("train", "val", "test", "ood")}

    def train(name, ratio, seed, settings, data, threshold, root, device, weight):
        assert ratio == (1, 1)
        assert threshold == 0.25
        assert weight == 0.05
        for dataset in data.values():
            assert dataset.core["horizon"] == settings["core"]["horizon"]
            assert dataset.core["history_len"] == settings["core"]["history_len"]
            assert dataset.normalizer_stats is normalizers
        trained.append((name, seed, copy.deepcopy(settings)))
        return result(name, seed, weight)

    monkeypatch.setattr(runner, "_datasets", datasets)
    monkeypatch.setattr(runner, "_run_one", train)
    data = datasets([], {}, [], config["core"])

    def execute():
        cached = runner._stage3_cached_results(config, manifest, tmp_path)
        runner._run_stage3(config, manifest, baselines, data, [], {}, [], tmp_path, "cpu", cached)

    return SimpleNamespace(manifest=manifest, trained=trained, builds=builds, execute=execute)


def test_reduced_grid_changes_exactly_one_factor(runner, config):
    original = copy.deepcopy(config)
    manifest = runner._stage3_manifest(config, "static50_dynamic50", 0.05)
    assert manifest["seeds"] == [1001, 2003, 3005]
    assert len(config["seeds"]) == 5
    assert manifest["max_new_training_runs"] == 18
    assert {(variant["factor"], variant["value"]) for variant in manifest["variants"].values()} == {
        ("latent_dim", 32), ("history_len", 2), ("history_len", 8),
        ("horizon", 4), ("horizon", 16), ("target_normalization", "batch_center"),
    }
    for variant in manifest["variants"].values():
        changes = [(section, key) for section in ("core", "loss")
                   for key, value in variant[section].items()
                   if value != manifest["baseline"][section][key]]
        assert len(changes) == 1
    assert config == original


def test_duplicate_and_baseline_values_are_skipped(runner, config):
    config["stage3"]["one_factor_at_a_time"]["latent_dim"] = [32, 64, 32]
    config["stage3"]["one_factor_at_a_time"]["horizon"] = [4, 8, 16]
    assert runner._stage3_manifest(config, "static50_dynamic50", 0)["max_new_training_runs"] == 18


@pytest.mark.parametrize("seeds", [[], [1001, 1001], [99]])
def test_invalid_seed_subsets(runner, config, seeds):
    config["stage3"]["seeds"] = seeds
    with pytest.raises(ValueError, match="subset"):
        runner._stage3_manifest(config, "static50_dynamic50", 0.05)


def test_selection_uses_all_five_validation_seeds_only(runner, config, tmp_path, stage2):
    candidate, weight, selection = runner._select_stage3_baseline(config, tmp_path, None, None)
    assert (candidate, weight) == ("static50_dynamic50", 0.05)
    assert selection["seeds"] == config["seeds"]
    for row in stage2:
        if row["action_weight"] == 0.05 and row["seed"] in (4007, 5009):
            row["validation_probe"]["macro_f1"] = 0.0
    runner._write_json(tmp_path / "stage2_results.json", {"runs": stage2})
    assert runner._select_stage3_baseline(config, tmp_path, None, None)[1] == 0.01


def test_incomplete_candidates_are_not_ranked(runner, config, tmp_path, stage2):
    runner._write_json(tmp_path / "stage2_results.json", {"runs": stage2[:-1]})
    assert runner._select_stage3_baseline(config, tmp_path, None, None)[1] == 0.01


def test_selection_is_frozen_and_overrides_accept_zero(runner, config, tmp_path, stage2):
    manifest = runner._stage3_manifest(config, "static50_dynamic50", 0)
    runner._write_json(tmp_path / "stage3_one_factor_manifest.json", manifest)
    assert runner._select_stage3_baseline(config, tmp_path, None, None)[1] == 0
    assert runner._select_stage3_baseline(config, tmp_path, "static50_dynamic50", 0.01)[1] == 0.01
    assert runner._select_stage3_baseline(config, tmp_path, None, 0)[1] == 0


def test_individual_stage2_files_work_without_summary(runner, config, tmp_path, stage2):
    (tmp_path / "stage2_results.json").unlink()
    for row in stage2:
        runner._write_json(tmp_path / f"{row['run_id']}.json", row)
    assert runner._select_stage3_baseline(config, tmp_path, None, None)[1] == 0.05
    manifest = runner._stage3_manifest(config, "static50_dynamic50", 0.05)
    assert len(runner._stage3_baselines(tmp_path, manifest, (1, 1))) == 3


def test_missing_baselines_fail_without_retraining(runner, config, tmp_path):
    with pytest.raises(ValueError, match="No completed Stage 2"):
        runner._select_stage3_baseline(config, tmp_path, None, None)
    manifest = runner._stage3_manifest(config, "static50_dynamic50", 0.05)
    with pytest.raises(ValueError, match="Missing completed Stage 2"):
        runner._stage3_baselines(tmp_path, manifest, (1, 1))


def test_sweep_naming_aggregation_and_rerun(sweep, tmp_path):
    before = (tmp_path / "stage2_results.json").read_bytes()
    sweep.execute()
    assert len(sweep.trained) == 18
    assert len(sweep.builds) == 5
    payload = json.loads((tmp_path / "stage3_results.json").read_text())
    assert payload["complete"]
    assert payload["manifest"]["pending_training_runs"] == 0
    assert len(payload["runs"]) == 21
    assert len(payload["aggregate_across_seeds"]) == 7
    for name in sweep.manifest["variants"]:
        summary = json.loads((tmp_path / name / "results.json").read_text())
        assert summary["complete"]
        assert len(summary["runs"]) == 3
        for seed in sweep.manifest["seeds"]:
            assert (tmp_path / f"{name}_seed{seed}.json").exists()
    sweep.execute()
    assert len(sweep.trained) == 18
    assert len(sweep.builds) == 5
    assert (tmp_path / "stage2_results.json").read_bytes() == before
    assert not (tmp_path / "transition_threshold.json").exists()
    assert not list(tmp_path.rglob("*.tmp"))


def test_interrupted_sweep_reuses_finished_fits(runner, sweep, tmp_path, monkeypatch):
    train = runner._run_one
    def interrupt(*args):
        if len(sweep.trained) == 2:
            raise RuntimeError("simulated interruption")
        return train(*args)
    monkeypatch.setattr(runner, "_run_one", interrupt)
    with pytest.raises(RuntimeError, match="interruption"):
        sweep.execute()
    payload = json.loads((tmp_path / "stage3_results.json").read_text())
    assert not payload["complete"]
    assert payload["manifest"]["pending_training_runs"] == 16
    monkeypatch.setattr(runner, "_run_one", train)
    sweep.execute()
    assert len(sweep.trained) == 18


def test_changed_config_or_incomplete_cache_is_not_overwritten(runner, config, sweep, tmp_path):
    sweep.execute()
    name = next(iter(sweep.manifest["variants"]))
    path = tmp_path / f"{name}_seed1001.json"
    row = json.loads(path.read_text())
    del row["validation_probe"]
    runner._write_json(path, row)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="refusing to overwrite"):
        runner._stage3_cached_results(config, sweep.manifest, tmp_path)
    assert path.read_bytes() == before
    sweep.manifest["baseline"]["core"]["batch_size"] = 32
    with pytest.raises(ValueError, match="baseline settings changed"):
        runner._stage3_cached_results(config, sweep.manifest, tmp_path)


def test_legacy_cached_results_are_reused(runner, config, sweep, tmp_path):
    sweep.execute()
    saved = copy.deepcopy(sweep.manifest)
    saved.pop("shards_dir")
    runner._write_json(tmp_path / "stage3_one_factor_manifest.json", saved)
    cached = runner._stage3_cached_results(config, sweep.manifest, tmp_path)
    assert len(cached) == 18
    variant = next(iter(sweep.manifest["variants"].values()))
    variant["core"]["learning_rate"] = 0.002
    with pytest.raises(ValueError, match="different settings"):
        runner._stage3_cached_results(config, sweep.manifest, tmp_path)


def test_model_receives_ablation_dimensions_and_loss(runner, config, monkeypatch):
    encoder = Mock()
    model = Mock()
    loss = Mock()
    monkeypatch.setitem(sys.modules, "cyber_jepa.models.jepa_phase5", SimpleNamespace(
        Phase5CyberJEPA=model, Phase5LossConfig=loss))
    monkeypatch.setitem(sys.modules, "cyber_jepa.representations.flat", SimpleNamespace(
        FlatVectorRepresentation=encoder))
    manifest = runner._stage3_manifest(config, "static50_dynamic50", 0.05)
    for variant in manifest["variants"].values():
        runner._model(variant["core"], variant["loss"])
        assert encoder.call_args.kwargs["hidden_dim"] == variant["core"]["hidden_dim"]
        assert encoder.call_args.kwargs["history_len"] == variant["core"]["history_len"]
        assert model.call_args.kwargs["max_horizon"] == variant["core"]["horizon"]
        assert model.call_args.kwargs["hidden_dim"] == variant["core"]["hidden_dim"]
        assert loss.call_args.kwargs == variant["loss"]


def test_dataset_builder_preserves_normalizers_and_natural_splits(runner, config, monkeypatch):
    constructors = []
    fitted = {"mean": [1.0], "std": [2.0]}
    def dataset(*args, **kwargs):
        constructors.append((args, kwargs))
        stats = fitted if kwargs.get("fit_normalizers") else kwargs["normalizer_stats"]
        return SimpleNamespace(normalizer_stats=stats)
    monkeypatch.setitem(sys.modules, "cyber_jepa.data.dataset", SimpleNamespace(CyberJEPADataset=dataset))
    splits = {"train": ["training"], "val": ["validation"], "test": ["test"]}
    transfer = ["transfer"]
    baseline = runner._datasets([], splits, transfer, config["core"])
    assert constructors[0][1]["fit_normalizers"]
    assert all(item.normalizer_stats is fitted for item in baseline.values())
    core = {**config["core"], "history_len": 8, "horizon": 16}
    runner._datasets([], splits, transfer, core, fitted)
    for args, kwargs in constructors[4:]:
        assert kwargs["normalizer_stats"] is fitted
        assert kwargs["history_len"] == 8
        assert kwargs["horizon"] == 16
        assert not kwargs.get("fit_normalizers", False)
    assert constructors[5][0][1] == splits["val"]
    assert constructors[6][0][1] == splits["test"]
    assert constructors[7][1]["trajectory_set"] == transfer


def test_completed_main_needs_no_torch_or_shards(runner, config, sweep, tmp_path, monkeypatch):
    sweep.execute()
    monkeypatch.setattr(runner, "_load_config", lambda path: config)
    monkeypatch.setattr(runner, "_datasets", Mock(side_effect=AssertionError("must not load data")))
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(sys, "argv", ["run_phase5.py", "--stage", "stage3", "--output-dir", str(tmp_path)])
    runner.main()
    assert len(sweep.trained) == 18


def test_pipeline_stage3_needs_no_flags_or_collection(monkeypatch):
    pipeline = load_script("run_phase5_pipeline")
    launch = Mock()
    monkeypatch.setattr(pipeline.subprocess, "run", launch)
    monkeypatch.setitem(sys.modules, "run_collection", None)
    monkeypatch.setattr(sys, "argv", ["run_phase5_pipeline.py", "--stage", "stage3"])
    pipeline.main()
    command = launch.call_args.args[0]
    assert command[command.index("--stage") + 1] == "stage3"
    assert "--selected-ratio" not in command
    assert "--selected-action-weight" not in command
    assert launch.call_args.kwargs["check"]


def test_real_pipeline_dry_run_without_training_imports(tmp_path, runner, config):
    rows = [
        result(f"static50_dynamic50_action{weight:g}", seed, weight)
        for weight in config["stage2"]["action_weights"]
        for seed in config["seeds"]
    ]
    runner._write_json(tmp_path / "stage2_results.json", {"runs": rows})
    before = sorted(tmp_path.rglob("*"))
    environment = {**os.environ, "PYTHONPATH": str(Path("src"))}
    completed = subprocess.run([
        sys.executable, "scripts/run_phase5_pipeline.py", "--stage", "stage3",
        "--output-dir", str(tmp_path), "--dry-run",
    ], capture_output=True, text=True, env=environment)
    assert completed.returncode == 0, completed.stderr
    assert '"pending_training_runs": 18' in completed.stdout
    assert "3 baseline results reused" in completed.stdout
    assert sorted(tmp_path.rglob("*")) == before


@pytest.mark.parametrize("stage,extra", [("stage1", []), ("stage2", ["--selected-ratio", "static50_dynamic50"])])
def test_earlier_stage_dry_runs_keep_five_seeds(runner, config, monkeypatch, capsys, stage, extra):
    monkeypatch.setattr(runner, "_load_config", lambda path: config)
    monkeypatch.setattr(sys, "argv", ["run_phase5.py", "--stage", stage, "--dry-run", *extra])
    runner.main()
    assert "seeds=5" in capsys.readouterr().out