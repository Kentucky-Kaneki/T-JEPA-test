"""Stage-gated Phase 5 runner: VICReg, transition balancing, then action contrast.

Stage 1 is the only default experiment. Its ratio selection is validation-only;
natural standard-test and policy-transfer sets are never resampled. Stage 2 must
be invoked explicitly with the selected Stage 1 ratio. Stage 3 trains reduced
one-factor-at-a-time candidates with three seeds, reusing the selected Stage 2
baseline and completed ablation results rather than repeating those fits.

Stage 3 automatically selects a completed Stage 2 baseline using validation
only, then freezes the selection for restarts. Optional --selected-ratio and
--selected-action-weight flags override selection. Keep --output-dir pointed
at the completed Stage 2 results. Baseline values come
from core/loss; stage3.one_factor_at_a_time lists only alternatives. Use the
same core/loss settings and shards used for Stage 2. The shared baseline uses
only the Stage 3 seed subset, with frozen normalizers and transition threshold.
--dry-run prints the planned sweep without loading shards or training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from cyber_jepa.training.phase5_orchestrator import Phase5Orchestrator

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader

    from cyber_jepa.models.jepa_phase5 import Phase5CyberJEPA


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def _ratio(value: list[int] | None) -> tuple[int, int] | None:
    return None if value is None else (int(value[0]), int(value[1]))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def _stage2_rows(root: Path, config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    results_path = root / "stage2_results.json"
    rows = {}
    if results_path.exists():
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        rows = {row["run_id"]: row for row in payload["runs"]}
    for candidate, ratio in config["stage1"]["ratios"].items():
        if ratio is None:
            continue
        for weight in config["stage2"]["action_weights"]:
            for seed in config["seeds"]:
                run_id = f"{candidate}_action{weight:g}_seed{seed}"
                path = root / f"{run_id}.json"
                if run_id not in rows and path.exists():
                    rows[run_id] = json.loads(path.read_text(encoding="utf-8"))
    return rows


def _select_stage3_baseline(config: dict[str, Any], root: Path, candidate: str | None, action_weight: float | None) -> tuple[str, float, dict[str, Any]]:
    if candidate is not None and config["stage1"]["ratios"].get(candidate) is None:
        raise ValueError("Stage 3 baseline must name a non-natural Stage 1 ratio.")
    if action_weight is not None and action_weight not in config["stage2"]["action_weights"]:
        raise ValueError("Stage 3 baseline action weight must be a tested Stage 2 value.")
    if candidate is not None and action_weight is not None:
        return candidate, action_weight, {"source": "explicit_overrides"}
    saved_path = root / "stage3_one_factor_manifest.json"
    if saved_path.exists() and candidate is None and action_weight is None:
        saved = json.loads(saved_path.read_text(encoding="utf-8"))
        if "selected_ratio" in saved and "selected_action_weight" in saved:
            selected = saved["selected_ratio"]
            weight = saved["selected_action_weight"]
            if config["stage1"]["ratios"].get(selected) is None or weight not in config["stage2"]["action_weights"]:
                raise ValueError("Frozen Stage 3 baseline is not in the current configuration.")
            return selected, weight, saved.get("selection", {"source": "saved_manifest"})
    if config["stage3"].get("baseline_selection", "validation_best") != "validation_best":
        raise ValueError("Stage 3 baseline_selection must be validation_best.")
    completed = _stage2_rows(root, config)
    eligible = []
    candidates = {}
    for ratio_name, ratio in config["stage1"]["ratios"].items():
        if ratio is None or candidate is not None and candidate != ratio_name:
            continue
        for weight in config["stage2"]["action_weights"]:
            if action_weight is not None and action_weight != weight:
                continue
            name = f"{ratio_name}_action{weight:g}"
            run_ids = [f"{name}_seed{seed}" for seed in config["seeds"]]
            if not all(run_id in completed for run_id in run_ids):
                continue
            for seed, run_id in zip(config["seeds"], run_ids):
                row = completed[run_id]
                if (row["run_id"] != run_id or row["seed"] != seed
                        or row["ratio"] != f"{ratio[0]}:{ratio[1]}" or row["action_weight"] != weight):
                    raise ValueError(f"Invalid Stage 2 result metadata: {run_id}")
                eligible.append(row)
            candidates[name] = (ratio_name, weight)
    if not eligible:
        raise ValueError(f"No completed Stage 2 candidate with all {len(config['seeds'])} seeds in {root}. Stage 3 will not rerun Stages 1/2.")
    rankings = Phase5Orchestrator.rank_by_validation(eligible)
    if any(not math.isfinite(row["validation_score"]) for row in rankings):
        raise ValueError("Stage 2 validation scores must be finite for automatic selection.")
    selected, weight = candidates[rankings[0]["candidate"]]
    return selected, weight, {"source": "stage2_validation", "seeds": config["seeds"], "ranking": rankings}


def _stage3_manifest(config: dict[str, Any], candidate: str, action_weight: float) -> dict[str, Any]:
    seeds = config["stage3"]["seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds).issubset(config["seeds"]):
        raise ValueError("Stage 3 seeds must be a nonempty, unique subset of the Stage 2 seeds.")
    baseline_name = f"{candidate}_action{action_weight:g}"
    baseline = {"core": dict(config["core"]), "loss": dict(config["loss"])}
    baseline["loss"]["action_weight"] = action_weight
    factors = {
        "latent_dim": ("core", "hidden_dim"),
        "history_len": ("core", "history_len"),
        "horizon": ("core", "horizon"),
        "target_normalization": ("loss", "target_normalization"),
    }
    variants = {}
    for factor, values in config["stage3"]["one_factor_at_a_time"].items():
        if factor not in factors:
            raise ValueError(f"Unsupported Stage 3 factor: {factor}")
        section, key = factors[factor]
        for value in values:
            if value == baseline[section][key]:
                continue
            if factor == "target_normalization":
                if value not in {"none", "batch_center"}:
                    raise ValueError("Stage 3 target_normalization must be none or batch_center.")
            elif isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Stage 3 {factor} values must be positive integers.")
            settings = {"core": dict(baseline["core"]), "loss": dict(baseline["loss"])}
            settings[section][key] = value
            name = f"stage3_{baseline_name}_{factor}_{value}"
            variants[name] = {"factor": factor, "value": value, **settings}
    return {
        "stage": "stage3", "seeds": seeds, "selected_ratio": candidate,
        "selected_action_weight": action_weight,
        "baseline": {"name": baseline_name, **baseline}, "variants": variants,
        "baseline_runs_reused": len(seeds), "max_new_training_runs": len(variants) * len(seeds),
    }


def _stage3_baselines(root: Path, manifest: dict[str, Any], ratio: tuple[int, int]) -> list[dict[str, Any]]:
    results_path = root / "stage2_results.json"
    completed = {}
    if results_path.exists():
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        completed = {row["run_id"]: row for row in payload["runs"]}
    rows = []
    for seed in manifest["seeds"]:
        run_id = f"{manifest['baseline']['name']}_seed{seed}"
        row = completed.get(run_id)
        result_path = root / f"{run_id}.json"
        if row is None and result_path.exists():
            row = json.loads(result_path.read_text(encoding="utf-8"))
        if row is None:
            raise ValueError(f"Missing completed Stage 2 baseline {run_id} in {root}; no baseline will be retrained.")
        if (row["run_id"] != run_id or row["seed"] != seed
                or row["ratio"] != f"{ratio[0]}:{ratio[1]}"
                or row["action_weight"] != manifest["selected_action_weight"]):
            raise ValueError(f"Stage 2 baseline metadata does not match {run_id}.")
        rows.append(row)
    if len({row["threshold"] for row in rows}) != 1 or not math.isfinite(rows[0]["threshold"]):
        raise ValueError("Stage 2 baseline seeds must share a frozen transition threshold.")
    return rows


def _datasets(shard_dirs: list[Path], splits: dict[str, Any], transfer: list[str], core: dict[str, Any], normalizers: Any = None) -> dict[str, Any]:
    from cyber_jepa.data.dataset import CyberJEPADataset

    train = CyberJEPADataset(
        shard_dirs, splits["train"], horizon=core["horizon"], history_len=core["history_len"],
        fit_normalizers=normalizers is None, normalizer_stats=normalizers,
    )
    normalizers = train.normalizer_stats
    datasets = {"train": train}
    for split in ("val", "test"):
        datasets[split] = CyberJEPADataset(shard_dirs, splits[split], horizon=core["horizon"], history_len=core["history_len"], normalizer_stats=normalizers)
    datasets["ood"] = CyberJEPADataset(shard_dirs, trajectory_set=transfer, horizon=core["horizon"], history_len=core["history_len"], normalizer_stats=normalizers)
    return datasets


def _stage3_fingerprint(config: dict[str, Any], manifest: dict[str, Any], variant: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps({
        "core": variant["core"], "loss": variant["loss"], "stage2": config["stage2"],
        "ratio": _ratio(config["stage1"]["ratios"][manifest["selected_ratio"]]),
        "threshold": manifest["threshold"],
    }, sort_keys=True).encode("utf-8")).hexdigest()


def _stage3_cached_results(config: dict[str, Any], manifest: dict[str, Any], root: Path) -> dict[str, dict[str, Any]]:
    saved_path = root / "stage3_one_factor_manifest.json"
    if saved_path.exists():
        saved = json.loads(saved_path.read_text(encoding="utf-8"))
        if saved.get("baseline", {}).get("name") == manifest["baseline"]["name"]:
            if saved["baseline"] != manifest["baseline"]:
                raise ValueError("The Stage 3 baseline settings changed; keep the original Stage 2 core/loss settings.")
            if "shards_dir" in saved and saved["shards_dir"] != manifest["shards_dir"]:
                raise ValueError("The Stage 3 shards directory changed; use the original dataset.")
    completed = {}
    required = {
        "validation", "standard_test", "policy_transfer", "validation_probe",
        "standard_probe", "policy_transfer_probe", "final_train_loss", "final_val_loss",
    }
    ratio = config["stage1"]["ratios"][manifest["selected_ratio"]]
    for name, variant in manifest["variants"].items():
        fingerprint = _stage3_fingerprint(config, manifest, variant)
        for seed in manifest["seeds"]:
            run_id = f"{name}_seed{seed}"
            path = root / f"{run_id}.json"
            if not path.exists():
                continue
            row = json.loads(path.read_text(encoding="utf-8"))
            if (row.get("config_sha256") != fingerprint or row.get("run_id") != run_id
                    or row.get("seed") != seed or row.get("threshold") != manifest["threshold"]
                    or row.get("ratio") != f"{ratio[0]}:{ratio[1]}"
                    or row.get("action_weight") != manifest["selected_action_weight"]
                    or not required.issubset(row)):
                raise ValueError(f"Existing result {path} is incomplete or has different settings; refusing to overwrite it.")
            completed[run_id] = row
    return completed


def _write_stage3_progress(root: Path, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    expected = manifest["baseline_runs_reused"] + manifest["max_new_training_runs"]
    manifest["completed_ablation_runs"] = len(rows) - manifest["baseline_runs_reused"]
    manifest["pending_training_runs"] = expected - len(rows)
    _write_json(root / "stage3_one_factor_manifest.json", manifest)
    payload = {
        "manifest": manifest, "runs": rows, "complete": len(rows) == expected,
        "aggregate_across_seeds": Phase5Orchestrator.aggregate_across_seeds(rows),
        "validation_selection": None,
    }
    _write_json(root / "stage3_results.json", payload)
    for name, variant in manifest["variants"].items():
        variant_rows = [row for row in rows if row["run_id"].rsplit("_seed", 1)[0] == name]
        _write_json(root / name / "results.json", {
            "name": name, "ablation": variant, "baseline": manifest["baseline"]["name"],
            "seeds": manifest["seeds"], "runs": variant_rows,
            "complete": len(variant_rows) == len(manifest["seeds"]),
            "aggregate_across_seeds": Phase5Orchestrator.aggregate_across_seeds(variant_rows),
        })


def _run_stage3(config: dict[str, Any], manifest: dict[str, Any], baseline_rows: list[dict[str, Any]], datasets: dict[str, Any] | None, shard_dirs: list[Path], splits: dict[str, Any], transfer: list[str], root: Path, device: torch.device | None, completed: dict[str, dict[str, Any]]) -> None:
    threshold = manifest["threshold"]
    ratio = _ratio(config["stage1"]["ratios"][manifest["selected_ratio"]])
    rows = [*baseline_rows, *completed.values()]
    _write_stage3_progress(root, manifest, rows)
    for name, variant in manifest["variants"].items():
        variant_config = {**config, "core": variant["core"], "loss": variant["loss"]}
        fingerprint = _stage3_fingerprint(config, manifest, variant)
        variant_datasets = None
        for seed in manifest["seeds"]:
            run_id = f"{name}_seed{seed}"
            result_path = root / f"{run_id}.json"
            if run_id in completed:
                print(f"Reusing {run_id}", flush=True)
                continue
            if datasets is None or device is None:
                raise ValueError("Training datasets and device are required for pending ablations.")
            if variant_datasets is None:
                if variant["factor"] in {"horizon", "history_len"}:
                    variant_datasets = _datasets(shard_dirs, splits, transfer, variant["core"], datasets["train"].normalizer_stats)
                else:
                    variant_datasets = datasets
            print(f"Training {run_id}", flush=True)
            row = _run_one(name, ratio, seed, variant_config, variant_datasets, threshold, root, device, manifest["selected_action_weight"])
            row["config_sha256"] = fingerprint
            _write_json(result_path, row)
            rows.append(row)
            _write_stage3_progress(root, manifest, rows)
    print(f"Stage 3 complete: {root / 'stage3_results.json'}", flush=True)


def _model(core: dict[str, Any], loss: dict[str, Any]) -> Phase5CyberJEPA:
    from cyber_jepa.models.jepa_phase5 import Phase5CyberJEPA, Phase5LossConfig
    from cyber_jepa.representations.flat import FlatVectorRepresentation

    encoder = FlatVectorRepresentation(
        obs_dim=52, hidden_dim=core["hidden_dim"], history_len=core["history_len"],
        num_layers=core["state_encoder_layers"], ffn_dim=core["state_encoder_ffn_dim"],
    )
    return Phase5CyberJEPA(
        encoder, hidden_dim=core["hidden_dim"], max_horizon=core["horizon"],
        aggregator_mode=core["aggregator_mode"],
        ema_momentum_init=core["ema_momentum_init"], ema_momentum_final=core["ema_momentum_final"],
        predictor_num_layers=core["predictor_layers"], predictor_ffn_dim=core["predictor_ffn_dim"],
        action_projection_depth=core["action_projection_depth"], loss_config=Phase5LossConfig(**loss),
    )


def _probe(model: Phase5CyberJEPA, train_loader: DataLoader, eval_loader: DataLoader, device: torch.device) -> dict[str, Any]:
    from cyber_jepa.evaluation.probes import LinearProbeEvaluator

    probe = LinearProbeEvaluator()
    train_z, train_y, _ = probe.extract_latents_and_labels(model.online_encoder, train_loader, device)
    eval_z, eval_y, _ = probe.extract_latents_and_labels(model.online_encoder, eval_loader, device)
    return probe.train_and_evaluate_probe(train_z, train_y, eval_z, eval_y)


def _run_one(name: str, ratio: tuple[int, int] | None, seed: int, config: dict[str, Any], datasets: dict[str, Any], threshold: float, root: Path, device: torch.device, action_weight: float) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    from cyber_jepa.data.phase5_balancing import TransitionDistanceDataset
    from cyber_jepa.evaluation.phase5_metrics import evaluate_phase5_model
    from cyber_jepa.training.trainer import Trainer
    from cyber_jepa.utils.reproducibility import seed_worker, set_deterministic_seed

    core, loss = config["core"], dict(config["loss"])
    loss["action_weight"] = action_weight
    loss.update({
        "action_initial_radius": config["stage2"]["initial_radius"],
        "action_future_margin": config["stage2"]["future_margin"],
        "action_latent_margin": config["stage2"]["latent_margin"],
    })
    generator = set_deterministic_seed(seed)
    balanced = TransitionDistanceDataset(datasets["train"], threshold)
    sampler = balanced.weighted_sampler(ratio, generator)
    train_loader = DataLoader(balanced, batch_size=core["batch_size"], shuffle=sampler is None, sampler=sampler, generator=generator, worker_init_fn=seed_worker)
    val_loader = DataLoader(datasets["val"], batch_size=core["batch_size"], shuffle=False)
    test_loader = DataLoader(datasets["test"], batch_size=core["batch_size"], shuffle=False)
    ood_loader = DataLoader(datasets["ood"], batch_size=core["batch_size"], shuffle=False)
    model = _model(core, loss).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=core["learning_rate"], weight_decay=core["weight_decay"])
    run_dir = root / name / f"seed_{seed}"
    trainer = Trainer(model, train_loader, val_loader, optimizer, torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=core["max_epochs"]), run_dir, device, max_epochs=core["max_epochs"], min_epochs=core["min_epochs"], patience=core["patience"])
    history = trainer.fit()
    return {
        "run_id": f"{name}_seed{seed}", "seed": seed, "ratio": "natural" if ratio is None else f"{ratio[0]}:{ratio[1]}", "action_weight": action_weight,
        "threshold": threshold, "final_train_loss": history["train_loss"][-1], "final_val_loss": history["val_loss"][-1],
        "validation": evaluate_phase5_model(model, val_loader, device, threshold), "standard_test": evaluate_phase5_model(model, test_loader, device, threshold),
        "policy_transfer": evaluate_phase5_model(model, ood_loader, device, threshold),
        "validation_probe": _probe(model, train_loader, val_loader, device), "standard_probe": _probe(model, train_loader, test_loader, device), "policy_transfer_probe": _probe(model, train_loader, ood_loader, device),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase5/core.yaml"))
    parser.add_argument("--stage", choices=["stage1", "stage2", "stage3"], default="stage1")
    parser.add_argument("--selected-ratio", help="Stages 2/3: Stage 1 candidate name, e.g. static50_dynamic50")
    parser.add_argument("--selected-action-weight", type=float, help="Stage 3: optional baseline override")
    parser.add_argument("--shards-dir", type=Path, default=Path("data/shards"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/phase5_runs"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = _load_config(args.config)
    if args.stage == "stage2" and config["stage1"]["ratios"].get(args.selected_ratio) is None:
        parser.error("Stage 2 requires --selected-ratio naming a non-natural Stage 1 candidate.")
    if args.stage == "stage3":
        try:
            candidate, weight, selection = _select_stage3_baseline(config, args.output_dir, args.selected_ratio, args.selected_action_weight)
            manifest = _stage3_manifest(config, candidate, weight)
            manifest["selection"] = selection
            manifest["shards_dir"] = str(args.shards_dir)
            baseline_rows = _stage3_baselines(args.output_dir, manifest, _ratio(config["stage1"]["ratios"][candidate]))
            manifest["threshold"] = baseline_rows[0]["threshold"]
            completed = _stage3_cached_results(config, manifest, args.output_dir)
        except (ValueError, KeyError) as error:
            parser.error(str(error))
        manifest["completed_ablation_runs"] = len(completed)
        manifest["pending_training_runs"] = manifest["max_new_training_runs"] - len(completed)
        print(f"Stage 3 baseline: {candidate}, action_weight={weight:g}; "
              f"{len(baseline_rows)} baseline results reused, {len(completed)} ablations reused, "
              f"{manifest['pending_training_runs']} fits pending.", flush=True)
    if args.dry_run:
        if args.stage == "stage3":
            print(json.dumps(manifest, indent=2))
        print(
            f"Validated Phase 5 config: stage={args.stage}, "
            f"seeds={len(config['stage3']['seeds'] if args.stage == 'stage3' else config['seeds'])}, shards_dir={args.shards_dir}. "
            "No shards were collected and no model was trained."
        )
        return
    if args.stage == "stage3" and not manifest["pending_training_runs"]:
        _run_stage3(config, manifest, baseline_rows, None, [], {}, [], args.output_dir, None, completed)
        print("All Stage 3 ablations are complete; no dataset loading or training needed.")
        return
    if not args.shards_dir.is_dir():
        parser.error(f"Missing existing shards directory: {args.shards_dir}")
    shard_dirs = sorted(path for path in args.shards_dir.iterdir() if (path / "transitions.parquet").exists())
    if len(shard_dirs) != 18:
        raise ValueError(f"Expected the fixed 18-shard collection; found {len(shard_dirs)}")
    import pandas as pd
    import torch

    from cyber_jepa.data.dataset import generate_group_splits, generate_policy_transfer_splits, verify_dataset_integrity
    from cyber_jepa.data.phase5_balancing import TransitionDistanceDataset

    if args.stage == "stage3":
        from cyber_jepa.data.storage import DatasetStorageManager

        for shard_dir in shard_dirs:
            DatasetStorageManager.verify_shard_checksums(shard_dir)
    verify_dataset_integrity(shard_dirs)
    transitions = pd.concat([pd.read_parquet(path / "transitions.parquet") for path in shard_dirs], ignore_index=True)
    splits = generate_group_splits(sorted(transitions["split_group_id"].unique()))
    transfer = generate_policy_transfer_splits(transitions)["bline_to_meander"]["test"]
    core = config["core"]
    datasets = _datasets(shard_dirs, splits, transfer, core)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.stage == "stage3":
        _run_stage3(config, manifest, baseline_rows, datasets, shard_dirs, splits, transfer, args.output_dir, device, completed)
        return
    threshold = TransitionDistanceDataset.calibration_threshold(datasets["train"])
    (args.output_dir / "transition_threshold.json").write_text(json.dumps({"distance": "rms_flat_observation_delta", "threshold": threshold, "calibration_split": "train", "frozen_before_evaluation": True}, indent=2), encoding="utf-8")
    candidates = config["stage1"]["ratios"] if args.stage == "stage1" else {args.selected_ratio or "": config["stage1"]["ratios"].get(args.selected_ratio or "")}
    action_weights = [0.0] if args.stage == "stage1" else config["stage2"]["action_weights"]
    manifest = {"stage": args.stage, "threshold": threshold, "candidates": candidates, "action_weights": action_weights, "seeds": config["seeds"]}
    rows = []
    for candidate, configured_ratio in candidates.items():
        for action_weight in action_weights:
            name = candidate if args.stage == "stage1" else f"{candidate}_action{action_weight:g}"
            for seed in config["seeds"]:
                row = _run_one(name, _ratio(configured_ratio), seed, config, datasets, threshold, args.output_dir, device, action_weight)
                rows.append(row)
                (args.output_dir / f"{row['run_id']}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    payload = {
        "manifest": manifest,
        "runs": rows,
        "aggregate_across_seeds": Phase5Orchestrator.aggregate_across_seeds(rows),
        "validation_selection": Phase5Orchestrator.rank_by_validation(rows) if args.stage == "stage1" else None,
    }
    (args.output_dir / f"{args.stage}_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
