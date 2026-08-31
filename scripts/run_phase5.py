"""Stage-gated Phase 5 runner: VICReg, transition balancing, then action contrast.

Stage 1 is the only default experiment. Its ratio selection is validation-only;
natural standard-test and policy-transfer sets are never resampled. Stage 2 must
be invoked explicitly with the selected Stage 1 ratio. Stage 3 is a manifest of
one-factor-at-a-time candidates, preventing accidental factorial sweeps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from cyber_jepa.data.dataset import CyberJEPADataset, generate_group_splits, generate_policy_transfer_splits, verify_dataset_integrity
from cyber_jepa.data.phase5_balancing import TransitionDistanceDataset
from cyber_jepa.evaluation.phase5_metrics import evaluate_phase5_model
from cyber_jepa.evaluation.probes import LinearProbeEvaluator
from cyber_jepa.models.jepa_phase5 import Phase5CyberJEPA, Phase5LossConfig
from cyber_jepa.representations.flat import FlatVectorRepresentation
from cyber_jepa.training.trainer import Trainer
from cyber_jepa.training.phase5_orchestrator import Phase5Orchestrator
from cyber_jepa.utils.reproducibility import seed_worker, set_deterministic_seed


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def _ratio(value: list[int] | None) -> tuple[int, int] | None:
    return None if value is None else (int(value[0]), int(value[1]))


def _model(core: dict[str, Any], loss: dict[str, Any]) -> Phase5CyberJEPA:
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
    probe = LinearProbeEvaluator()
    train_z, train_y, _ = probe.extract_latents_and_labels(model.online_encoder, train_loader, device)
    eval_z, eval_y, _ = probe.extract_latents_and_labels(model.online_encoder, eval_loader, device)
    return probe.train_and_evaluate_probe(train_z, train_y, eval_z, eval_y)


def _run_one(name: str, ratio: tuple[int, int] | None, seed: int, config: dict[str, Any], datasets: dict[str, Any], threshold: float, root: Path, device: torch.device, action_weight: float) -> dict[str, Any]:
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
    parser.add_argument("--selected-ratio", help="Stage 2 only: Stage 1 candidate name, e.g. static50_dynamic50")
    parser.add_argument("--shards-dir", type=Path, default=Path("data/shards"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/phase5_runs"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = _load_config(args.config)
    if args.dry_run:
        print(
            f"Validated Phase 5 config: stage={args.stage}, "
            f"seeds={len(config['seeds'])}, shards_dir={args.shards_dir}. "
            "No shards were collected and no model was trained."
        )
        return
    if args.stage == "stage3":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "stage3_one_factor_manifest.json").write_text(json.dumps(config["stage3"], indent=2), encoding="utf-8")
        print("Wrote one-factor-at-a-time Stage 3 manifest; select a single factor before training.")
        return
    shard_dirs = sorted(path for path in args.shards_dir.iterdir() if (path / "transitions.parquet").exists())
    if len(shard_dirs) != 18:
        raise ValueError(f"Expected the fixed 18-shard collection; found {len(shard_dirs)}")
    verify_dataset_integrity(shard_dirs)
    transitions = pd.concat([pd.read_parquet(path / "transitions.parquet") for path in shard_dirs], ignore_index=True)
    splits = generate_group_splits(sorted(transitions["split_group_id"].unique()))
    transfer = generate_policy_transfer_splits(transitions)["bline_to_meander"]["test"]
    core = config["core"]
    train = CyberJEPADataset(shard_dirs, splits["train"], horizon=core["horizon"], history_len=core["history_len"], fit_normalizers=True)
    normalizers = train.normalizer_stats
    datasets = {"train": train, "val": CyberJEPADataset(shard_dirs, splits["val"], horizon=core["horizon"], history_len=core["history_len"], normalizer_stats=normalizers), "test": CyberJEPADataset(shard_dirs, splits["test"], horizon=core["horizon"], history_len=core["history_len"], normalizer_stats=normalizers), "ood": CyberJEPADataset(shard_dirs, trajectory_set=transfer, horizon=core["horizon"], history_len=core["history_len"], normalizer_stats=normalizers)}
    threshold = TransitionDistanceDataset.calibration_threshold(train)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "transition_threshold.json").write_text(json.dumps({"distance": "rms_flat_observation_delta", "threshold": threshold, "calibration_split": "train", "frozen_before_evaluation": True}, indent=2), encoding="utf-8")
    candidates = config["stage1"]["ratios"] if args.stage == "stage1" else {args.selected_ratio or "": config["stage1"]["ratios"].get(args.selected_ratio or "")}
    if args.stage == "stage2" and not args.selected_ratio or args.stage == "stage2" and candidates.get(args.selected_ratio or "") is None:
        raise ValueError("Stage 2 requires --selected-ratio naming a non-natural Stage 1 candidate.")
    action_weights = [0.0] if args.stage == "stage1" else config["stage2"]["action_weights"]
    manifest = {"stage": args.stage, "threshold": threshold, "candidates": candidates, "action_weights": action_weights, "seeds": config["seeds"]}
    if args.dry_run:
        print(json.dumps(manifest, indent=2)); return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
