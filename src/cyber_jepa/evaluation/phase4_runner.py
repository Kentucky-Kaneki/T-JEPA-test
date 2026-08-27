"""
Phase 4 train + evaluate runner.

Streamlines "train each variant with the patched trainer" and "evaluate each
trained variant, reusing metrics/diagnostics/probes unmodified" into two
functions (`train_variant`, `evaluate_variant`) that `run_phase4.py` just loops
over. Also carries:

  - `run_vram_preflight`  (addition #2 - standalone replacement for extending
    `orchestrator.run_preflight_vram_check`, scoped to the 4 fused variants
    instead of Test 1's 4 `MODEL_REGISTRY` architectures).
  - `build_selection_reports` (addition #4 - keeps "does fusion point/masking
    matter" (Phase-4-only) and "does fusion beat Test 1" (combined) as two
    separate preregistered-rule reports instead of one call that conflates
    them). `selection.py` itself is untouched - this only changes how it's
    *called*.
"""

import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from cyber_jepa.evaluation.diagnostics_extended import compute_extended_latent_diagnostics
from cyber_jepa.evaluation.leakage_probe import run_leakage_probe
from cyber_jepa.evaluation.metrics import compute_action_degradation, compute_predictive_metrics
from cyber_jepa.evaluation.phase4_variants import Phase4Variant
from cyber_jepa.evaluation.probes import LinearProbeEvaluator
from cyber_jepa.evaluation.selection import apply_preregistered_selection_rule
from cyber_jepa.models.jepa_fused import CyberJEPAFused
from cyber_jepa.training.trainer import Trainer


def train_variant(
    variant: Phase4Variant,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    max_epochs: int = 50,
    min_epochs: int = 15,
    patience: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> tuple[CyberJEPAFused, dict[str, Any]]:
    """Build + train one Phase 4 fused variant end to end. Returns (trained_model, history)."""
    model = variant.build()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=opt,
        scheduler=scheduler,
        run_dir=output_dir / variant.name,
        device=device,
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        patience=patience,
    )
    history = trainer.fit()
    return model, history


@torch.no_grad()
def _collect_probe_split(model: CyberJEPAFused, loader: DataLoader, device: torch.device):
    evaluator = LinearProbeEvaluator()
    latents, labels, _ = evaluator.extract_latents_and_labels(model.online_encoder, loader, device)
    return latents, labels


def evaluate_variant(
    model: CyberJEPAFused,
    variant: Phase4Variant,
    train_loader: DataLoader,
    test_loader: DataLoader,
    ood_loader: DataLoader,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Run the full eval suite for one trained variant and pack the result into the
    run-dict shape `selection.apply_preregistered_selection_rule` expects.

    `test_loader` must be the Phase 3 standard test split and `ood_loader` must
    use the same policy-transfer direction as Phase 3.  Keeping them explicit
    prevents an in-distribution score being reported as OOD performance.
    """
    model.eval()

    batch = next(iter(test_loader))
    hist = batch["history_flat"].to(device)
    actions = batch["action_seq"].to(device)
    target = batch["target_flat"].to(device)

    # --- action-zeroed / action-shuffled degradation (unmodified metrics.py) ---
    degradation = compute_action_degradation(model, hist, actions, target)

    # --- latent geometry / collapse diagnostics (unmodified diagnostics_extended.py) ---
    with torch.no_grad():
        _, pred_latent, target_latent = model(hist, actions, target)
    geometry = compute_extended_latent_diagnostics(pred_latent)

    # --- beats-persistence check: predict-no-change baseline vs the model ---
    with torch.no_grad():
        persistence_latent = model.online_encoder.encode_context(hist, actions=None, return_context_tokens=False)
    pred_metrics = compute_predictive_metrics(pred_latent, target_latent, persistence_latents=persistence_latent)
    beats_persistence = pred_metrics.get("improvement_over_persistence", 0.0) > 0.0

    # --- standard action-blind linear probe (unmodified probes.py), same as Test 1 ---
    evaluator = LinearProbeEvaluator()
    train_lat, train_labels = _collect_probe_split(model, train_loader, device)
    test_lat, test_labels = _collect_probe_split(model, test_loader, device)
    probe_result = evaluator.train_and_evaluate_probe(train_lat, train_labels, test_lat, test_labels, is_classification=True)

    ood_lat, ood_labels = _collect_probe_split(model, ood_loader, device)
    ood_probe_result = evaluator.train_and_evaluate_probe(train_lat, train_labels, ood_lat, ood_labels, is_classification=True)

    # --- action-leakage probe: the actual test of the strict/permissive ablation (addition #1) ---
    leakage_result = run_leakage_probe(model.online_encoder, train_loader, test_loader, device)

    t0 = time.time()
    with torch.no_grad():
        model(hist, actions, target)
    inference_latency_ms = (time.time() - t0) * 1000.0 / max(1, hist.shape[0])

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "run_id": f"{variant.name}_seed{seed}",
        "representation": variant.name,
        "seed": seed,
        "is_collapsed": geometry["is_collapsed"],
        "beats_persistence": beats_persistence,
        "action_sensitive": degradation["action_sensitive"],
        "ood_future_compromise_macro_f1": ood_probe_result.get("macro_f1", 0.0),
        "current_compromise_macro_f1": probe_result.get("macro_f1", 0.0),
        "ood_changed_latent_improvement": pred_metrics.get("improvement_over_persistence", 0.0),
        "effective_rank": geometry["effective_rank"],
        "inference_latency_ms": inference_latency_ms,
        "num_trainable_params": num_trainable_params,
        # extra detail kept alongside the selection-rule-shaped fields above
        "action_degradation": degradation,
        "leakage_probe": leakage_result,
        "latent_geometry": geometry,
        "standard_probe": probe_result,
        "ood_probe": ood_probe_result,
    }


def run_vram_preflight(
    variants: list[Phase4Variant],
    device: torch.device,
    vram_limit_gb: float = 3.6,
    batch_size: int = 128,
    history_len: int = 4,
    horizon: int = 4,
) -> dict[str, float]:
    """VRAM preflight scoped to the 4 Phase 4 fused variants (addition #2). Replaces
    extending `orchestrator.run_preflight_vram_check`, which only ever knew about
    Test 1's `MODEL_REGISTRY` (flat/feature/host/hierarchical `CyberJEPA`
    instances) and never touched `CyberJEPAFused`."""
    results: dict[str, float] = {}
    if device.type != "cuda":
        print("CPU device - skipping VRAM preflight.")
        return {v.name: 0.0 for v in variants}

    for variant in variants:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

        model = variant.build().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        hist = torch.randn(batch_size, history_len, 52, device=device)
        actions = torch.randint(0, 66, (batch_size, horizon), device=device)
        target = torch.randn(batch_size, 52, device=device)

        try:
            loss, _, _ = model(hist, actions, target)
            loss.backward()
            opt.step()
            model.update_target_encoder(step=1, total_steps=100)

            peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            results[variant.name] = peak_gb
            print(f"Preflight '{variant.name}': peak VRAM = {peak_gb:.2f} GB (limit {vram_limit_gb} GB)")
            if peak_gb > vram_limit_gb:
                raise MemoryError(f"'{variant.name}' peak VRAM {peak_gb:.2f} GB exceeds {vram_limit_gb} GB")
        except Exception as e:
            print(f"Preflight '{variant.name}' FAILED: {e}")
            results[variant.name] = float("nan")

        torch.cuda.empty_cache()

    return results


def build_selection_reports(
    phase4_runs: list[dict[str, Any]],
    test1_runs: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """Addition #4: two separate preregistered-rule reports instead of one call
    that conflates "does fusion point/masking matter" with "does fusion beat the
    original Test 1 design". `selection.py` is called exactly as-is, twice, with
    different run subsets - it is not modified."""
    output_dir.mkdir(parents=True, exist_ok=True)

    phase4_only = apply_preregistered_selection_rule(
        phase4_runs, output_report_path=output_dir / "selection_phase4_only.json",
    )
    combined = apply_preregistered_selection_rule(
        test1_runs + phase4_runs, output_report_path=output_dir / "selection_test1_vs_test2.json",
    ) if test1_runs else None

    return {"phase4_only": phase4_only, "test1_vs_test2": combined}
