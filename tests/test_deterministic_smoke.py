"""
Deterministic Execution Smoke Test for Cyber-JEPA Phase 3.

Verifies that running two independent model instantiations with identical seeds
produces identical initial parameters, batch IDs, epoch losses, checkpoint parameters,
and probe predictions within defined numerical tolerances.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from cyber_jepa.data.collector import collect_shard, get_scenario1b_path
from cyber_jepa.data.dataset import CyberJEPADataset
from cyber_jepa.evaluation.probes import LinearProbeEvaluator
from cyber_jepa.models.jepa import CyberJEPA
from cyber_jepa.representations.flat import FlatVectorRepresentation
from cyber_jepa.training.trainer import JEPATrainer
from cyber_jepa.utils.reproducibility import Phase3SeedConfig, seed_worker, set_deterministic_seed


def _hash_tensor(t: torch.Tensor) -> str:
    """Compute SHA-256 hash of a PyTorch tensor's flat float32 representation."""
    data_bytes = t.detach().cpu().float().numpy().tobytes()
    return hashlib.sha256(data_bytes).hexdigest()[:16]


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_deterministic_execution_smoke(tmp_path: Path):
    """Run two identical 1-epoch training cycles and assert strict deterministic equivalence."""
    scen_path = get_scenario1b_path()
    shard_dir = tmp_path / "shard_smoke"

    # 1. Collect synthetic test shard
    collect_shard(
        scenario_path=scen_path,
        red_policy="bline",
        blue_policy="random",
        episodes=4,
        max_steps=10,
        seed=4201,
        dataset_id="smoke_test",
        output_dir=shard_dir,
    )

    seed_cfg = Phase3SeedConfig(model_seed=1001, dataloader_seed=1001)

    # Function to execute one training run
    def _run_single_smoke(run_dir: Path):
        # Deterministic seed initialization
        gen = set_deterministic_seed(seed_cfg.model_seed)

        test_groups = [
            "group_4201_000",
            "group_4201_001",
            "group_4201_002",
            "group_4201_003",
        ]
        ds = CyberJEPADataset(
            shard_dirs=[shard_dir],
            split_group_set=test_groups,
            horizon=2,
            history_len=4,
            fit_normalizers=True,
        )

        train_loader = DataLoader(
            ds,
            batch_size=4,
            shuffle=True,
            generator=gen,
            worker_init_fn=seed_worker,
        )

        # Batch sequence verification
        batch_ids = []
        for batch in train_loader:
            batch_ids.append(batch["t_context"].tolist())

        # Construct model
        encoder = FlatVectorRepresentation(hidden_dim=32, history_len=4)
        jepa = CyberJEPA(online_encoder=encoder, hidden_dim=32)

        first_param = next(jepa.online_encoder.parameters())
        initial_param_hash = _hash_tensor(first_param)

        optimizer = torch.optim.AdamW(jepa.parameters(), lr=1e-3)
        trainer = JEPATrainer(
            model=jepa,
            train_loader=train_loader,
            val_loader=train_loader,
            optimizer=optimizer,
            scheduler=None,
            run_dir=run_dir,
            device=torch.device("cpu"),
            max_epochs=2,
            min_epochs=1,
            patience=2,
        )

        history = trainer.fit()

        final_param_weight = next(jepa.online_encoder.parameters()).clone().detach()

        # Probe extraction
        evaluator = LinearProbeEvaluator()
        latents, labels, _ = evaluator.extract_latents_and_labels(
            encoder=jepa.online_encoder,
            data_loader=train_loader,
            device=torch.device("cpu"),
        )

        return {
            "batch_ids": batch_ids,
            "initial_param_hash": initial_param_hash,
            "loss_history": history["train_loss"],
            "final_param_weight": final_param_weight,
            "latents": latents,
        }

    # Run twice
    res1 = _run_single_smoke(tmp_path / "run1")
    res2 = _run_single_smoke(tmp_path / "run2")

    # 2. Assert Batch sequence identity
    msg_batch = "Batch ordering mismatch between deterministic runs!"
    assert res1["batch_ids"] == res2["batch_ids"], msg_batch

    # 3. Assert Initial Model Hash identity
    assert res1["initial_param_hash"] == res2["initial_param_hash"], "Initial model hash mismatch!"

    # 4. Assert Epoch Losses within tolerance (1e-5)
    np.testing.assert_allclose(
        res1["loss_history"],
        res2["loss_history"],
        rtol=1e-5,
        atol=1e-5,
        err_msg="Loss history mismatch between deterministic runs!",
    )

    # 5. Assert Final Checkpoint Parameters within tolerance (1e-5)
    np.testing.assert_allclose(
        res1["final_param_weight"].numpy(),
        res2["final_param_weight"].numpy(),
        rtol=1e-5,
        atol=1e-5,
        err_msg="Final parameters mismatch between deterministic runs!",
    )

    # 6. Assert Probe Latents within tolerance (1e-5)
    np.testing.assert_allclose(
        res1["latents"],
        res2["latents"],
        rtol=1e-5,
        atol=1e-5,
        err_msg="Extracted latents mismatch between deterministic runs!",
    )
