"""
Unit and integration tests for Trainer, Checkpointing, and SweepOrchestrator.

Verifies:
- CPU forward/backward training epoch execution
- 32-window overfit test (loss decreases over 15 epochs)
- Atomic checkpoint save and reload yielding identical evaluation outputs
- Pre-flight VRAM profiling and manifest initialization
"""

from pathlib import Path
import torch
import torch.nn as nn
import pytest
from torch.utils.data import DataLoader, TensorDataset

from cyber_jepa.representations.flat import FlatVectorRepresentation
from cyber_jepa.representations.host import HostTokenRepresentation
from cyber_jepa.models.jepa import CyberJEPA
from cyber_jepa.training.trainer import Trainer
from cyber_jepa.training.orchestrator import SweepOrchestrator


def test_trainer_cpu_step(tmp_path: Path):
    """Verify single CPU training epoch execution."""
    encoder = FlatVectorRepresentation(hidden_dim=64, ffn_dim=256)
    jepa = CyberJEPA(online_encoder=encoder, hidden_dim=64)

    dummy_hist = torch.randn(16, 4, 52)
    dummy_act = torch.randint(0, 66, (16, 4))
    dummy_target = torch.randn(16, 52)
    dataset = TensorDataset(dummy_hist, dummy_act, dummy_target)

    # Wrap in dict dataset loader
    def collate_fn(batch):
        return {
            "history_flat": torch.stack([b[0] for b in batch]),
            "action_seq": torch.stack([b[1] for b in batch]),
            "target_flat": torch.stack([b[2] for b in batch]),
        }

    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    opt = torch.optim.AdamW(jepa.parameters(), lr=1e-3)
    device = torch.device("cpu")

    trainer = Trainer(
        model=jepa,
        train_loader=loader,
        val_loader=loader,
        optimizer=opt,
        scheduler=None,
        run_dir=tmp_path / "run_cpu",
        device=device,
        max_epochs=2,
        min_epochs=1,
        accum_steps=2,
    )

    t_loss = trainer.train_epoch(epoch=1)
    v_loss = trainer.evaluate()

    assert t_loss >= 0.0
    assert v_loss >= 0.0


def test_32_window_overfit(tmp_path: Path):
    """Verify 32-window overfit test: loss decreases over 15 epochs without NaNs."""
    torch.manual_seed(42)
    encoder = HostTokenRepresentation(hidden_dim=64, ffn_dim=256)
    jepa = CyberJEPA(online_encoder=encoder, hidden_dim=64)

    dummy_hist = torch.randn(32, 4, 52)
    dummy_act = torch.randint(0, 66, (32, 2))
    dummy_target = torch.randn(32, 52)

    def collate_fn(batch):
        return {
            "history_flat": torch.stack([b[0] for b in batch]),
            "action_seq": torch.stack([b[1] for b in batch]),
            "target_flat": torch.stack([b[2] for b in batch]),
        }

    dataset = TensorDataset(dummy_hist, dummy_act, dummy_target)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    opt = torch.optim.AdamW(jepa.parameters(), lr=1e-3)
    device = torch.device("cpu")

    trainer = Trainer(
        model=jepa,
        train_loader=loader,
        val_loader=loader,
        optimizer=opt,
        scheduler=None,
        run_dir=tmp_path / "run_overfit",
        device=device,
        max_epochs=15,
        min_epochs=15,
        accum_steps=1,
    )

    loss_epoch1 = trainer.train_epoch(epoch=1)
    for ep in range(2, 16):
        loss_curr = trainer.train_epoch(epoch=ep)

    assert not torch.isnan(torch.tensor(loss_curr))
    assert loss_curr < loss_epoch1, f"Overfit loss did not decrease: initial={loss_epoch1:.4f}, final={loss_curr:.4f}"


def test_checkpoint_save_and_load_reproducibility(tmp_path: Path):
    """Verify saving best.pt checkpoint and reloading yields identical evaluation outputs."""
    encoder = FlatVectorRepresentation(hidden_dim=64, ffn_dim=256)
    jepa1 = CyberJEPA(online_encoder=encoder, hidden_dim=64)

    dummy_hist = torch.randn(8, 4, 52)
    dummy_act = torch.randint(0, 66, (8, 2))
    dummy_target = torch.randn(8, 52)

    def collate_fn(batch):
        return {
            "history_flat": torch.stack([b[0] for b in batch]),
            "action_seq": torch.stack([b[1] for b in batch]),
            "target_flat": torch.stack([b[2] for b in batch]),
        }

    dataset = TensorDataset(dummy_hist, dummy_act, dummy_target)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    opt = torch.optim.AdamW(jepa1.parameters(), lr=1e-3)
    device = torch.device("cpu")
    run_dir = tmp_path / "run_ckpt"

    trainer1 = Trainer(
        model=jepa1,
        train_loader=loader,
        val_loader=loader,
        optimizer=opt,
        scheduler=None,
        run_dir=run_dir,
        device=device,
        max_epochs=1,
    )

    val_loss1 = trainer1.evaluate()
    trainer1.save_checkpoint(run_dir / "best.pt", epoch=1, val_loss=val_loss1)

    # Reload model from best.pt
    encoder2 = FlatVectorRepresentation(hidden_dim=64, ffn_dim=256)
    jepa2 = CyberJEPA(online_encoder=encoder2, hidden_dim=64)
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    jepa2.online_encoder.load_state_dict(ckpt["online_encoder"])
    jepa2.target_encoder.load_state_dict(ckpt["target_encoder"])
    jepa2.action_encoder.load_state_dict(ckpt["action_encoder"])
    jepa2.predictor.load_state_dict(ckpt["predictor"])

    trainer2 = Trainer(
        model=jepa2,
        train_loader=loader,
        val_loader=loader,
        optimizer=opt,
        scheduler=None,
        run_dir=run_dir,
        device=device,
        max_epochs=1,
    )

    val_loss2 = trainer2.evaluate()
    assert abs(val_loss1 - val_loss2) < 1e-6, "Reloaded model evaluation loss mismatched!"


def test_orchestrator_preflight_and_manifest(tmp_path: Path):
    """Verify SweepOrchestrator manifest initialization and CPU pre-flight profiling."""
    sweep_cfg_path = Path("configs/sweeps/core_grid.yaml")
    assert sweep_cfg_path.exists()

    orchestrator = SweepOrchestrator(sweep_cfg_path, output_dir=tmp_path / "sweep_run")
    assert (tmp_path / "sweep_run" / "sweep_manifest.json").exists()

    results = orchestrator.run_preflight_vram_check(device=torch.device("cpu"))
    assert set(results.keys()) == {"flat", "feature", "host", "hierarchical"}
