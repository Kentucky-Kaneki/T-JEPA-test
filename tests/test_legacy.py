"""
Regression tests for legacy Adult Income T-JEPA reference implementation.

Verifies Section 2.3 and 14.4 constraints:
Existing legacy Adult Income modules (model.py, trainer.py, data.py, downstream.py, run.py)
must remain intact, importable, and functional.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def test_legacy_modules_importable():
    """Verify legacy Adult Income modules are importable without errors."""
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))

    import data as legacy_data
    import downstream as legacy_downstream
    import model as legacy_model
    import trainer as legacy_trainer

    assert hasattr(legacy_model, "TJEPA")
    assert hasattr(legacy_trainer, "pretrain_tjepa")
    assert hasattr(legacy_data, "load_adult")
    assert hasattr(legacy_downstream, "DownstreamMLP")


def test_legacy_model_forward_pass():
    """Verify legacy TJEPA model can execute a forward pass on synthetic tabular batch."""
    import model as legacy_model

    # Instantiate legacy TJEPA model: 2 numerical features (dim 1, 1) and 2 categorical (dim 3, 5)
    tjepa = legacy_model.TJEPA(
        feature_dims=[1, 1, 3, 5],
        hidden_dim=32,
        num_heads=2,
        num_layers=2,
        ffn_dim=64,
        num_reg_tokens=1,
    )

    B = 4
    x1 = torch.randn(B, 1)
    x2 = torch.randn(B, 1)
    x3 = F.one_hot(torch.tensor([0, 1, 2, 0]), num_classes=3).float()
    x4 = F.one_hot(torch.tensor([0, 1, 2, 3]), num_classes=5).float()

    x_batch = [x1, x2, x3, x4]

    # Encode context & target
    loss = tjepa(x_batch)
    assert loss.dim() == 0
    assert not torch.isnan(loss)
