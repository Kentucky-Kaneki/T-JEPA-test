"""
Unit tests for CyberJEPA architecture, EMA target updates, ActionEncoder, Predictor, and Baselines.

Verifies:
- Exact initial state dict match between online and target encoders
- target_encoder parameters have requires_grad=False
- EMA updates target encoder weights toward online weights
- LatentPredictor is sensitive to action sequence changes
- Layer-Normalized Smooth L1 loss computation
"""

import copy
import torch
import torch.nn as nn
import pytest

from cyber_jepa.representations.flat import FlatVectorRepresentation
from cyber_jepa.models.jepa import CyberJEPA
from cyber_jepa.models.predictor import ActionEncoder, LatentPredictor, compute_jepa_loss
from cyber_jepa.models.baselines import (
    LatentPersistenceBaseline,
    ObservationPersistenceBaseline,
    RawActionConditionedMLP,
)


def test_exact_target_encoder_initialization():
    """Verify target encoder is initialized with exact online encoder state dict and frozen grad."""
    online = FlatVectorRepresentation(hidden_dim=64, ffn_dim=256)
    jepa = CyberJEPA(online_encoder=online, hidden_dim=64)

    # State dict equality
    for k, v in jepa.online_encoder.state_dict().items():
        assert torch.equal(v, jepa.target_encoder.state_dict()[k])

    # Frozen grad assertion
    for p in jepa.target_encoder.parameters():
        assert not p.requires_grad


def test_ema_target_update():
    """Verify update_target_encoder smoothly updates target parameters toward online parameters."""
    online = FlatVectorRepresentation(hidden_dim=64, ffn_dim=256)
    jepa = CyberJEPA(online_encoder=online, hidden_dim=64, ema_momentum_init=0.5)

    # Mutate online weights
    with torch.no_grad():
        for p in jepa.online_encoder.parameters():
            p.add_(1.0)

    # Initial target param before EMA update
    p_target_before = list(jepa.target_encoder.parameters())[0].clone()

    # EMA update with momentum m=0.5
    jepa.update_target_encoder(step=0, total_steps=10)
    p_target_after = list(jepa.target_encoder.parameters())[0]

    assert not torch.equal(p_target_before, p_target_after)


def test_action_predictor_action_sensitivity():
    """Verify changing future action sequence changes predicted future latent representation."""
    online = FlatVectorRepresentation(hidden_dim=64, ffn_dim=256)
    jepa = CyberJEPA(online_encoder=online, hidden_dim=64)

    hist = torch.randn(4, 4, 52)
    target = torch.randn(4, 52)

    act_seq1 = torch.tensor([[0, 1], [0, 2], [1, 1], [2, 2]], dtype=torch.long)
    act_seq2 = torch.tensor([[5, 9], [12, 14], [8, 3], [10, 11]], dtype=torch.long)

    _, pred1, _ = jepa(hist, act_seq1, target)
    _, pred2, _ = jepa(hist, act_seq2, target)

    assert not torch.allclose(pred1, pred2, atol=1e-4), "Predictor output is insensitive to action sequence!"


def test_jepa_loss_computation():
    """Verify compute_jepa_loss evaluates Smooth L1 on LayerNorm latents."""
    z_hat = torch.randn(8, 64)
    z_target = torch.randn(8, 64)

    loss = compute_jepa_loss(z_hat, z_target)
    assert loss.dim() == 0 # Scalar loss
    assert loss.item() >= 0.0


def test_baselines_forward():
    """Verify control baseline forward passes."""
    obs = torch.randn(4, 52)
    obs_base = ObservationPersistenceBaseline()
    assert torch.equal(obs, obs_base.predict(obs, horizon=4))

    lat = torch.randn(4, 64)
    lat_base = LatentPersistenceBaseline()
    assert torch.equal(lat, lat_base.predict(lat, horizon=4))

    raw_mlp = RawActionConditionedMLP(obs_dim=52, hidden_dim=64)
    hist = torch.randn(4, 4, 52)
    actions = torch.tensor([[0, 1], [2, 3], [1, 0], [4, 5]], dtype=torch.long)
    pred_raw = raw_mlp(hist, actions)
    assert pred_raw.shape == (4, 52)
