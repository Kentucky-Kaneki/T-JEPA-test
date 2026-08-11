"""
Unit tests for CyberJEPA architecture, EMA target updates, ActionEncoder, Predictor, Aggregators, and Baselines.

Verifies:
- Exact initial state dict match between online and target encoders
- target_encoder parameters have requires_grad=False and stay frozen
- Single-frame target encoding (T=1)
- ContextAggregator interventions (legacy_last_step_mean, learned_query_pool, token_preserving_predictor)
- Subsystem parameter accounting
"""

import torch

from cyber_jepa.models.aggregators import LegacyLastStepMean, LearnedQueryPool, TokenPreservingAggregator
from cyber_jepa.models.baselines import (
    LatentPersistenceBaseline,
    ObservationPersistenceBaseline,
    RawActionConditionedMLP,
)
from cyber_jepa.models.jepa import CyberJEPA, count_subsystem_parameters
from cyber_jepa.models.predictor import compute_jepa_loss
from cyber_jepa.representations.flat import FlatVectorRepresentation
from cyber_jepa.representations.feature import FeatureTokenRepresentation


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


def test_single_frame_target_encoding():
    """Verify single-frame target observation (T=1) produces valid target latent [B, hidden_dim]."""
    online = FlatVectorRepresentation(hidden_dim=64, ffn_dim=256)
    jepa = CyberJEPA(online_encoder=online, hidden_dim=64)

    hist = torch.randn(4, 4, 52)
    single_target = torch.randn(4, 52)
    act_seq = torch.tensor([[0, 1], [0, 2], [1, 1], [2, 2]], dtype=torch.long)

    loss, pred_z, target_z = jepa(hist, act_seq, single_target)
    assert target_z.shape == (4, 64)
    assert pred_z.shape == (4, 64)
    assert loss.dim() == 0


def test_aggregator_modes():
    """Verify all three aggregator modes operate properly in CyberJEPA."""
    for mode in ["legacy_last_step_mean", "learned_query_pool", "token_preserving_predictor"]:
        feat_enc = FeatureTokenRepresentation(num_features=52, hidden_dim=64, ffn_dim=256)
        jepa = CyberJEPA(online_encoder=feat_enc, hidden_dim=64, aggregator_mode=mode)

        hist = torch.randn(2, 4, 52)
        target = torch.randn(2, 52)
        act_seq = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)

        loss, pred_z, target_z = jepa(hist, act_seq, target)
        assert pred_z.shape == (2, 64)
        assert target_z.shape == (2, 64)


def test_subsystem_parameter_accounting():
    """Verify parameter counting per subsystem."""
    online = FlatVectorRepresentation(hidden_dim=64, ffn_dim=256)
    jepa = CyberJEPA(online_encoder=online, hidden_dim=64, aggregator_mode="learned_query_pool")
    counts = count_subsystem_parameters(jepa)

    assert "tokenizer_and_encoder" in counts
    assert "aggregator" in counts
    assert "action_encoder" in counts
    assert "total_trainable" in counts
    assert counts["total_trainable"] > 0


def test_jepa_loss_computation():
    """Verify compute_jepa_loss evaluates Smooth L1 on LayerNorm latents."""
    z_hat = torch.randn(8, 64)
    z_target = torch.randn(8, 64)

    loss = compute_jepa_loss(z_hat, z_target)
    assert loss.dim() == 0
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
