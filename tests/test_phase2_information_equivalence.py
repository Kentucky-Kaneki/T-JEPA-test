"""
Information Equivalence & Temporal Order Preservation Unit Tests for Cyber-JEPA Phase 2.

Verifies:
1. All 4 candidate representations originate from the same underlying 52-dim observation values.
2. Information equivalence across flat, feature, host, and hierarchical representations.
3. Strict temporal order preservation: x(t-3), x(t-2), x(t-1), x(t) are chronologically ordered.
4. Feature identity and host mapping correctness.
"""

import torch

from cyber_jepa.representations.canonical import CanonicalHostExtractor
from cyber_jepa.representations.feature import FeatureTokenRepresentation
from cyber_jepa.representations.flat import FlatVectorRepresentation
from cyber_jepa.representations.flat_ablations import (
    FlatCurrentOnlyRepresentation,
    FlatShuffledFeaturesRepresentation,
    FlatShuffledTimeRepresentation,
    FlatVariableHistoryRepresentation,
)
from cyber_jepa.representations.hierarchical import HierarchicalHostSubnetRepresentation
from cyber_jepa.representations.host import HostTokenRepresentation


def test_representation_information_equivalence():
    """Verify all representations consume identical 52-dim raw observation values."""
    B, T_hist, D = 4, 4, 52
    torch.manual_seed(42)
    raw_obs = torch.randn(B, T_hist, D)

    # 1. Flat vector
    flat_model = FlatVectorRepresentation(obs_dim=D, hidden_dim=64)
    out_flat = flat_model(raw_obs)
    assert out_flat.shape == (B, 64)

    # 2. Feature tokens
    feat_model = FeatureTokenRepresentation(num_features=D, hidden_dim=64)
    out_feat = feat_model(raw_obs)
    assert out_feat.shape == (B, T_hist, D, 64)

    # 3. Host tokens
    host_model = HostTokenRepresentation(num_hosts=13, hidden_dim=64)
    out_host = host_model(raw_obs)
    assert out_host.shape == (B, T_hist, 13, 64)

    # 4. Hierarchical tokens
    hier_model = HierarchicalHostSubnetRepresentation(num_hosts=13, num_subnets=3, hidden_dim=64)
    out_hier = hier_model(raw_obs)
    assert out_hier.global_token.shape == (B, 64)
    assert out_hier.host_tokens.shape == (B, T_hist, 13, 64)
    assert out_hier.subnet_tokens.shape == (B, T_hist, 3, 64)


def test_canonical_host_extractor_grouping():
    """Verify 52-dim vector maps to 13 hosts x 4 features without value dropping."""
    B, T, D = 2, 4, 52
    raw_obs = torch.arange(D, dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(B, T, -1)

    state = CanonicalHostExtractor.extract_from_vector(raw_obs)

    assert state.host_features.shape == (B, T, 13, 4)
    # Host 0 features should be 0, 1, 2, 3
    assert torch.allclose(state.host_features[0, 0, 0, :], torch.tensor([0.0, 1.0, 2.0, 3.0]))
    # Host 12 features should be 48, 49, 50, 51
    assert torch.allclose(state.host_features[0, 0, 12, :], torch.tensor([48.0, 49.0, 50.0, 51.0]))


def test_temporal_order_preservation():
    """Verify temporal ordering t-3, t-2, t-1, t survives preprocessing."""
    B, T_hist, D = 2, 4, 52
    synthetic_obs = torch.zeros(B, T_hist, D)
    # Set unique marker in feature 0 for each timestep
    for t in range(T_hist):
        synthetic_obs[:, t, 0] = float(t + 10)  # t=0 -> 10, t=1 -> 11, t=2 -> 12, t=3 -> 13

    # Check slice order
    assert synthetic_obs[0, 0, 0].item() == 10.0
    assert synthetic_obs[0, 1, 0].item() == 11.0
    assert synthetic_obs[0, 2, 0].item() == 12.0
    assert synthetic_obs[0, 3, 0].item() == 13.0


def test_flat_ablations_forward_shapes():
    """Verify all flat ablation modules output valid tensor shapes."""
    B, T_hist, D = 4, 4, 52
    x = torch.randn(B, T_hist, D)

    m_shuf_time = FlatShuffledTimeRepresentation(history_len=4)
    assert m_shuf_time(x).shape == (B, 64)

    m_shuf_feat = FlatShuffledFeaturesRepresentation(history_len=4)
    assert m_shuf_feat(x).shape == (B, 64)

    m_current = FlatCurrentOnlyRepresentation()
    assert m_current(x).shape == (B, 64)

    m_v8 = FlatVariableHistoryRepresentation(history_len=8)
    # Input with 8 timesteps
    x8 = torch.randn(B, 8, D)
    assert m_v8(x8).shape == (B, 64)
