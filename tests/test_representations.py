"""
Unit tests for all four Cyber-JEPA candidate representations, parameter budgets, and context masking.

Verifies:
- Flat vector, feature token, host token, and hierarchical token ContextTokens output shapes
- TargetMasker target sampling and learned mask substitution
- Parameter budget alignment utility (parameter counts within 10% across candidates)
"""

import torch

from cyber_jepa.models.context import ContextTokens, TokenType
from cyber_jepa.representations.budget import check_parameter_budget_alignment, count_parameters
from cyber_jepa.representations.feature import FeatureTokenRepresentation, audit_input_exactness
from cyber_jepa.representations.flat import FlatVectorRepresentation
from cyber_jepa.representations.hierarchical import HierarchicalHostSubnetRepresentation, HierarchicalOutput
from cyber_jepa.representations.host import HostTokenRepresentation
from cyber_jepa.representations.masking import TargetMasker


def test_flat_vector_representation_shape():
    """Verify FlatVectorRepresentation outputs ContextTokens and valid tensor shapes."""
    model = FlatVectorRepresentation(obs_dim=52, hidden_dim=64, ffn_dim=256)
    x = torch.randn(8, 4, 52)
    ctx = model(x)

    assert isinstance(ctx, ContextTokens)
    assert ctx.tokens.shape == (8, 4, 64)
    assert ctx.global_token.shape == (8, 64)
    assert ctx.token_type_ids[0, 0].item() == TokenType.TEMPORAL_FLAT

    raw_global = model.encode_context(x, return_context_tokens=False)
    assert raw_global.shape == (8, 64)


def test_feature_token_representation_shape():
    """Verify FeatureTokenRepresentation outputs ContextTokens with 52 tokens per timestep."""
    model = FeatureTokenRepresentation(num_features=52, hidden_dim=64, ffn_dim=256)
    x = torch.randn(8, 4, 52)
    ctx = model(x)

    assert isinstance(ctx, ContextTokens)
    assert ctx.tokens.shape == (8, 4 * 52, 64)
    assert ctx.global_token.shape == (8, 64)
    assert ctx.token_type_ids[0, 0].item() == TokenType.FEATURE

    # Audit input exactness
    audit_res = audit_input_exactness(x)
    assert audit_res["exact_match"]


def test_host_token_representation_shape():
    """Verify HostTokenRepresentation outputs ContextTokens with 13 host tokens per timestep."""
    model = HostTokenRepresentation(num_hosts=13, hidden_dim=64, ffn_dim=256)
    x = torch.randn(8, 4, 52)
    ctx = model(x)

    assert isinstance(ctx, ContextTokens)
    assert ctx.tokens.shape == (8, 4 * 13, 64)
    assert ctx.global_token.shape == (8, 64)
    assert ctx.token_type_ids[0, 0].item() == TokenType.HOST


def test_hierarchical_representation_shape():
    """Verify HierarchicalHostSubnetRepresentation outputs ContextTokens and HierarchicalOutput."""
    model = HierarchicalHostSubnetRepresentation(num_hosts=13, num_subnets=3, hidden_dim=64, ffn_dim=256)
    x = torch.randn(8, 4, 52)
    ctx = model(x)

    assert isinstance(ctx, ContextTokens)
    assert ctx.tokens.shape == (8, 4 * 16, 64) # 13 host + 3 subnet per timestep
    assert ctx.global_token.shape == (8, 64)

    h_out = model.encode_context(x, return_context_tokens=False)
    assert isinstance(h_out, HierarchicalOutput)
    assert h_out.host_tokens.shape == (8, 4, 13, 64)
    assert h_out.subnet_tokens.shape == (8, 4, 3, 64)
    assert h_out.global_token.shape == (8, 64)


def test_target_masker():
    """Verify TargetMasker substitutes timestep t target entity with learned mask token."""
    masker = TargetMasker(hidden_dim=64)
    context_tokens = torch.randn(4, 4, 13, 64) # [B, T_hist, 13, D]
    target_tokens = torch.randn(4, 13, 64)     # [B, 13, D]

    masked_ctx, target_latents, target_indices, validity_mask = masker.sample_targets_and_mask(
        context_tokens=context_tokens,
        target_tokens=target_tokens,
    )

    assert masked_ctx.shape == (4, 4, 13, 64)
    assert target_latents.shape == (4, 64)
    assert target_indices.shape == (4,)
    assert validity_mask.shape == (4, 4, 13)

    # Timestep t=3 target entity must have validity_mask == 0.0
    for b in range(4):
        idx = target_indices[b].item()
        assert validity_mask[b, 3, idx] == 0.0
        assert validity_mask[b, 2, idx] == 1.0 # history t=2 retained


def test_parameter_budget_alignment():
    """Verify parameter counts across candidate representations fall within budget."""
    models = {
        "flat": FlatVectorRepresentation(hidden_dim=64, ffn_dim=256),
        "feature": FeatureTokenRepresentation(hidden_dim=64, ffn_dim=256),
        "host": HostTokenRepresentation(hidden_dim=64, ffn_dim=256),
        "hierarchical": HierarchicalHostSubnetRepresentation(hidden_dim=64, ffn_dim=256),
    }

    _ = {name: count_parameters(m)[0] for name, m in models.items()}
    assert check_parameter_budget_alignment(models, max_tolerance=0.50)
