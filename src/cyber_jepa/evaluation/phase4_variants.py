"""
Phase 4 Variant Registry - Test 2 (fused action pathway).

Builds the four new conditions confirmed for this round:
    flat_fused_strict, flat_fused_permissive, feature_fused_strict, feature_fused_permissive

against Test 1's already-trained `flat` and `feature_token_predictor` baselines
(passed in, not retrained here), with capacity matched per the confirmed rule - see
`capacity_matching.py`. Also extends `evaluation.selection.PREFERENCE_ORDER`
additively (append-only, in place) so the preregistered selection rule can tie-break
among the new representation names too, without editing `selection.py`.

Everything downstream of `build()` - `metrics.compute_action_degradation`,
`diagnostics.compute_latent_geometry_diagnostics`,
`diagnostics_extended.compute_extended_latent_diagnostics`,
`probes.LinearProbeEvaluator`, `selection.apply_preregistered_selection_rule` - is
reused unmodified; only the model-construction side is new.
"""

from dataclasses import dataclass
from typing import Callable

import torch.nn as nn

from cyber_jepa.models.jepa import CyberJEPA
from cyber_jepa.models.jepa_fused import CyberJEPAFused
from cyber_jepa.models.capacity_matching import (
    compute_test1_action_pathway_budget,
    search_matched_fused_config,
    MatchedFusedConfig,
)
from cyber_jepa.representations.flat_fused import FlatFusedRepresentation
from cyber_jepa.representations.feature_fused import FeatureFusedRepresentation
from cyber_jepa.models.fusion_masking import STRICT, PERMISSIVE
from cyber_jepa.evaluation.selection import PREFERENCE_ORDER

NEW_REPRESENTATIONS = [
    "flat_fused_strict", "flat_fused_permissive",
    "feature_fused_strict", "feature_fused_permissive",
]
for _rep in NEW_REPRESENTATIONS:
    if _rep not in PREFERENCE_ORDER:
        PREFERENCE_ORDER.append(_rep)


@dataclass
class Phase4Variant:
    name: str
    representation: str          # "flat" or "feature"
    attention_mode: str          # "strict" or "permissive"
    build: Callable[[], CyberJEPAFused]
    matched_config: MatchedFusedConfig


def _count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_flat_fused_variant(
    test1_flat_model: CyberJEPA,
    attention_mode: str,
    obs_dim: int = 52,
    hidden_dim: int = 64,
    num_heads: int = 4,
    history_len: int = 4,
    max_horizon: int = 16,
) -> Phase4Variant:
    target_params = compute_test1_action_pathway_budget(test1_flat_model)

    def build_and_count(num_layers: int, ffn_dim: int) -> int:
        enc = FlatFusedRepresentation(
            obs_dim=obs_dim, hidden_dim=hidden_dim, num_heads=num_heads,
            num_layers=num_layers, ffn_dim=ffn_dim, history_len=history_len,
            max_horizon=max_horizon, attention_mode=attention_mode,
        )
        model = CyberJEPAFused(enc, hidden_dim=hidden_dim, max_horizon=max_horizon)
        return _count_trainable_params(model)

    matched = search_matched_fused_config(build_and_count, target_params)

    def build() -> CyberJEPAFused:
        enc = FlatFusedRepresentation(
            obs_dim=obs_dim, hidden_dim=hidden_dim, num_heads=num_heads,
            num_layers=matched.num_layers, ffn_dim=matched.ffn_dim, history_len=history_len,
            max_horizon=max_horizon, attention_mode=attention_mode,
        )
        return CyberJEPAFused(enc, hidden_dim=hidden_dim, max_horizon=max_horizon)

    return Phase4Variant(
        name=f"flat_fused_{attention_mode}",
        representation="flat", attention_mode=attention_mode,
        build=build, matched_config=matched,
    )


def build_feature_fused_variant(
    test1_feature_model: CyberJEPA,
    attention_mode: str,
    obs_dim: int = 52,
    hidden_dim: int = 64,
    num_heads: int = 4,
    history_len: int = 4,
    max_horizon: int = 16,
) -> Phase4Variant:
    target_params = compute_test1_action_pathway_budget(test1_feature_model)

    def build_and_count(num_layers: int, ffn_dim: int) -> int:
        enc = FeatureFusedRepresentation(
            obs_dim=obs_dim, hidden_dim=hidden_dim, num_heads=num_heads,
            num_layers=num_layers, ffn_dim=ffn_dim, history_len=history_len,
            max_horizon=max_horizon, attention_mode=attention_mode,
        )
        model = CyberJEPAFused(enc, hidden_dim=hidden_dim, max_horizon=max_horizon)
        return _count_trainable_params(model)

    matched = search_matched_fused_config(build_and_count, target_params)

    def build() -> CyberJEPAFused:
        enc = FeatureFusedRepresentation(
            obs_dim=obs_dim, hidden_dim=hidden_dim, num_heads=num_heads,
            num_layers=matched.num_layers, ffn_dim=matched.ffn_dim, history_len=history_len,
            max_horizon=max_horizon, attention_mode=attention_mode,
        )
        return CyberJEPAFused(enc, hidden_dim=hidden_dim, max_horizon=max_horizon)

    return Phase4Variant(
        name=f"feature_fused_{attention_mode}",
        representation="feature", attention_mode=attention_mode,
        build=build, matched_config=matched,
    )


def build_all_phase4_variants(
    test1_flat_model: CyberJEPA,
    test1_feature_model: CyberJEPA,
    **model_kwargs: int,
) -> list[Phase4Variant]:
    """Builds all four confirmed Test 2 conditions, each capacity-matched against its
    own Test 1 counterpart (flat_fused variants matched to the flat baseline,
    feature_fused variants matched to the feature_token_predictor baseline)."""
    return [
        build_flat_fused_variant(test1_flat_model, STRICT, **model_kwargs),
        build_flat_fused_variant(test1_flat_model, PERMISSIVE, **model_kwargs),
        build_feature_fused_variant(test1_feature_model, STRICT, **model_kwargs),
        build_feature_fused_variant(test1_feature_model, PERMISSIVE, **model_kwargs),
    ]
