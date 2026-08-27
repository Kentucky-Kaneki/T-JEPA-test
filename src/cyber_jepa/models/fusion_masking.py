"""
Attention Masking for Early-Fusion State+Action Sequences (Phase 4, Test 2).

The neutral-world-model principle (architecture doc SS13-14) requires that the
*context* representation not be told which action is about to happen - only the
Predictor should combine state + action. When state and action tokens share a single
self-attention stack, that separation has to be enforced with an explicit mask rather
than being "free" (as it is in Test 1's two-module design).

Two masking modes are exposed as the confirmed Test 2 masking ablation:
    "strict"     - state/global tokens CANNOT attend to action tokens (preserves the
                   neutral-C_t property of the original architecture).
    "permissive" - full bidirectional attention (simplest fused design, no guarantee
                   of a neutral context representation).

Action tokens may always attend to state/global tokens (they need to "read" context to
predict its consequence) and to each other, in both modes.
"""

import torch

STRICT = "strict"
PERMISSIVE = "permissive"
FUSION_ATTENTION_MODES = (STRICT, PERMISSIVE)

NEG_INF = float("-inf")


def build_fusion_attention_mask(
    state_len: int,
    action_len: int,
    mode: str,
    device: torch.device | None = None,
) -> torch.Tensor | None:
    """
    Build a static [L, L] additive attention mask for a sequence laid out as
    [state_tokens (length state_len, includes the global/reg token)] +
    [action_tokens (length action_len)].

    Returns None for "permissive" (no masking needed - callers can skip the `mask=`
    kwarg entirely, which also skips PyTorch's masked-attention codepath for speed).
    Returns a float mask with 0.0 for allowed and -inf for blocked (query_row, key_col)
    pairs for "strict".
    """
    if mode not in FUSION_ATTENTION_MODES:
        raise ValueError(f"Unknown fusion attention mode: {mode!r}, expected one of {FUSION_ATTENTION_MODES}")

    if mode == PERMISSIVE:
        return None

    total_len = state_len + action_len
    mask = torch.zeros((total_len, total_len), device=device)
    # Block state/global rows (queries) from attending to action columns (keys).
    mask[:state_len, state_len:] = NEG_INF
    return mask
