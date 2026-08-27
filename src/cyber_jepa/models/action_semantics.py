"""
Canonical Scenario1b Action Semantics (Phase 4).

Factors out the discrete-action -> (type, host, subnet) resolution table used by
`cyber_jepa.models.predictor.ActionEncoder` so that Phase 4 fusion representations
(`flat_fused.py`, `feature_fused.py`) can resolve the *exact same* semantic fields for
a given action index, without importing or modifying `predictor.py`.

IMPORTANT: this is a byte-for-byte port of
`ActionEncoder._build_scenario1b_action_tables` (predictor.py, Phase 3, frozen/unchanged
for this study). It intentionally preserves that method's existing quirks (e.g. Misinform
actions currently share type-id 0 with Sleep) so that Test 2 (fused) is not accidentally
given richer action semantics than Test 1 (separate encoder, already trained) - the only
variable under study is *where* action information is fused, not *how* rich it is. If
`predictor.py`'s table is ever corrected, mirror the fix here too.
"""

import torch

NUM_ACTIONS = 66
NUM_ACTION_TYPES = 16   # matches ActionEncoder.type_emb vocab size
NUM_HOSTS = 14          # 0 = NONE, 1..13 = Scenario1b host slots
NUM_SUBNETS = 4         # 0 = NONE, 1..3 = subnets (unused by the current table, reserved)


def build_scenario1b_action_tables() -> tuple[list[int], list[int], list[int]]:
    """Build deterministic lookup tables for discrete action indices 0..65.

    Returns (type_ids, host_ids, subnet_ids), each of length `NUM_ACTIONS`.
    """
    types = [0] * 66
    hosts = [0] * 66
    subnets = [0] * 66

    types[0] = 0; hosts[0] = 0; subnets[0] = 0  # Sleep
    types[1] = 1; hosts[1] = 0; subnets[1] = 0  # Monitor

    # 2..17 Analyse
    for idx in range(2, 18):
        types[idx] = 2
        if idx <= 14:
            hosts[idx] = (idx - 2) + 1
        else:
            hosts[idx] = 0

    # 18..33 Remove
    for idx in range(18, 34):
        types[idx] = 3
        if idx <= 30:
            hosts[idx] = (idx - 18) + 1
        else:
            hosts[idx] = 0

    # 34..49 Misinform / Decoy
    for idx in range(34, 50):
        types[idx] = 0
        if idx <= 46:
            hosts[idx] = (idx - 34) + 1
        else:
            hosts[idx] = 0

    # 50..65 Restore
    for idx in range(50, 66):
        types[idx] = 4
        if idx <= 62:
            hosts[idx] = (idx - 50) + 1
        else:
            hosts[idx] = 0

    return types, hosts, subnets


def build_scenario1b_action_lookup_buffers(device: torch.device | None = None) -> dict[str, torch.Tensor]:
    """Return the three lookup tables as torch.long tensors ready for embedding-index use."""
    type_ids, host_ids, subnet_ids = build_scenario1b_action_tables()
    return {
        "type_map": torch.tensor(type_ids, dtype=torch.long, device=device),
        "host_map": torch.tensor(host_ids, dtype=torch.long, device=device),
        "subnet_map": torch.tensor(subnet_ids, dtype=torch.long, device=device),
    }
