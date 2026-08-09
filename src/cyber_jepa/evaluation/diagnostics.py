"""
Latent Geometry & Representation Collapse Diagnostics for Cyber-JEPA.

Computes per-dimension standard deviation, covariance spectrum, effective rank,
pairwise cosine similarity, and evaluates representation collapse constraints.
"""

from typing import Any
import numpy as np
import torch
import torch.nn.functional as F


def compute_latent_geometry_diagnostics(latents: torch.Tensor) -> dict[str, Any]:
    """
    Computes latent space geometry metrics and collapse indicators.

    Input: latents tensor [N, D]
    Returns: dict of geometry metrics and collapse flag
    """
    if latents.dim() > 2:
        latents = latents.reshape(-1, latents.shape[-1])

    N, D = latents.shape
    device = latents.device

    lat_np = latents.detach().cpu().numpy()

    # 1. Per-dimension standard deviation
    dim_std = np.std(lat_np, axis=0)
    median_std = float(np.median(dim_std))
    near_constant_frac = float(np.mean(dim_std < 0.01))

    # 2. Covariance Spectrum & Effective Rank
    centered = lat_np - np.mean(lat_np, axis=0, keepdims=True)
    cov = (centered.T @ centered) / max(1, N - 1)
    singular_values = np.linalg.svd(cov, compute_uv=False)

    s_sum = singular_values.sum()
    if s_sum > 1e-12:
        p = singular_values / s_sum
        p = p[p > 1e-12]
        entropy = -float(np.sum(p * np.log(p)))
        eff_rank = float(np.exp(entropy))
    else:
        eff_rank = 1.0

    eff_rank_frac = eff_rank / D

    # 3. Pairwise Cosine Similarity
    norm_lat = F.normalize(latents.detach(), dim=-1)
    # Subsample pairwise for speed if N is large
    if N > 1000:
        idx = torch.randperm(N)[:1000]
        norm_lat = norm_lat[idx]

    sim_matrix = norm_lat @ norm_lat.T # [1000, 1000]
    # Zero out diagonal
    diag_mask = torch.eye(sim_matrix.shape[0], device=device, dtype=torch.bool)
    sim_matrix.masked_fill_(diag_mask, 0.0)
    mean_pairwise_cos_sim = float(sim_matrix.sum().item() / max(1, sim_matrix.numel() - sim_matrix.shape[0]))

    # 4. Representation Collapse Condition (Section 12.3)
    is_collapsed = bool(median_std < 0.01 or eff_rank_frac < 0.10)

    return {
        "median_std": median_std,
        "near_constant_fraction": near_constant_frac,
        "effective_rank": eff_rank,
        "effective_rank_fraction": eff_rank_frac,
        "mean_pairwise_cosine_sim": mean_pairwise_cos_sim,
        "singular_values_top5": singular_values[:5].tolist(),
        "is_collapsed": is_collapsed,
    }
