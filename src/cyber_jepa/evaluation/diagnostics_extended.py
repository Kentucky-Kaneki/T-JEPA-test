"""
Extended Latent Geometry & Representation Diagnostics for Cyber-JEPA Phase 2.

Computes:
- Effective rank
- Latent variance (median, min, max)
- Variance explained by PC1, PC2, PC5, PC10
- Full singular value spectrum
- PCA 2D visualization rendering
"""

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from sklearn.decomposition import PCA

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cyber_jepa.evaluation.diagnostics import compute_latent_geometry_diagnostics


def compute_extended_latent_diagnostics(
    latents: torch.Tensor,
    save_spectrum_path: Path | None = None,
) -> dict[str, Any]:
    """
    Computes comprehensive latent geometry metrics including PCA variance explained.
    """
    # Get base diagnostics
    base_diag = compute_latent_geometry_diagnostics(latents)

    if latents.dim() > 2:
        latents = latents.reshape(-1, latents.shape[-1])

    lat_np = latents.detach().cpu().numpy()
    N, D = lat_np.shape

    # Per-dimension variance
    dim_var = np.var(lat_np, axis=0)
    min_var = float(np.min(dim_var))
    max_var = float(np.max(dim_var))
    median_var = float(np.median(dim_var))

    # PCA Singular Value Spectrum & Variance Explained
    pca = PCA(n_components=min(N, D))
    pca.fit(lat_np)

    evr = pca.explained_variance_ratio_
    cum_evr = np.cumsum(evr)

    var_pc1 = float(evr[0]) if len(evr) >= 1 else 0.0
    var_pc2 = float(evr[1]) if len(evr) >= 2 else 0.0
    var_pc5 = float(cum_evr[min(4, len(cum_evr)-1)]) if len(cum_evr) >= 1 else 0.0
    var_pc10 = float(cum_evr[min(9, len(cum_evr)-1)]) if len(cum_evr) >= 1 else 0.0

    singular_values = pca.singular_values_.tolist()

    if save_spectrum_path is not None:
        save_spectrum_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_spectrum_path, pca.singular_values_)

    extended_dict = {
        **base_diag,
        "min_feature_variance": min_var,
        "max_feature_variance": max_var,
        "median_feature_variance": median_var,
        "var_explained_pc1": var_pc1,
        "var_explained_pc2": var_pc2,
        "var_explained_pc5": var_pc5,
        "var_explained_pc10": var_pc10,
        "singular_values_top10": singular_values[:10],
    }

    return extended_dict


def plot_latent_pca_2d(
    latents: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: Path,
    label_name: str = "Critical Server Compromised",
) -> Path:
    """Generate and save a 2D PCA scatter plot colored by oracle target label."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(latents)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=labels,
        cmap="coolwarm",
        alpha=0.6,
        edgecolors="none",
        s=15,
    )
    plt.colorbar(scatter, label=label_name)
    plt.title(f"PCA 2D: {title}\n(PC1: {pca.explained_variance_ratio_[0]:.1%}, PC2: {pca.explained_variance_ratio_[1]:.1%})")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path
