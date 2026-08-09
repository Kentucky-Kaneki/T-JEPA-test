"""
Reproducibility Utilities for Cyber-JEPA Phase 3.

Enforces seed determinism across Random, NumPy, PyTorch CPU/CUDA, CuDNN,
and DataLoader worker initialization.
"""

import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Phase3SeedConfig:
    """Separate seed configuration for Phase 3 experimental pipeline."""
    split_seed: int = 42
    cohort_seed: int = 4201
    training_subset_seed: int = 7301
    model_seed: int = 1001
    dataloader_seed: int = 1001
    bootstrap_seed: int = 9901


def set_deterministic_seed(seed: int, warn_only: bool = True) -> torch.Generator:
    """
    Set deterministic seeds across random, numpy, and torch.

    Returns:
        torch.Generator: Seeded PyTorch generator for DataLoaders.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
    except Exception:
        pass

    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id: int) -> None:
    """Worker initialization function for PyTorch DataLoader."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
