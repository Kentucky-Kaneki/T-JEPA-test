"""Transition-distance labels and reproducible weighted samplers for Phase 5."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


class TransitionDistanceDataset(Dataset):
    def __init__(self, base: Dataset, threshold: float) -> None:
        self.base, self.threshold = base, threshold
        self.distances = np.asarray([self.distance(base[index]) for index in range(len(base))])
        self.is_dynamic = self.distances > threshold

    @staticmethod
    def distance(sample: dict) -> float:
        history = torch.as_tensor(sample["history_flat"], dtype=torch.float32)
        target = torch.as_tensor(sample["target_flat"], dtype=torch.float32)
        return float(torch.sqrt(torch.mean((history[-1] - target).square())))

    @classmethod
    def calibration_threshold(cls, base: Dataset) -> float:
        values = [cls.distance(base[index]) for index in range(len(base))]
        if not values:
            raise ValueError("Cannot calibrate transition distance from an empty training dataset")
        return float(np.median(values))

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict:
        item = dict(self.base[index])
        item["transition_distance"] = torch.tensor(self.distances[index], dtype=torch.float32)
        item["is_dynamic"] = torch.tensor(self.is_dynamic[index], dtype=torch.bool)
        return item

    def weighted_sampler(self, ratio: tuple[int, int] | None, generator: torch.Generator) -> WeightedRandomSampler | None:
        if ratio is None:
            return None
        static_share, dynamic_share = ratio[0] / sum(ratio), ratio[1] / sum(ratio)
        counts = (max(1, int((~self.is_dynamic).sum())), max(1, int(self.is_dynamic.sum())))
        weights = np.where(self.is_dynamic, dynamic_share / counts[1], static_share / counts[0])
        return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(self), replacement=True, generator=generator)
