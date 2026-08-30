import torch
from torch.utils.data import Dataset

from cyber_jepa.data.phase5_balancing import TransitionDistanceDataset
from cyber_jepa.models.jepa_phase5 import vicreg_covariance_loss, vicreg_variance_loss


class _ToyDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        history = torch.zeros(4, 2)
        target = torch.zeros(2) if index < 2 else torch.ones(2)
        return {"history_flat": history, "target_flat": target, "action_seq": torch.tensor([index])}


def test_transition_distance_balancing_labels_and_sampler():
    dataset = TransitionDistanceDataset(_ToyDataset(), threshold=0.5)
    assert dataset.is_dynamic.tolist() == [False, False, True, True]
    assert dataset.weighted_sampler((1, 1), torch.Generator().manual_seed(1)) is not None


def test_vicreg_regularizers_penalize_collapsed_latents():
    collapsed = torch.zeros(8, 4)
    diverse = torch.randn(8, 4)
    assert vicreg_variance_loss(collapsed, 1.0) > vicreg_variance_loss(diverse, 1.0)
    assert vicreg_covariance_loss(collapsed) == 0.0
