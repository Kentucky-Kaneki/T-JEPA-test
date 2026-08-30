"""Phase 5 separate-action JEPA objective with VICReg regularisation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from cyber_jepa.models.jepa import CyberJEPA
from cyber_jepa.models.predictor import TargetSpec


@dataclass(frozen=True)
class Phase5LossConfig:
    variance_weight: float = 1.0
    covariance_weight: float = 0.04
    variance_target_std: float = 1.0
    action_weight: float = 0.0
    action_initial_radius: float = 0.10
    action_future_margin: float = 0.20
    action_latent_margin: float = 0.20
    target_normalization: str = "none"


def _normalise(prediction: torch.Tensor, target: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "none":
        return prediction, target
    if mode == "l2":
        return F.normalize(prediction, dim=-1), F.normalize(target, dim=-1)
    if mode == "batch_center":
        return prediction - prediction.mean(0, keepdim=True), target - target.mean(0, keepdim=True)
    raise ValueError(f"Unknown target_normalization: {mode}")


def vicreg_variance_loss(latents: torch.Tensor, target_std: float) -> torch.Tensor:
    if latents.shape[0] < 2:
        return latents.new_zeros(())
    return F.relu(target_std - torch.sqrt(latents.var(0, unbiased=False) + 1e-4)).mean()


def vicreg_covariance_loss(latents: torch.Tensor) -> torch.Tensor:
    count, dim = latents.shape
    if count < 2:
        return latents.new_zeros(())
    centered = latents - latents.mean(0, keepdim=True)
    covariance = centered.T @ centered / (count - 1)
    off_diagonal = covariance - torch.diag(torch.diagonal(covariance))
    return off_diagonal.square().sum() / dim


def observed_future_action_contrast_loss(
    history_obs: torch.Tensor, target_obs: torch.Tensor, action_seq: torch.Tensor,
    prediction: torch.Tensor, initial_radius: float, future_margin: float, latent_margin: float,
) -> torch.Tensor:
    """Hinge separation for observed-future-divergent comparable pairs only."""
    if prediction.shape[0] < 2:
        return prediction.new_zeros(())
    initial_distance = torch.cdist(history_obs[:, -1], history_obs[:, -1])
    future_distance = torch.cdist(target_obs, target_obs)
    different_action = action_seq[:, 0, None].ne(action_seq[:, 0][None, :])
    upper = torch.triu(torch.ones_like(initial_distance, dtype=torch.bool), diagonal=1)
    valid = (initial_distance <= initial_radius) & (future_distance >= future_margin) & different_action & upper
    if not valid.any():
        return prediction.new_zeros(())
    return F.relu(latent_margin - torch.cdist(prediction, prediction)[valid]).mean()


class Phase5CyberJEPA(CyberJEPA):
    """Phase 5 retains the separate state/action/predictor architecture."""

    def __init__(self, *args: object, loss_config: Phase5LossConfig | None = None, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.loss_config = loss_config or Phase5LossConfig()
        self.last_loss_terms: dict[str, float] = {}

    def forward(self, history_obs: torch.Tensor, action_seq: torch.Tensor, target_obs: torch.Tensor, target_spec: TargetSpec | None = None, host_known_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, prediction, target = super().forward(history_obs, action_seq, target_obs, target_spec, host_known_mask)
        pred_for_loss, target_for_loss = _normalise(prediction, target, self.loss_config.target_normalization)
        jepa = F.smooth_l1_loss(F.layer_norm(pred_for_loss, pred_for_loss.shape[-1:]), F.layer_norm(target_for_loss, target_for_loss.shape[-1:]))
        variance = vicreg_variance_loss(prediction, self.loss_config.variance_target_std)
        covariance = vicreg_covariance_loss(prediction)
        action = observed_future_action_contrast_loss(history_obs, target_obs, action_seq, prediction, self.loss_config.action_initial_radius, self.loss_config.action_future_margin, self.loss_config.action_latent_margin) if self.loss_config.action_weight else prediction.new_zeros(())
        total = jepa + self.loss_config.variance_weight * variance + self.loss_config.covariance_weight * covariance + self.loss_config.action_weight * action
        self.last_loss_terms = {"jepa": float(jepa.detach()), "variance": float(variance.detach()), "covariance": float(covariance.detach()), "action": float(action.detach()), "total": float(total.detach())}
        return total, prediction, target
