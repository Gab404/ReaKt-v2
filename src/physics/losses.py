"""
src/physics/losses.py
=====================
Physics-informed loss functions for the Logi-PINN model.

The physics regulariser enforces two ODEs:
  - Logistic growth (biomass):   ΔX ≈ r · X · (1 − X/X_max) · Δt
  - Luedeking-Piret (penicillin): ΔP ≈ α · ΔX + β · X · Δt
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pi_lstm import BioreactorLogiPINN


def compute_pinn_loss(
    model:          BioreactorLogiPINN,
    x_batch:        torch.Tensor,   # (B, T, input_size)
    y_batch:        torch.Tensor,   # (B, T, 2)  [bio_norm, pen_norm]
    dt_n:           float,          # normalised time step
    lambda_physics: float = 0.0,
    lambda_r_neg:   float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Full PI-LSTM loss: data MSE + physics residual + growth-rate penalty.

    Parameters
    ----------
    model          : BioreactorLogiPINN
    x_batch        : normalised input sequences  (B, T, D)
    y_batch        : normalised target sequences (B, T, 2)
    dt_n           : normalised dt  (= dt_physical / time_range)
    lambda_physics : weight for physics residual term
    lambda_r_neg   : weight for r ≥ 0 penalty

    Returns
    -------
    loss      : scalar total loss
    loss_dict : breakdown for logging
    """
    pred = model(x_batch)   # (B, T, 2)

    # ── Data MSE (NaN-masked) ─────────────────────────────────────────────────
    valid_bio = ~torch.isnan(y_batch[..., 0])
    valid_pen = ~torch.isnan(y_batch[..., 1])

    l_data_bio = F.mse_loss(pred[..., 0][valid_bio], y_batch[..., 0][valid_bio]) \
                 if valid_bio.any() else pred.new_tensor(0.0)
    l_data_pen = F.mse_loss(pred[..., 1][valid_pen], y_batch[..., 1][valid_pen]) \
                 if valid_pen.any() else pred.new_tensor(0.0)
    l_data = l_data_bio + l_data_pen

    # ── Physics residuals ─────────────────────────────────────────────────────
    if lambda_physics > 0.0:
        r_n    = model.r
        y_max  = model.y_max
        alpha  = model.alpha
        beta   = model.beta

        bio_seq = pred[:, :, 0]   # (B, T)
        pen_seq = pred[:, :, 1]   # (B, T)

        # Consecutive differences
        delta_bio_nn = bio_seq[:, 1:] - bio_seq[:, :-1]   # (B, T-1)
        delta_pen_nn = pen_seq[:, 1:] - pen_seq[:, :-1]   # (B, T-1)

        # ODE predictions
        delta_bio_ode = (r_n * bio_seq[:, :-1]
                         * (1.0 - bio_seq[:, :-1] / y_max)
                         * dt_n)
        delta_pen_ode = alpha * delta_bio_nn + beta * bio_seq[:, :-1] * dt_n

        l_phys_bio = F.mse_loss(delta_bio_nn, delta_bio_ode)
        l_phys_pen = F.mse_loss(delta_pen_nn, delta_pen_ode)
        l_physics  = l_phys_bio + l_phys_pen
    else:
        l_physics  = pred.new_tensor(0.0)
        l_phys_bio = pred.new_tensor(0.0)
        l_phys_pen = pred.new_tensor(0.0)

    # ── Soft penalty: growth rate must be non-negative ────────────────────────
    r_neg_penalty = F.relu(-model.r) * lambda_r_neg

    # ── Total ─────────────────────────────────────────────────────────────────
    loss = l_data + lambda_physics * l_physics + r_neg_penalty

    loss_dict = {
        "loss":       float(loss.item()),
        "loss_data":  float(l_data.item()),
        "loss_phys":  float((lambda_physics * l_physics).item()),
        "loss_bio":   float(l_phys_bio.item()),
        "loss_pen":   float(l_phys_pen.item()),
    }
    return loss, loss_dict
