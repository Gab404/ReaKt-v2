"""
src/models/delta_cdae_pi_lstm.py
================================
DeltaCDAEPILSTMModel — Physics-Informed LSTM that predicts the **increment**
of Penicillin concentration (ΔP = P[t] − P[t−1]) rather than the absolute
concentration P[t].

Architecture
------------
  Input    : (B, T, 64)  — CDAE latent vectors (pre-extracted from Raman)
  LSTM     : input_size=64 → hidden_size  (num_layers, batch_first)
  Shared   : LayerNorm(hidden) → Linear(hidden, fc_hidden) → LeakyReLU(0.1)
  head_pen : Linear(fc_hidden, 1)  → delta_pen_norm  (unbounded; ΔP can be
             negative when dilution / consumption dominates)
  head_rate: Linear(fc_hidden, 1)  → r_net   (unbounded)
  Output   : (B, T, 2)  —  [:, :, 0] = delta_pen_norm  (predicted ΔP)
                            [:, :, 1] = r_net  (net production rate)

Differences from CDAEPILSTMModel
---------------------------------
  • head_pen has **no Sigmoid** — since ΔP ∈ (−∞, +∞).
  • The physics loss (compute_delta_cdae_pinn_loss) directly enforces
      delta_pen_norm ≈ k_prod · r_net · dt_n
    without requiring finite-differencing of the concentration predictions.
    This is a tighter constraint because the model output IS the increment.

Physics parameter
-----------------
  k_prod  : positive scaling factor (Softplus of raw learnable parameter)
             used in the simplified mass-balance physics loss:
               ΔP_norm ≈ k_prod · r_net · dt_n
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeltaCDAEPILSTMModel(nn.Module):
    """
    CDAE-Physics-Informed LSTM predicting Penicillin **increments** (ΔP).

    Inputs  : (B, T, input_size=64)  — CDAE latent vectors
    Outputs : (B, T, 2)
                delta_pen_norm  ∈ (−∞, +∞)  (normalised ΔP, linear output)
                r_net           ∈ (−∞, +∞)  (production rate, raw)

    Parameters
    ----------
    input_size      : CDAE latent dimension (default 64)
    hidden_size     : LSTM hidden state size (default 64)
    num_lstm_layers : stacked LSTM layers (default 2)
    fc_hidden       : shared FC intermediate size (default 32)
    lstm_dropout    : dropout between LSTM layers (> 0 only if layers > 1)
    k_prod_init     : initial value of the physics scale parameter k_prod
    """

    def __init__(
        self,
        input_size:      int   = 64,
        hidden_size:     int   = 64,
        num_lstm_layers: int   = 2,
        fc_hidden:       int   = 32,
        lstm_dropout:    float = 0.0,
        k_prod_init:     float = 1.0,
    ) -> None:
        super().__init__()

        self.input_size      = input_size
        self.hidden_size     = hidden_size
        self.num_lstm_layers = num_lstm_layers

        # ── Temporal backbone ─────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if num_lstm_layers > 1 else 0.0,
        )

        # ── Shared projection ─────────────────────────────────────────────────
        self.shared = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, fc_hidden),
            nn.LeakyReLU(0.1),
        )

        # ── Output heads ──────────────────────────────────────────────────────
        # delta_pen head: NO activation — ΔP can be negative (dilution/degradation)
        self.head_pen  = nn.Linear(fc_hidden, 1)

        # rate head: no activation — r_net mirrors sign of ΔP
        self.head_rate = nn.Linear(fc_hidden, 1)

        # ── Physics parameter: k_prod (positive via Softplus) ─────────────────
        self._k_prod_raw = nn.Parameter(
            torch.tensor(_softplus_inv(k_prod_init), dtype=torch.float32)
        )

        self._init_weights()

    # ── Constrained physics parameter ─────────────────────────────────────────

    @property
    def k_prod(self) -> torch.Tensor:
        return F.softplus(self._k_prod_raw)

    def get_physics_params(self) -> Dict[str, float]:
        return {"k_prod": float(self.k_prod.item())}

    # ── Parameter groups ──────────────────────────────────────────────────────

    def physics_parameters(self) -> List[nn.Parameter]:
        return [self._k_prod_raw]

    def network_parameters(self) -> List[nn.Parameter]:
        physics_ids = {id(p) for p in self.physics_parameters()}
        return [p for p in self.parameters() if id(p) not in physics_ids]

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, 64)  — MinMaxScaled CDAE latent vectors

        Returns
        -------
        out : (B, T, 2)
              out[:, :, 0] = delta_pen_norm  (normalised ΔP, linear)
              out[:, :, 1] = r_net           (production rate, linear)
        """
        h, _  = self.lstm(x)            # (B, T, hidden_size)
        z     = self.shared(h)          # (B, T, fc_hidden)
        delta = self.head_pen(z)        # (B, T, 1)
        rate  = self.head_rate(z)       # (B, T, 1)
        return torch.cat([delta, rate], dim=-1)   # (B, T, 2)

    # ── Weight initialisation ─────────────────────────────────────────────────

    def _init_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

        for module in [*list(self.shared), self.head_pen, self.head_rate]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)


# ── Helper ────────────────────────────────────────────────────────────────────

def _softplus_inv(y: float, beta: float = 1.0) -> float:
    """Inverse of Softplus: compute x such that softplus(x) = y."""
    return math.log(math.expm1(beta * y)) / beta
