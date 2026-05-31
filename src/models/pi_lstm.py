"""
src/models/pi_lstm.py
=====================
BioreactorLogiPINN — pure nn.Module definition.

No data loading, training, or evaluation logic here.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class BioreactorLogiPINN(nn.Module):
    """
    Physics-Informed LSTM for dual-output prediction of biomass X and
    penicillin P in a fed-batch fermentation.

    Architecture
    ------------
    Input  : (B, seq_len, input_size)  — normalised process (+ Raman) features
    LSTM   : input_size → hidden_size
    Shared : LayerNorm → Linear → LeakyReLU
    Heads  : Linear → Sigmoid  (one per target, output ∈ (0,1))
    Output : (B, seq_len, 2)   — [bio_norm, pen_norm]

    Physics parameters (all positive via Softplus, learnable):
    ----------
    r_n     : normalised logistic growth rate
    y_max_n : normalised carrying capacity (biomass)
    alpha_n : Luedeking-Piret growth-associated coefficient
    beta_n  : Luedeking-Piret maintenance coefficient

    These are optimised with a separate (higher) learning rate by the trainer.
    """

    def __init__(
        self,
        input_size:      int   = 23,
        hidden_size:     int   = 64,
        num_lstm_layers: int   = 1,
        fc_hidden:       int   = 32,
        lstm_dropout:    float = 0.0,
        # Physics parameter initialisations (in raw/unconstrained space)
        r_n_init:     float = 45.0,
        y_max_n_init: float = 1.05,
        alpha_n_init: float = 1.0,
        beta_n_init:  float = 1.0,
    ):
        super().__init__()

        self.input_size      = input_size
        self.hidden_size     = hidden_size
        self.num_lstm_layers = num_lstm_layers

        # ── Backbone ──────────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if num_lstm_layers > 1 else 0.0,
        )

        # ── Shared FC ─────────────────────────────────────────────────────────
        self.shared = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, fc_hidden),
            nn.LeakyReLU(0.1),
        )

        # ── Output heads (one per target) ─────────────────────────────────────
        self.head_bio = nn.Sequential(nn.Linear(fc_hidden, 1), nn.Sigmoid())
        self.head_pen = nn.Sequential(nn.Linear(fc_hidden, 1), nn.Sigmoid())

        # ── Physics parameters (in normalised / unconstrained space) ──────────
        # Actual values = Softplus(raw) — always positive
        self._r_raw    = nn.Parameter(torch.tensor(_softplus_inv(r_n_init)))
        self._ymax_raw = nn.Parameter(torch.tensor(_softplus_inv(y_max_n_init)))
        self._alpha_raw = nn.Parameter(torch.tensor(_softplus_inv(alpha_n_init)))
        self._beta_raw  = nn.Parameter(torch.tensor(_softplus_inv(beta_n_init)))

        self._init_weights()

    # ── Constrained physics parameters ───────────────────────────────────────

    @property
    def r(self) -> torch.Tensor:
        return F.softplus(self._r_raw)

    @property
    def y_max(self) -> torch.Tensor:
        return F.softplus(self._ymax_raw)

    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self._alpha_raw)

    @property
    def beta(self) -> torch.Tensor:
        return F.softplus(self._beta_raw)

    def get_physics_params(self) -> Dict[str, float]:
        return {
            "r_n":     float(self.r.item()),
            "y_max_n": float(self.y_max.item()),
            "alpha_n": float(self.alpha.item()),
            "beta_n":  float(self.beta.item()),
        }

    # ── Network parameters (excludes physics) ────────────────────────────────

    def network_parameters(self):
        physics_ids = {id(p) for p in self.physics_parameters()}
        return [p for p in self.parameters() if id(p) not in physics_ids]

    def physics_parameters(self):
        return [self._r_raw, self._ymax_raw, self._alpha_raw, self._beta_raw]

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, input_size)

        Returns
        -------
        out : (B, T, 2)  — [bio_norm, pen_norm]
        """
        h, _ = self.lstm(x)                          # (B, T, H)
        z    = self.shared(h)                        # (B, T, fc_hidden)
        bio  = self.head_bio(z)                      # (B, T, 1)
        pen  = self.head_pen(z)                      # (B, T, 1)
        return torch.cat([bio, pen], dim=-1)         # (B, T, 2)

    # ── Weight initialisation ─────────────────────────────────────────────────

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Forget-gate bias = 1 (improves long-range memory)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

        for layer in list(self.shared) + list(self.head_bio) + list(self.head_pen):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _softplus_inv(y: float, beta: float = 1.0) -> float:
    """Inverse of Softplus: x such that softplus(x) = y."""
    return math.log(math.expm1(beta * y)) / beta
