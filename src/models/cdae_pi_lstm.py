"""
src/models/cdae_pi_lstm.py
==========================
CDAEPILSTMModel — Physics-Informed LSTM for Penicillin prediction
using CDAE latent vectors of Raman spectra as the sole input feature.

Architecture
------------
  Input   : (B, T, 64)  — CDAE latent vectors (pre-extracted from Raman)
  LSTM    : input_size=64 → hidden_size  (num_layers, batch_first)
  Shared  : LayerNorm(hidden) → Linear(hidden, fc_hidden) → LeakyReLU(0.1)
  head_pen: Linear(fc_hidden, 1) → Sigmoid   → pen_norm ∈ (0, 1)
  head_rate: Linear(fc_hidden, 1)             → r_net   (unbounded)
  Output  : (B, T, 2)  —  [:, :, 0] = pen_norm
                           [:, :, 1] = r_net  (net production rate)

Physics parameter
-----------------
  k_prod  : positive scaling factor (Softplus of raw learnable parameter)
             used in the mass-balance physics loss:
               ΔP ≈ k_prod · r_net · Δt
             Optimised at a higher learning rate (1e-2 by default) by the
             CDAEPILSTMAlgorithm trainer.

Design rationale
----------------
  The two-head output structure is motivated by the simplified penicillin
  mass balance:  dP/dt ≈ r_net(t)
  The network predicts BOTH the concentration P and its rate of change r_net.
  The physics loss penalises inconsistency between these two predictions,
  acting as a self-supervised regulariser that encodes domain knowledge
  without requiring extra labelled data.

  Biomass is NOT predicted here because the CDAE is trained purely on Raman
  spectra, which encode biochemical composition rather than process dynamics.
  For benchmarks requiring biomass RMSE the standard PI-LSTM (with process
  variables) should be used alongside.
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class CDAEPILSTMModel(nn.Module):
    """
    CDAE-Physics-Informed LSTM for Penicillin soft-sensing from Raman spectra.

    Inputs  : (B, T, input_size=64)  — CDAE latent vectors
    Outputs : (B, T, 2)              — [pen_norm, r_net]
                pen_norm  ∈ (0, 1) via Sigmoid  (MinMaxScaler range)
                r_net     ∈ (-∞, +∞)            (production rate, raw)

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
        # LayerNorm stabilises the wide dynamic range of CDAE latents;
        # LeakyReLU avoids dead neurons while allowing negative activations.
        self.shared = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, fc_hidden),
            nn.LeakyReLU(0.1),
        )

        # ── Output heads ──────────────────────────────────────────────────────
        # pen head: Sigmoid so output is strictly in (0, 1)
        # (consistent with MinMaxScaler range used during training)
        self.head_pen  = nn.Sequential(nn.Linear(fc_hidden, 1), nn.Sigmoid())

        # rate head: no activation — r_net can be positive (production) or
        # negative (degradation/dilution dominated), mimicking dP/dt
        self.head_rate = nn.Linear(fc_hidden, 1)

        # ── Physics parameter: k_prod (positive via Softplus) ─────────────────
        # Physical meaning: overall mass-balance scale factor linking the
        # dimensionless rate output r_net to the normalised penicillin increment:
        #   ΔP_norm ≈ k_prod · r_net · dt_n
        # Stored in unconstrained (raw) space; k_prod = Softplus(_k_prod_raw).
        self._k_prod_raw = nn.Parameter(
            torch.tensor(_softplus_inv(k_prod_init), dtype=torch.float32)
        )

        self._init_weights()

    # ── Constrained physics parameter ─────────────────────────────────────────

    @property
    def k_prod(self) -> torch.Tensor:
        """
        Positive production scale factor k_prod = Softplus(_k_prod_raw).

        Constrained to be strictly positive so the mass-balance residual
        has a consistent sign interpretation.
        """
        return F.softplus(self._k_prod_raw)

    def get_physics_params(self) -> Dict[str, float]:
        return {"k_prod": float(self.k_prod.item())}

    # ── Parameter groups (used by CDAEPILSTMAlgorithm._make_optimizer) ────────

    def physics_parameters(self) -> List[nn.Parameter]:
        """Physics parameters only — optimised at a higher learning rate."""
        return [self._k_prod_raw]

    def network_parameters(self) -> List[nn.Parameter]:
        """All parameters except physics parameters."""
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
              out[:, :, 0] = pen_norm  (Penicillin, normalised, Sigmoid)
              out[:, :, 1] = r_net     (production rate, raw linear)
        """
        h, _  = self.lstm(x)            # (B, T, hidden_size)
        z     = self.shared(h)          # (B, T, fc_hidden)
        pen   = self.head_pen(z)        # (B, T, 1)
        rate  = self.head_rate(z)       # (B, T, 1)
        return torch.cat([pen, rate], dim=-1)   # (B, T, 2)

    # ── Weight initialisation ─────────────────────────────────────────────────

    def _init_weights(self) -> None:
        """
        Standard LSTM best-practices initialisation:
          - Xavier uniform for input–hidden weights
          - Orthogonal for hidden–hidden weights
          - Zero bias, except forget-gate bias = 1 (improves long-range memory)
        FC layers: Xavier uniform weights, zero bias.
        """
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Set forget-gate bias to 1 to help retain long-range context
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

        for module in [*list(self.shared), *list(self.head_pen), self.head_rate]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)


# ── Helper ────────────────────────────────────────────────────────────────────

def _softplus_inv(y: float, beta: float = 1.0) -> float:
    """Inverse of Softplus: compute x such that softplus(x) = y."""
    return math.log(math.expm1(beta * y)) / beta
