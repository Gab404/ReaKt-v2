"""
src/data/cvae_encoder.py
========================
CVAERamanEncoderV2 -- Frozen CVAE Raman feature extractor.

This encoder wraps the CVAE trained with Savitzky-Golay first-derivative
preprocessing + StandardScaler normalisation.  It is the companion of the
CVAE trained in train_autoencoder.py and is compatible with the
PenicillinDataModule.raman_encoder interface.

Differences from CDAERamanEncoderV2
-------------------------------------
  CDAE : deterministic encoder  z = f(x)                    -> latent (N, 64)
  CVAE : variational encoder    (mu, log_var) = f(x)        -> latent (N, 64)
         At inference the MEAN mu is used as the deterministic
         representation (no sampling noise).

The preprocessing pipeline is IDENTICAL to CDAERamanEncoderV2:
  raw Raman (2001D) -> SG(d=1, w=15, p=2) -> StandardScaler -> CVAE encoder

Usage
-----
    enc = CVAERamanEncoderV2(
        ckpt_path   = "checkpoints/cvae_best.pt",
        scaler_path = "checkpoints/cvae_scaler.joblib",
        device      = torch.device("cuda"),
    )
    latents = enc.encode_dataframe(df_full)   # (N, 64)  -- posterior means
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch
from scipy.signal import savgol_filter

# ── Project path bootstrap ────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.autoencoder.model import CVAE_Raman
from src.data.dataset import RAMAN_START_COL_IDX, RAMAN_LATENT_DIM


class CVAERamanEncoderV2:
    """
    Frozen CVAE-based Raman feature extractor (V2 preprocessing pipeline).

    Pipeline for each row of the input DataFrame
    ---------------------------------------------
    raw spectra  (N, 2001)
        -> clamp negatives / NaN to 0
        -> Savitzky-Golay first-derivative  (window=15, poly=2, deriv=1)
        -> StandardScaler.transform()       (fitted on training set)
        -> CVAE_Raman.encode() -> mu        (frozen; posterior mean used)
    ->  latents (N, 64)

    Rows where all Raman channels are zero (instrument warm-up at batch start)
    receive a zero latent vector.  PenicillinDataModule.load() then
    forward-fills these rows using the next valid spectrum.

    Parameters
    ----------
    ckpt_path    : path to ``checkpoints/cvae_best.pt``
    scaler_path  : path to ``checkpoints/cvae_scaler.joblib``
    device       : torch device for CVAE inference
    n_raman_cols : wavenumber channels to use (default 2001)
    latent_dim   : CVAE latent dimensionality (default 64)
    encode_batch : mini-batch size for GPU encoding (default 1024)
    sg_window    : SG filter window length (default 15, must be odd)
    sg_poly      : SG polynomial order (default 2)
    sg_deriv     : SG derivative order (default 1 = first derivative)
    """

    def __init__(
        self,
        ckpt_path:    str,
        scaler_path:  str,
        device:       torch.device,
        n_raman_cols: int   = 2001,
        latent_dim:   int   = RAMAN_LATENT_DIM,
        encode_batch: int   = 1024,
        sg_window:    int   = 15,
        sg_poly:      int   = 2,
        sg_deriv:     int   = 1,
    ) -> None:
        self.device       = device
        self.n_raman_cols = n_raman_cols
        self.latent_dim   = latent_dim
        self.encode_batch = encode_batch
        self.sg_window    = sg_window
        self.sg_poly      = sg_poly
        self.sg_deriv     = sg_deriv

        # ── Load CVAE model ───────────────────────────────────────────────────
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Checkpoint produced by train_cvae() in train_autoencoder.py:
        # {"epoch": ..., "model_state_dict": ..., "input_length": ..., "latent_dim": ...}
        input_length      = int(state.get("input_length", n_raman_cols))
        loaded_latent_dim = int(state.get("latent_dim",   latent_dim))

        self._model = CVAE_Raman(
            input_length=input_length,
            latent_dim=loaded_latent_dim,
        ).to(device)
        self._model.load_state_dict(state["model_state_dict"])

        # Freeze all parameters permanently
        for p in self._model.parameters():
            p.requires_grad_(False)
        self._model.eval()

        # Update latent_dim to match the loaded model
        self.latent_dim = loaded_latent_dim

        # ── Load StandardScaler ───────────────────────────────────────────────
        self._scaler = joblib.load(scaler_path)

        print(
            f"  [CVAERamanEncoderV2] Loaded from {Path(ckpt_path).name} | "
            f"input={input_length}  latent={self.latent_dim}  "
            f"SG(w={sg_window},p={sg_poly},d={sg_deriv})"
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _apply_sg(self, X: np.ndarray) -> np.ndarray:
        """Apply Savitzky-Golay first-derivative filter row-wise."""
        return savgol_filter(
            X.astype(np.float64),
            window_length=self.sg_window,
            polyorder=self.sg_poly,
            deriv=self.sg_deriv,
            axis=1,
        ).astype(np.float32)

    # ── Core encoding API ─────────────────────────────────────────────────────

    @torch.no_grad()
    def encode(self, spectra: np.ndarray) -> np.ndarray:
        """
        Encode a batch of ALREADY preprocessed spectra (SG + scaled).

        Uses the posterior mean mu as the deterministic latent representation
        (no sampling noise at inference).

        Parameters
        ----------
        spectra : (N, n_raman_cols) float32  -- SG-filtered, StandardScaler-
                  transformed spectra ready to pass through the CVAE encoder

        Returns
        -------
        latents : (N, latent_dim) float32  -- posterior means
        """
        if len(spectra) == 0:
            return np.zeros((0, self.latent_dim), dtype=np.float32)

        out_parts = []
        for start in range(0, len(spectra), self.encode_batch):
            chunk = torch.tensor(
                spectra[start : start + self.encode_batch],
                dtype=torch.float32,
                device=self.device,
            )
            # CVAE_Raman.encode() returns (mu, log_var); we use mu
            mu, _ = self._model.encode(chunk.unsqueeze(1))   # (B, 1, L) -> (mu, lv)
            out_parts.append(mu.cpu().numpy())

        return np.concatenate(out_parts, axis=0)

    @torch.no_grad()
    def encode_dataframe(self, df) -> np.ndarray:
        """
        Full preprocessing + encoding pipeline on a raw CSV DataFrame.

        Called by PenicillinDataModule.load() once per CSV load.
        Zero-energy rows (instrument warm-up) receive a zero latent vector;
        the DataModule then forward-fills them.

        Parameters
        ----------
        df : pandas DataFrame containing all CSV columns
             (including the 2001 Raman wavenumber columns starting at
             column index RAMAN_START_COL_IDX)

        Returns
        -------
        latents : (N, latent_dim)  float32  (same row order as df)
                  -- posterior means mu (deterministic, no sampling)
        """
        N = len(df)
        latents = np.zeros((N, self.latent_dim), dtype=np.float32)

        # ── Extract Raman columns by absolute column position ─────────────────
        all_cols   = df.columns.tolist()
        avail      = len(all_cols) - RAMAN_START_COL_IDX
        n_use      = min(self.n_raman_cols, avail)
        raman_cols = all_cols[RAMAN_START_COL_IDX : RAMAN_START_COL_IDX + n_use]
        X_raw      = df[raman_cols].values.astype(np.float32)

        # ── Identify valid rows (non-zero energy = instrument running) ────────
        energy_mask = np.abs(X_raw).sum(axis=1) > 0
        n_valid     = int(energy_mask.sum())
        if n_valid == 0:
            return latents

        print(f"  Raman (CVAE V2): encoding {n_valid:,} / {N:,} rows ...")

        # ── Preprocess valid rows ─────────────────────────────────────────────
        X_valid = X_raw[energy_mask].copy()

        # Clamp any NaN / negative values before SG filter
        X_valid = np.nan_to_num(X_valid, nan=0.0, posinf=0.0, neginf=0.0)
        X_valid = np.clip(X_valid, a_min=0.0, a_max=None)

        # 1. Savitzky-Golay first-derivative (per-spectrum, axis=1)
        X_sg = self._apply_sg(X_valid)

        # 2. StandardScaler transform (fitted on training data in train_autoencoder.py)
        X_scaled = self._scaler.transform(X_sg).astype(np.float32)

        # 3. CVAE encoder (frozen, no grad) -- returns posterior mean mu
        latents[energy_mask] = self.encode(X_scaled)

        return latents

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def model(self) -> CVAE_Raman:
        """The underlying frozen CVAE model."""
        return self._model
