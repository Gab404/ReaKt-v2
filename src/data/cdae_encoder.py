"""
src/data/cdae_encoder.py
========================
CDAERamanEncoderV2 — Frozen CDAE Raman feature extractor (V2).

This encoder wraps the NEW CDAE trained with Savitzky-Golay first-derivative
preprocessing + StandardScaler normalisation.  It is the companion of the
CDAE trained in train_autoencoder.py and is compatible with the
PenicillinDataModule.raman_encoder interface.

Key differences from V1 (src/data/raman_encoder.py)
----------------------------------------------------
  V1 : per-spectrum MinMax normalisation → [0, 1]  before the OLD CDAE
  V2 : SG(d=1) + StandardScaler(mean/std of TRAINING set) → NEW CDAE

Usage
-----
    enc = CDAERamanEncoderV2(
        ckpt_path   = "checkpoints/cdae_best.pt",
        scaler_path = "checkpoints/cdae_scaler.joblib",
        device      = torch.device("cuda"),
    )
    latents = enc.encode_dataframe(df_full)   # (N, 64)
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

from src.autoencoder.model import CDAE_Raman
from src.data.dataset import RAMAN_START_COL_IDX, RAMAN_LATENT_DIM


class CDAERamanEncoderV2:
    """
    Frozen CDAE-based Raman feature extractor (V2 preprocessing pipeline).

    Pipeline for each row of the input DataFrame
    ---------------------------------------------
    raw spectra  (N, 2001)
        → clamp negatives / NaN to 0
        → Savitzky-Golay first-derivative  (window=15, poly=2, deriv=1)
        → StandardScaler.transform()       (fitted on training set)
        → CDAE_Raman.encode()              (frozen)
    →  latents (N, 64)

    Rows where all Raman channels are zero (instrument warm-up at batch start)
    receive a zero latent vector.  PenicillinDataModule.load() then
    forward-fills these rows using the next valid spectrum.

    Parameters
    ----------
    ckpt_path    : path to ``checkpoints/cdae_best.pt``
    scaler_path  : path to ``checkpoints/cdae_scaler.joblib``
    device       : torch device for CDAE inference
    n_raman_cols : wavenumber channels to use (default 2001)
    latent_dim   : CDAE latent dimensionality (default 64)
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

        # ── Load CDAE model ───────────────────────────────────────────────────
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Checkpoint produced by train_cdae() in train_autoencoder.py:
        # {"epoch": ..., "model_state_dict": ..., "input_length": ..., "latent_dim": ...}
        input_length = int(state.get("input_length", n_raman_cols))
        loaded_latent_dim = int(state.get("latent_dim", latent_dim))

        self._model = CDAE_Raman(
            input_length=input_length,
            latent_dim=loaded_latent_dim,
            noise_std=0.0,          # no noise during inference
        ).to(device)
        self._model.load_state_dict(state["model_state_dict"])

        # Freeze all parameters permanently
        for p in self._model.parameters():
            p.requires_grad_(False)
        self._model.eval()

        # Update latent_dim to match the loaded model (in case of override)
        self.latent_dim = loaded_latent_dim

        # ── Load StandardScaler ───────────────────────────────────────────────
        self._scaler = joblib.load(scaler_path)

        print(
            f"  [CDAERamanEncoderV2] Loaded from {Path(ckpt_path).name} | "
            f"input={input_length}  latent={self.latent_dim}  "
            f"SG(w={sg_window},p={sg_poly},d={sg_deriv})"
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _apply_sg(self, X: np.ndarray) -> np.ndarray:
        """
        Apply Savitzky-Golay first-derivative filter row-wise.

        Parameters
        ----------
        X : (N, n_raman_cols)  float32 raw intensities

        Returns
        -------
        X_sg : (N, n_raman_cols)  float32 first-derivative spectra
        """
        return savgol_filter(
            X.astype(np.float64),      # savgol prefers float64
            window_length=self.sg_window,
            polyorder=self.sg_poly,
            deriv=self.sg_deriv,
            axis=1,                    # filter along wavenumber axis
        ).astype(np.float32)

    # ── Core encoding API ─────────────────────────────────────────────────────

    @torch.no_grad()
    def encode(self, spectra: np.ndarray) -> np.ndarray:
        """
        Encode a batch of ALREADY preprocessed spectra (SG + scaled).

        Parameters
        ----------
        spectra : (N, n_raman_cols) float32  — SG-filtered, StandardScaler-
                  transformed spectra ready to pass through the CDAE encoder

        Returns
        -------
        latents : (N, latent_dim) float32
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
            # CDAE_Raman.encode() accepts (B, L) → returns (B, latent_dim)
            latent_chunk = self._model.encode(chunk).cpu().numpy()
            out_parts.append(latent_chunk)

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
        """
        N = len(df)
        latents = np.zeros((N, self.latent_dim), dtype=np.float32)

        # ── Extract Raman columns by absolute column position ─────────────────
        all_cols    = df.columns.tolist()
        avail       = len(all_cols) - RAMAN_START_COL_IDX
        n_use       = min(self.n_raman_cols, avail)
        raman_cols  = all_cols[RAMAN_START_COL_IDX : RAMAN_START_COL_IDX + n_use]
        X_raw       = df[raman_cols].values.astype(np.float32)

        # ── Identify valid rows (non-zero energy = instrument running) ────────
        energy_mask = np.abs(X_raw).sum(axis=1) > 0
        n_valid     = int(energy_mask.sum())
        if n_valid == 0:
            return latents

        print(f"  Raman (V2): encoding {n_valid:,} / {N:,} rows ...")

        # ── Preprocess valid rows ─────────────────────────────────────────────
        X_valid = X_raw[energy_mask].copy()

        # Clamp any NaN / negative values before SG filter
        X_valid = np.nan_to_num(X_valid, nan=0.0, posinf=0.0, neginf=0.0)
        X_valid = np.clip(X_valid, a_min=0.0, a_max=None)

        # 1. Savitzky-Golay first-derivative (per-spectrum, axis=1)
        X_sg = self._apply_sg(X_valid)

        # 2. StandardScaler transform (fitted on training data in train_autoencoder.py)
        X_scaled = self._scaler.transform(X_sg).astype(np.float32)

        # 3. CDAE encoder (frozen, no grad)
        latents[energy_mask] = self.encode(X_scaled)

        return latents

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def model(self) -> CDAE_Raman:
        """The underlying frozen CDAE model."""
        return self._model
