"""
src/data/raman_encoder.py
=========================
Wrapper around the trained CDAE_Raman autoencoder.

The encoder weights are frozen permanently — this class is never used
for training the CDAE, only for inference (feature extraction).

Usage
-----
    enc = RamanEncoder("checkpoints/cdae_best_model.pth", device)
    latents = enc.encode_dataframe(df_full)   # np.ndarray (N, 64)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn

# Add autoencoder source to path so CDAE_Raman can be imported
_AE_SRC = Path(__file__).resolve().parents[2] / "src" / "autoencoder"
if str(_AE_SRC) not in sys.path:
    sys.path.insert(0, str(_AE_SRC))

from model import CDAE_Raman  # noqa: E402  (local import after path manipulation)

from src.data.dataset import (
    RAMAN_FLAG_COL,
    RAMAN_START_COL_IDX,
    RAMAN_N_COLS,
    RAMAN_LATENT_DIM,
)


class RamanEncoder:
    """
    Frozen CDAE encoder used to compress 2200-wavenumber Raman spectra
    into 64-dimensional latent vectors.

    Parameters
    ----------
    ckpt_path      : path to the trained CDAE checkpoint (.pth)
    device         : torch device for inference
    input_length   : number of Raman wavenumber columns (default 2200)
    latent_dim     : latent space dimensionality (default 64)
    encode_batch   : mini-batch size for GPU encoding (default 1024)
    flag_col       : name of the column indicating Raman availability
    start_col_idx  : 0-based column index where Raman spectra start in the CSV
    """

    def __init__(
        self,
        ckpt_path:    Union[str, Path],
        device:       torch.device,
        input_length: int = RAMAN_N_COLS,
        latent_dim:   int = RAMAN_LATENT_DIM,
        encode_batch: int = 1024,
        flag_col:     str = RAMAN_FLAG_COL,
        start_col_idx: int = RAMAN_START_COL_IDX,
    ):
        self.device       = device
        self.input_length = input_length
        self.latent_dim   = latent_dim
        self.encode_batch = encode_batch
        self.flag_col     = flag_col
        self.start_col_idx = start_col_idx

        # Build model and load weights
        self._model = CDAE_Raman(
            input_length=input_length,
            latent_dim=latent_dim,
        ).to(device)

        state = torch.load(ckpt_path, map_location="cpu")
        # Handle checkpoints that wrap state_dict under a key
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        self._model.load_state_dict(state)

        # Freeze all parameters — this encoder is never trained further
        for p in self._model.parameters():
            p.requires_grad_(False)
        self._model.eval()

    # ── Core encoding methods ─────────────────────────────────────────────────

    @torch.no_grad()
    def encode(self, spectra: np.ndarray) -> np.ndarray:
        """
        Encode a batch of spectra.

        Parameters
        ----------
        spectra : (N, input_length) float32 array, pre-normalised to [0, 1]

        Returns
        -------
        latents : (N, latent_dim) float32 array
        """
        out = []
        for start in range(0, len(spectra), self.encode_batch):
            chunk = torch.tensor(
                spectra[start : start + self.encode_batch],
                dtype=torch.float32,
                device=self.device,
            )
            out.append(self._model.encode(chunk).cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.latent_dim))

    @torch.no_grad()
    def encode_dataframe(self, df) -> np.ndarray:
        """
        Encode all Raman spectra in a raw CSV DataFrame.

        Rows where the Raman flag indicates no spectrum (flag value == 2)
        receive a zero latent vector.

        Parameters
        ----------
        df : full pandas DataFrame (all CSV columns present)

        Returns
        -------
        latents : (N, latent_dim) float32 array  (same row order as df)
        """
        N = len(df)
        latents = np.zeros((N, self.latent_dim), dtype=np.float32)

        # Identify rows that have a valid (non-zero) Raman spectrum.
        # The flag column is unreliable (always == 1 in the dataset), so we
        # use a signal-energy check matching the original legacy code.
        raman_cols_all = df.columns[self.start_col_idx :
                                     self.start_col_idx + self.input_length]
        raw_all = df[raman_cols_all].values.astype(np.float32)
        valid_mask = np.nansum(np.abs(raw_all), axis=1) > 0

        n_valid = valid_mask.sum()
        if n_valid == 0:
            return latents

        print(f"  Raman: encoding {n_valid:,} / {N:,} rows ...")

        # Extract spectra for valid rows only
        raw = df.loc[valid_mask, raman_cols_all].values.astype(np.float32)

        # NaN → 0 BEFORE normalisation (matches CDAE training pipeline).
        # Every spectrum contains a small number of NaN wavenumber bins;
        # zeroing them first ensures min=0 and the correct denominator.
        np.nan_to_num(raw, copy=False, nan=0.0)

        # Per-spectrum min-max normalisation to [0, 1]  (same as CDAE training)
        row_min = raw.min(axis=1, keepdims=True)
        row_max = raw.max(axis=1, keepdims=True)
        denom   = np.clip(row_max - row_min, a_min=1e-8, a_max=None)
        raw_norm = (raw - row_min) / denom

        latents[valid_mask] = self.encode(raw_norm)
        return latents

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def model(self) -> nn.Module:
        return self._model
