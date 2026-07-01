"""
src/data/pls_encoder.py
========================
PLSRamanEncoderV1 -- Frozen PLS Raman feature extractor.

This encoder wraps a scikit-learn PLS model to reduce the dimensionality
of preprocessed Raman spectra to a fixed number of components.

Pipeline:
  raw Raman (2001D) -> SG(d=1, w=15, p=2) -> StandardScaler
  -> PLS model (fitted on training data) -> PLS components (e.g., 4D)

Usage
-----
    enc = PLSRamanEncoderV1(
        pls_model_path   = "checkpoints/pls_best.pkl",
        scaler_path      = "checkpoints/pls_scaler.joblib",
        n_components:    4,
        n_raman_cols:    2001,
    )
    pls_components = enc.encode_dataframe(df_full)   # (N, n_components)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression

# ── Project path bootstrap ────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.dataset import RAMAN_START_COL_IDX


class PLSRamanEncoderV1:
    """
    Frozen PLS-based Raman feature extractor (V1 preprocessing pipeline).

    Parameters
    ----------
    pls_model_path   : path to the saved PLS model file (e.g., .pkl or .joblib)
    scaler_path      : path to the saved StandardScaler file
    n_raman_cols     : number of wavenumber channels to use (default 2001)
    n_components     : number of PLS components to extract (default 4)
    sg_window        : SG filter window length (default 15, must be odd)
    sg_poly          : SG polynomial order (default 2)
    sg_deriv         : SG derivative order (default 1 = first derivative)
    """

    def __init__(
        self,
        pls_model_path:   str,
        scaler_path:      str,
        n_raman_cols:     int = 2001,
        n_components:     int = 4,
        sg_window:        int = 15,
        sg_poly:          int = 2,
        sg_deriv:         int = 1,
    ) -> None:
        self.n_raman_cols = n_raman_cols
        self.n_components = n_components
        self.sg_window    = sg_window
        self.sg_poly      = sg_poly
        self.sg_deriv     = sg_deriv

        # ── Load PLS model ────────────────────────────────────────────────────
        try:
            self._pls_model = joblib.load(pls_model_path)
            print(f"  [PLSRamanEncoderV1] Loaded PLS model from {Path(pls_model_path).name}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"PLS model not found at {pls_model_path}. "
                "Please ensure the PLS model is trained and saved."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load PLS model: {e}")

        # Check if the loaded object is a PLSRegression model
        if not isinstance(self._pls_model, PLSRegression):
            print(
                "Warning: Loaded object is not a scikit-learn PLSRegression model. "
                f"Type found: {type(self._pls_model)}. "
                "Ensure it has `transform` method."
            )

        # ── Load StandardScaler ───────────────────────────────────────────────
        try:
            self._scaler = joblib.load(scaler_path)
            print(f"  [PLSRamanEncoderV1] Loaded scaler from {Path(scaler_path).name}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Scaler not found at {scaler_path}. "
                "Please ensure the scaler is trained and saved."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load scaler: {e}")

        print(
            f"  [PLSRamanEncoderV1] Initialized with {self.n_components} components "
            f"via SG(w={sg_window},p={sg_poly},d={sg_deriv})"
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

    def encode(self, spectra: np.ndarray) -> np.ndarray:
        """
        Encode a batch of ALREADY preprocessed spectra (SG + scaled).

        Parameters
        ----------
        spectra : (N, n_raman_cols) float32  -- SG-filtered, StandardScaler-
                  transformed spectra ready to pass through the PLS model

        Returns
        -------
        pls_components : (N, n_components) float32
        """
        if len(spectra) == 0:
            return np.zeros((0, self.n_components), dtype=np.float32)

        # PLS model expects samples in rows, features in columns
        pls_components = self._pls_model.transform(spectra)

        # Ensure the output has the correct number of components and dtype
        if pls_components.shape[1] != self.n_components:
            raise ValueError(
                f"PLS model output {pls_components.shape[1]} components, "
                f"but expected {self.n_components}."
            )
        return pls_components.astype(np.float32)

    def encode_dataframe(self, df) -> np.ndarray:
        """
        Full preprocessing + encoding pipeline on a raw CSV DataFrame.

        Parameters
        ----------
        df : pandas DataFrame containing all CSV columns

        Returns
        -------
        pls_components : (N, n_components)  float32  (same row order as df)
        """
        N = len(df)
        pls_components = np.zeros((N, self.n_components), dtype=np.float32)

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
            return pls_components

        print(f"  PLS Raman V1: encoding {n_valid:,} / {N:,} rows ...")

        # ── Preprocess valid rows ─────────────────────────────────────────────
        X_valid = X_raw[energy_mask].copy()

        X_valid = np.nan_to_num(X_valid, nan=0.0, posinf=0.0, neginf=0.0)
        X_valid = np.clip(X_valid, a_min=0.0, a_max=None)

        # 1. Savitzky-Golay first-derivative
        X_sg = self._apply_sg(X_valid)

        # 2. StandardScaler transform
        X_scaled = self._scaler.transform(X_sg).astype(np.float32)

        # 3. PLS transform
        pls_components[energy_mask] = self.encode(X_scaled)

        return pls_components
