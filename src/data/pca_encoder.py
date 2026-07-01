"""
src/data/pca_encoder.py
=======================
PCARamanEncoderV1 -- Frozen PCA Raman feature extractor.

This encoder wraps a scikit-learn PCA model + StandardScaler to reduce the
dimensionality of preprocessed Raman spectra to a fixed number of
principal components.

Pipeline (byte-for-byte identical to CDAERamanEncoderV2 except for the
final dim-reduction step which is PCA instead of CDAE):
    raw Raman (2001D) -> SG(d=1, w=15, p=2) -> StandardScaler
    -> PCA model (fitted on training data) -> PCA components (e.g., 4D)

Both the PCA and the StandardScaler are loaded from disk; they must have
been fitted previously by ``pca_baseline.py`` (which saves them as
``./checkpoints/pca_best.joblib`` and ``./checkpoints/pca_scaler.joblib``).

Usage
-----
    enc = PCARamanEncoderV1(
        pca_model_path = "checkpoints/pca_best.joblib",
        scaler_path    = "checkpoints/pca_scaler.joblib",
        n_components   = 4,
        n_raman_cols   = 2001,
    )
    pca_components = enc.encode_dataframe(df_full)   # (N, n_components)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA

# -- Project path bootstrap --------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.dataset import RAMAN_START_COL_IDX


class PCARamanEncoderV1:
    """
    Frozen PCA-based Raman feature extractor (V1 preprocessing pipeline).

    Pipeline for each row of the input DataFrame
    ---------------------------------------------
    raw spectra  (N, 2001)
        -> clamp negatives / NaN to 0
        -> Savitzky-Golay first-derivative  (window=15, poly=2, deriv=1)
        -> StandardScaler.transform()       (fitted on training set)
        -> PCA.transform()                  (fitted on training set)
    ->  pca_components (N, n_components)

    Rows where all Raman channels are zero (instrument warm-up at batch start)
    receive a zero feature vector.  PenicillinDataModule.load() then
    forward-fills these rows using the next valid spectrum.

    Parameters
    ----------
    pca_model_path : path to the saved PCA model file (joblib or pickle)
    scaler_path    : path to the saved StandardScaler file
    n_raman_cols   : number of wavenumber channels to use (default 2001)
    n_components   : number of PCA components to extract (default 4)
    sg_window      : SG filter window length (default 15, must be odd)
    sg_poly        : SG polynomial order (default 2)
    sg_deriv       : SG derivative order (default 1 = first derivative)
    """

    def __init__(
        self,
        pca_model_path:   str,
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

        # -- Load PCA model ------------------------------------------------
        try:
            self._pca_model = joblib.load(pca_model_path)
            print(f"  [PCARamanEncoderV1] Loaded PCA model from "
                  f"{Path(pca_model_path).name}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"PCA model not found at {pca_model_path}. "
                "Please run pca_baseline.py first to fit and save the PCA encoder."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load PCA model: {e}")

        # Check that the loaded object is a PCA model
        if not isinstance(self._pca_model, PCA):
            print(
                "Warning: Loaded object is not a scikit-learn PCA model. "
                f"Type found: {type(self._pca_model)}. "
                "Ensure it has a `transform` method."
            )

        # Check that the loaded PCA has the expected number of components
        loaded_n_components = int(getattr(self._pca_model, "n_components_",
                                          self.n_components))
        if loaded_n_components != self.n_components:
            print(
                f"  [PCARamanEncoderV1] Warning: PCA model has "
                f"{loaded_n_components} components but n_components={self.n_components} "
                f"was requested.  Using the loaded value ({loaded_n_components})."
            )
            self.n_components = loaded_n_components

        # -- Load StandardScaler -------------------------------------------
        try:
            self._scaler = joblib.load(scaler_path)
            print(f"  [PCARamanEncoderV1] Loaded scaler from "
                  f"{Path(scaler_path).name}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Scaler not found at {scaler_path}. "
                "Please run pca_baseline.py first to fit and save the scaler."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load scaler: {e}")

        print(
            f"  [PCARamanEncoderV1] Initialised with {self.n_components} components "
            f"via SG(w={sg_window},p={sg_poly},d={sg_deriv})"
        )

    # -- Internal helpers ------------------------------------------------------

    def _apply_sg(self, X: np.ndarray) -> np.ndarray:
        """
        Apply Savitzky-Golay first-derivative filter row-wise (along axis=1,
        the spectral / wavenumber axis).  Same parameters as PLS, SVR and
        CDAE pipelines for strict comparability.
        """
        return savgol_filter(
            X.astype(np.float64),
            window_length=self.sg_window,
            polyorder=self.sg_poly,
            deriv=self.sg_deriv,
            axis=1,
        ).astype(np.float32)

    # -- Core encoding API -----------------------------------------------------

    def encode(self, spectra: np.ndarray) -> np.ndarray:
        """
        Encode a batch of ALREADY preprocessed spectra (SG + scaled).

        Parameters
        ----------
        spectra : (N, n_raman_cols) float32  -- SG-filtered, StandardScaler-
                  transformed spectra ready to pass through the PCA model

        Returns
        -------
        pca_components : (N, n_components) float32
        """
        if len(spectra) == 0:
            return np.zeros((0, self.n_components), dtype=np.float32)

        # PCA model expects samples in rows, features in columns
        pca_components = self._pca_model.transform(spectra)

        # Ensure the output has the correct number of components and dtype
        if pca_components.shape[1] != self.n_components:
            raise ValueError(
                f"PCA model output {pca_components.shape[1]} components, "
                f"but expected {self.n_components}."
            )
        return pca_components.astype(np.float32)

    def encode_dataframe(self, df) -> np.ndarray:
        """
        Full preprocessing + encoding pipeline on a raw CSV DataFrame.

        Called by PenicillinDataModule.load() once per CSV load.  Zero-energy
        rows (instrument warm-up) receive a zero feature vector; the
        DataModule then forward-fills them.

        Parameters
        ----------
        df : pandas DataFrame containing all CSV columns
             (including the 2001 Raman wavenumber columns starting at
             column index RAMAN_START_COL_IDX)

        Returns
        -------
        pca_components : (N, n_components)  float32  (same row order as df)
        """
        N = len(df)
        pca_components = np.zeros((N, self.n_components), dtype=np.float32)

        # -- Extract Raman columns by absolute column position -------------
        all_cols   = df.columns.tolist()
        avail      = len(all_cols) - RAMAN_START_COL_IDX
        n_use      = min(self.n_raman_cols, avail)
        raman_cols = all_cols[RAMAN_START_COL_IDX : RAMAN_START_COL_IDX + n_use]
        X_raw      = df[raman_cols].values.astype(np.float32)

        # -- Identify valid rows (non-zero energy = instrument running) ----
        energy_mask = np.abs(X_raw).sum(axis=1) > 0
        n_valid     = int(energy_mask.sum())
        if n_valid == 0:
            return pca_components

        print(f"  PCA Raman V1: encoding {n_valid:,} / {N:,} rows ...")

        # -- Preprocess valid rows -----------------------------------------
        X_valid = X_raw[energy_mask].copy()

        # Clamp any NaN / negative values before SG filter
        X_valid = np.nan_to_num(X_valid, nan=0.0, posinf=0.0, neginf=0.0)
        X_valid = np.clip(X_valid, a_min=0.0, a_max=None)

        # 1. Savitzky-Golay first-derivative
        X_sg = self._apply_sg(X_valid)

        # 2. StandardScaler transform (fitted on training data in pca_baseline.py)
        X_scaled = self._scaler.transform(X_sg).astype(np.float32)

        # 3. PCA transform
        pca_components[energy_mask] = self.encode(X_scaled)

        return pca_components
