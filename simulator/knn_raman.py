"""
simulator/knn_raman.py
======================
KNNRamanSelector — retrieves a Raman latent vector for any process state by
finding the k-nearest neighbours in a pre-built cache of (process_features,
Raman_latent) pairs from the training dataset.

The cache is built once (first call, ~5 min) and persisted to a .npz file.
Subsequent calls load the cache instantly.

Usage
-----
    selector = KNNRamanSelector(
        csv_path="100_Batches_IndPenSim_V3.csv",
        raman_encoder=encoder,
        device=device,
    )
    latent = selector.query(process_state_dict)  # (64,) float32
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.data.dataset import PROCESS_FEATURE_COLS, RAMAN_LATENT_DIM


class KNNRamanSelector:
    """
    KNN-based Raman latent retrieval from a pre-encoded CSV cache.

    Parameters
    ----------
    csv_path      : path to the full IndPenSim CSV (with Raman columns)
    raman_encoder : RamanEncoder instance for encoding spectra
    device        : torch device for encoder inference
    cache_path    : .npz file to store / load the pre-computed cache
    n_neighbors   : number of nearest neighbours for retrieval (default 5)
    """

    # All 23 PROCESS_FEATURE_COLS are used as query features
    QUERY_FEATURES = PROCESS_FEATURE_COLS

    def __init__(
        self,
        csv_path: Union[str, Path],
        raman_encoder,
        device,
        cache_path: Union[str, Path] = "./outputs/raman_knn_cache.npz",
        n_neighbors: int = 5,
    ):
        self.n_neighbors = n_neighbors
        self._cache_path = Path(cache_path)

        # ── Load or build cache ───────────────────────────────────────────────
        if self._cache_path.exists():
            print(f"[KNNRamanSelector] Loading cache from {self._cache_path} ...")
            cache = np.load(str(self._cache_path))
            query_arr   = cache["query_features"]   # (N_valid, 23)
            latents_arr = cache["latents"]           # (N_valid, 64)
            print(f"[KNNRamanSelector] Cache loaded: {len(query_arr):,} rows.")
        else:
            print(f"[KNNRamanSelector] Cache not found. Building from {csv_path} ...")
            query_arr, latents_arr = self._build_cache(
                csv_path, raman_encoder, device, self._cache_path
            )

        # ── Fit StandardScaler on query features ─────────────────────────────
        self._scaler = StandardScaler()
        query_scaled = self._scaler.fit_transform(query_arr.astype(np.float64))

        # ── Fit NearestNeighbors ──────────────────────────────────────────────
        self._nn = NearestNeighbors(
            n_neighbors=n_neighbors,
            algorithm="ball_tree",
            metric="euclidean",
            n_jobs=-1,
        )
        self._nn.fit(query_scaled)
        self._latents = latents_arr   # (N_valid, 64) float32

    # ── Public API ────────────────────────────────────────────────────────────

    def query(self, process_state_dict: Dict[str, float]) -> np.ndarray:
        """
        Retrieve a weighted-average Raman latent for a given process state.

        Parameters
        ----------
        process_state_dict : dict {col_name: float} for all 23 PROCESS_FEATURE_COLS

        Returns
        -------
        latent : (64,) float32 — distance-weighted average of k neighbours
        """
        # Build query row in the order of QUERY_FEATURES
        row = np.array(
            [process_state_dict.get(c, 0.0) for c in self.QUERY_FEATURES],
            dtype=np.float64,
        ).reshape(1, -1)

        row_scaled = self._scaler.transform(row)

        distances, indices = self._nn.kneighbors(row_scaled)   # (1, k)
        distances = distances[0]   # (k,)
        indices   = indices[0]     # (k,)

        # Distance-weighted average (inverse-distance weighting)
        # Guard against exact matches (distance == 0)
        if distances.min() < 1e-10:
            weights = np.zeros(self.n_neighbors)
            weights[distances.argmin()] = 1.0
        else:
            inv_dist = 1.0 / distances
            weights  = inv_dist / inv_dist.sum()

        latent = (self._latents[indices] * weights[:, None]).sum(axis=0)
        return latent.astype(np.float32)

    # ── Cache building ────────────────────────────────────────────────────────

    def _build_cache(
        self,
        csv_path: Union[str, Path],
        raman_encoder,
        device,
        cache_path: Path,
    ):
        """
        Encode all valid Raman rows in the CSV and persist to .npz.

        Valid rows: those with non-zero Raman signal energy
        (matches the check in RamanEncoder.encode_dataframe).

        Returns
        -------
        query_arr   : (N_valid, 23) float32
        latents_arr : (N_valid, 64) float32
        """
        import pandas as pd

        print(f"  Loading CSV {csv_path} ...")
        df = pd.read_csv(csv_path)
        print(f"  CSV shape: {df.shape[0]:,} rows × {df.shape[1]} cols")

        # ── Identify valid Raman rows ─────────────────────────────────────────
        from src.data.dataset import RAMAN_START_COL_IDX, RAMAN_N_COLS

        raman_cols = df.columns[RAMAN_START_COL_IDX : RAMAN_START_COL_IDX + RAMAN_N_COLS]
        raw_raman  = df[raman_cols].values.astype(np.float32)
        valid_mask = np.nansum(np.abs(raw_raman), axis=1) > 0
        n_valid    = valid_mask.sum()

        print(f"  Valid Raman rows: {n_valid:,} / {len(df):,}")

        # ── Query features (process columns only, from valid rows) ────────────
        query_arr = df.loc[valid_mask, self.QUERY_FEATURES].values.astype(np.float32)

        # ── Encode Raman spectra ──────────────────────────────────────────────
        print("  Encoding Raman spectra (this may take ~5 minutes) ...")
        latents_all = raman_encoder.encode_dataframe(df)   # (N, 64)
        latents_arr = latents_all[valid_mask]              # (N_valid, 64)

        # ── Persist ───────────────────────────────────────────────────────────
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(cache_path),
            query_features=query_arr,
            latents=latents_arr,
        )
        print(f"  Cache saved → {cache_path}")

        return query_arr, latents_arr
