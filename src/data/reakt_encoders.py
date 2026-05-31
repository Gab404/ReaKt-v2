"""
src/data/reakt_encoders.py
==========================
Wrappers around the FusionModel V4 and CoAtNetFusion V5 Raman feature
extractors from the REAKT+ project.

Both classes expose the same interface as ``RamanEncoder``:

    enc = FusionModelV4Encoder(ckpt_path, scaler_path, device)
    enc = CoAtNetV5Encoder(ckpt_path, scaler_path, device)

    features = enc.encode(spectra)          # (N, feature_dim)  pre-scaled
    features = enc.encode_dataframe(df)     # (N, feature_dim)  raw CSV df

Key difference vs ``RamanEncoder`` (CDAE):
  - RamanEncoder uses per-spectrum min-max normalisation to [0, 1].
  - V4 / V5 encoders use a ``StandardScaler`` fitted on the training set,
    which must be supplied as a ``scaler_raman.pkl`` (joblib) file.

Feature dimensions:
  - FusionModelV4Encoder  → 512  (Conv branch: 32 channels × 16 pooled positions)
  - CoAtNetV5Encoder      → 32   (Attention branch: 32-dim after global avg pool)
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import joblib
import numpy as np
import torch
import torch.nn as nn

from src.data.dataset import RAMAN_N_COLS, RAMAN_START_COL_IDX


# ── Private architecture helpers ─────────────────────────────────────────────
# These are minimal reconstructions of the Raman branches from the REAKT+
# training scripts (train_CNN.py and train_coatnet.py).  Only the sub-graphs
# needed for feature extraction are included; prediction heads and physics
# branches are discarded.


def _build_v4_raman_branch() -> nn.Sequential:
    """Raman branch of FusionModel V4  →  output dim 512 (32 × 16)."""
    return nn.Sequential(
        nn.Conv1d(1, 16, kernel_size=11, stride=2, padding=5), nn.ReLU(),
        nn.MaxPool1d(3),
        nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(),
        nn.AdaptiveAvgPool1d(16),
        nn.Flatten(),   # 32 * 16 = 512
    )


class _PositionalEncoding1D(nn.Module):
    """Learnable positional embedding (same as CoAtNetFusion training code)."""

    def __init__(self, embed_dim: int, max_len: int = 500):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, D)
        return x + self.pe[:, : x.size(1), :]


class _SelfAttention1D(nn.Module):
    """Single-block multi-head self-attention (same as CoAtNetFusion training code)."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, D)
        out, _ = self.attention(x, x, x)
        return out


class _CoAtNetV5RamanBranch(nn.Module):
    """
    Raman branch of CoAtNetFusion V5  →  output dim 32.

    Forward path:
        x  (B, 1, 2200)
        → conv_stem           (B, 32, T')
        → transpose           (B, T', 32)
        → pos_encoder         (B, T', 32)
        → attention_block     (B, T', 32)
        → transpose + pool    (B, 32)
    """

    def __init__(self):
        super().__init__()
        self.conv_stem = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, stride=2, padding=5), nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(),
        )
        self.pos_encoder    = _PositionalEncoding1D(embed_dim=32)
        self.attention_block = _SelfAttention1D(embed_dim=32, num_heads=4)
        self.pool           = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_stem(x).transpose(1, 2)      # (B, T', 32)
        feat = self.pos_encoder(feat)
        feat = self.attention_block(feat).transpose(1, 2)  # (B, 32, T')
        return self.pool(feat).squeeze(-1)             # (B, 32)


# ── Public encoder classes ────────────────────────────────────────────────────


class FusionModelV4Encoder:
    """
    Frozen Raman feature extractor — ``raman_branch`` of FusionModel V4.

    Produces 512-dimensional features per spectrum via:
        Conv1d(1→16) → MaxPool → Conv1d(16→32) → AdaptiveAvgPool(16) → Flatten

    Preprocessing: ``StandardScaler`` loaded from a ``.pkl`` file (the same
    scaler used when training V4 in ``REAKT+/REAKT+/train_CNN.py``).

    Parameters
    ----------
    ckpt_path     : path to ``reakt_fusion_v4_best.pth``
    scaler_path   : path to ``scaler_raman.pkl`` (joblib StandardScaler)
    device        : torch device for inference
    input_length  : number of Raman wavenumber columns (default 2200)
    encode_batch  : mini-batch size for GPU encoding (default 1024)
    start_col_idx : 0-based column index where Raman spectra start in the CSV
                    (default ``RAMAN_START_COL_IDX`` = 39)
    """

    FEATURE_DIM: int = 512

    def __init__(
        self,
        ckpt_path:     Union[str, Path],
        scaler_path:   Union[str, Path],
        device:        torch.device,
        input_length:  int = RAMAN_N_COLS,
        encode_batch:  int = 1024,
        start_col_idx: int = RAMAN_START_COL_IDX,
    ):
        self.device        = device
        self.input_length  = input_length
        self.feature_dim   = self.FEATURE_DIM
        self.encode_batch  = encode_batch
        self.start_col_idx = start_col_idx

        # Load the StandardScaler that was used during V4 training.
        # The scaler's n_features_in_ may differ from input_length by ±1 due
        # to CSV version differences (REAKT+ CSV vs V3 CSV column counts).
        # We use the scaler's own feature count for all DataFrame slicing and
        # zero-pad/truncate as needed; the model handles variable-length input
        # via AdaptiveAvgPool1d(16) so this is safe.
        self._scaler = joblib.load(scaler_path)
        self._scaler_n_features: int = self._scaler.n_features_in_

        # Build standalone raman_branch and load weights from the full checkpoint.
        # The V4 state dict contains keys "raman_branch.0.weight", etc.  We strip
        # the "raman_branch." prefix so the keys match the bare Sequential.
        branch = _build_v4_raman_branch().to(device)
        full_state = torch.load(ckpt_path, map_location="cpu")
        if "model_state_dict" in full_state:
            full_state = full_state["model_state_dict"]

        prefix = "raman_branch."
        branch_state = {
            k[len(prefix):]: v
            for k, v in full_state.items()
            if k.startswith(prefix)
        }
        branch.load_state_dict(branch_state)

        for p in branch.parameters():
            p.requires_grad_(False)
        branch.eval()
        self._model = branch

    # ── Core methods ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode(self, spectra: np.ndarray) -> np.ndarray:
        """
        Extract features from StandardScaler-transformed spectra.

        Parameters
        ----------
        spectra  : (N, input_length) float32 array, already StandardScaler-
                   transformed (i.e. the output of ``self._scaler.transform()``)

        Returns
        -------
        features : (N, 512) float32 array
        """
        out = []
        for start in range(0, len(spectra), self.encode_batch):
            chunk = torch.tensor(
                spectra[start : start + self.encode_batch],
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(1)  # (B, 1, input_length)
            out.append(self._model(chunk).cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.feature_dim), dtype=np.float32)

    @torch.no_grad()
    def encode_dataframe(self, df) -> np.ndarray:
        """
        Encode all Raman spectra in a raw CSV DataFrame.

        Rows with no valid Raman spectrum (zero signal energy) receive a
        zero feature vector.

        Parameters
        ----------
        df : full pandas DataFrame (all CSV columns present)

        Returns
        -------
        features : (N, 512) float32 array  (same row order as df)
        """
        N = len(df)
        features = np.zeros((N, self.feature_dim), dtype=np.float32)

        # Extract exactly scaler.n_features_in_ Raman columns; zero-pad if
        # the DataFrame ends before that many columns are available.
        n_raman = self._scaler_n_features
        raman_cols_avail = df.columns[self.start_col_idx : self.start_col_idx + n_raman]
        n_avail = len(raman_cols_avail)

        raw_all = np.zeros((N, n_raman), dtype=np.float32)
        raw_all[:, :n_avail] = df[raman_cols_avail].values.astype(np.float32)
        valid_mask = np.nansum(np.abs(raw_all), axis=1) > 0

        n_valid = valid_mask.sum()
        if n_valid == 0:
            return features

        print(f"  Raman (V4): encoding {n_valid:,} / {N:,} rows ...")

        raw = raw_all[valid_mask]
        np.nan_to_num(raw, copy=False, nan=0.0)

        # StandardScaler transform — matches V4 training preprocessing
        scaled = self._scaler.transform(raw).astype(np.float32)

        features[valid_mask] = self.encode(scaled)
        return features

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def model(self) -> nn.Module:
        return self._model


class CoAtNetV5Encoder:
    """
    Frozen Raman feature extractor — Raman branch of CoAtNetFusion V5.

    Produces 32-dimensional features per spectrum via:
        Conv stem → Positional Encoding → Multi-head Self-Attention (4 heads)
        → Global Average Pool

    Preprocessing: ``StandardScaler`` loaded from a ``.pkl`` file (the same
    scaler used when training V5 in ``REAKT+/REAKT+/train_coatnet.py``).

    Parameters
    ----------
    ckpt_path     : path to ``reakt_coatnet_v5_best.pth``
    scaler_path   : path to ``scaler_raman.pkl`` (joblib StandardScaler)
    device        : torch device for inference
    input_length  : number of Raman wavenumber columns (default 2200)
    encode_batch  : mini-batch size for GPU encoding (default 1024)
    start_col_idx : 0-based column index where Raman spectra start in the CSV
                    (default ``RAMAN_START_COL_IDX`` = 39)
    """

    FEATURE_DIM: int = 32

    def __init__(
        self,
        ckpt_path:     Union[str, Path],
        scaler_path:   Union[str, Path],
        device:        torch.device,
        input_length:  int = RAMAN_N_COLS,
        encode_batch:  int = 1024,
        start_col_idx: int = RAMAN_START_COL_IDX,
    ):
        self.device        = device
        self.input_length  = input_length
        self.feature_dim   = self.FEATURE_DIM
        self.encode_batch  = encode_batch
        self.start_col_idx = start_col_idx

        # Load the StandardScaler that was used during V5 training.
        # See FusionModelV4Encoder for notes on scaler/CSV column count handling.
        self._scaler = joblib.load(scaler_path)
        self._scaler_n_features: int = self._scaler.n_features_in_

        # Build the V5 Raman branch module and load matching weights.
        # The V5 state dict mixes Raman-side keys (conv_stem.*, pos_encoder.*,
        # attention_block.*) with physics-side keys (phys_branch.*) and the
        # fusion head (fusion_head.*).  We load only the Raman-side keys.
        branch = _CoAtNetV5RamanBranch().to(device)
        full_state = torch.load(ckpt_path, map_location="cpu")
        if "model_state_dict" in full_state:
            full_state = full_state["model_state_dict"]

        raman_top_level = {"conv_stem", "pos_encoder", "attention_block", "pool"}
        branch_state = {
            k: v
            for k, v in full_state.items()
            if k.split(".")[0] in raman_top_level
        }
        branch.load_state_dict(branch_state)

        for p in branch.parameters():
            p.requires_grad_(False)
        branch.eval()
        self._model = branch

    # ── Core methods ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode(self, spectra: np.ndarray) -> np.ndarray:
        """
        Extract features from StandardScaler-transformed spectra.

        Parameters
        ----------
        spectra  : (N, input_length) float32 array, already StandardScaler-
                   transformed (i.e. the output of ``self._scaler.transform()``)

        Returns
        -------
        features : (N, 32) float32 array
        """
        out = []
        for start in range(0, len(spectra), self.encode_batch):
            chunk = torch.tensor(
                spectra[start : start + self.encode_batch],
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(1)  # (B, 1, input_length)
            out.append(self._model(chunk).cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.feature_dim), dtype=np.float32)

    @torch.no_grad()
    def encode_dataframe(self, df) -> np.ndarray:
        """
        Encode all Raman spectra in a raw CSV DataFrame.

        Rows with no valid Raman spectrum (zero signal energy) receive a
        zero feature vector.

        Parameters
        ----------
        df : full pandas DataFrame (all CSV columns present)

        Returns
        -------
        features : (N, 32) float32 array  (same row order as df)
        """
        N = len(df)
        features = np.zeros((N, self.feature_dim), dtype=np.float32)

        # Extract exactly scaler.n_features_in_ Raman columns; zero-pad if
        # the DataFrame ends before that many columns are available.
        n_raman = self._scaler_n_features
        raman_cols_avail = df.columns[self.start_col_idx : self.start_col_idx + n_raman]
        n_avail = len(raman_cols_avail)

        raw_all = np.zeros((N, n_raman), dtype=np.float32)
        raw_all[:, :n_avail] = df[raman_cols_avail].values.astype(np.float32)
        valid_mask = np.nansum(np.abs(raw_all), axis=1) > 0

        n_valid = valid_mask.sum()
        if n_valid == 0:
            return features

        print(f"  Raman (V5): encoding {n_valid:,} / {N:,} rows ...")

        raw = raw_all[valid_mask]
        np.nan_to_num(raw, copy=False, nan=0.0)

        # StandardScaler transform — matches V5 training preprocessing
        scaled = self._scaler.transform(raw).astype(np.float32)

        features[valid_mask] = self.encode(scaled)
        return features

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def model(self) -> nn.Module:
        return self._model
