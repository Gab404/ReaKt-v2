"""
train_autoencoder.py
====================
Production training pipeline for the CDAE_Raman and CVAE_Raman
autoencoder architectures (src/autoencoder/model.py).

Preprocessing pipeline (strict adherence to IndPenSim paper)
-------------------------------------------------------------
  1.  Load raw Raman spectra from the IndPenSim V3 CSV
        shape: (N, RAMAN_N_COLS)  -- e.g. (N, 2001)
  2.  Savitzky-Golay first-derivative filter (per-spectrum, axis=1)
        scipy.signal.savgol_filter(X, window_length=15, polyorder=2, deriv=1)
  3.  StandardScaler fitted on training data only  (zero mean, unit variance)
  4.  Reshape to PyTorch tensors  (B, 1, 2001)  -- channel-first for Conv1d

Model training
--------------
  CDAE: Forward pass adds Gaussian noise N(0, noise_std) internally.
        Loss = MSE(reconstruction, clean_target).

  CVAE: Forward pass returns (x_hat, mu, log_var).
        Loss = MSE(x_hat, x) + beta * KL(q(z|x) || N(0,I)).
        KL annealing: beta linearly increases from 0 to BETA_MAX over the
        first KL_WARMUP_EPOCHS epochs to prevent posterior collapse.

Usage
-----
  # Train CDAE (default)
  python train_autoencoder.py --csv path/to/100_Batches.csv --model cdae

  # Train CVAE
  python train_autoencoder.py --csv path/to/100_Batches.csv --model cvae

  # Override any CONFIG constant
  python train_autoencoder.py --model cvae --epochs 200 --batch_size 256 --lr 5e-4
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

# ── Project imports ───────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.autoencoder.model import (
    CDAE_Raman,
    CVAE_Raman,
    cdae_loss,
    vae_loss_function,
)
from src.data.dataset import RAMAN_START_COL_IDX

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── ─────────────────────────────────────────────────────────────────────────
# CONFIGURATION  -- override via CLI or by editing these constants
# ── ─────────────────────────────────────────────────────────────────────────

CSV_PATH:      str   = "./100_Batches_IndPenSim_V3.csv"
MODEL_TYPE:    str   = "cdae"    # "cdae" or "cvae"

# Raman spectral window
N_RAMAN_COLS:  int   = 2001      # wavenumber channels after SG filtering

# Savitzky-Golay preprocessing
SG_WINDOW:     int   = 15
SG_POLY:       int   = 2
SG_DERIV:      int   = 1         # first derivative

# Architecture
LATENT_DIM:    int   = 64
NOISE_STD:     float = 0.1       # CDAE Gaussian noise std

# Training
EPOCHS:        int   = 150
BATCH_SIZE:    int   = 256   # 256 fully utilises RTX tensor cores; was 128
LR:            float = 1e-3
WEIGHT_DECAY:  float = 1e-4
VAL_FRAC:      float = 0.15      # fraction of data used as validation
SEED:          int   = 42

# AMP (Automatic Mixed Precision) -- float16 on CUDA tensor cores (~2x faster)
USE_AMP:       bool  = True      # set False for CPU or older GPUs without fp16

# Learning-rate scheduler (ReduceLROnPlateau)
LR_PATIENCE:   int   = 10
LR_FACTOR:     float = 0.5
LR_MIN:        float = 1e-6

# Early stopping
ES_PATIENCE:   int   = 25
ES_MIN_DELTA:  float = 1e-5

# CVAE-specific
BETA_MAX:      float = 0.001     # maximum KL weight (beta-VAE)
KL_WARMUP:     int   = 30        # epochs over which beta ramps from 0 -> BETA_MAX

# Checkpointing
OUTPUT_DIR:    Path  = Path("./outputs/autoencoder")
CKPT_DIR:      Path  = Path("./checkpoints")


# ── ─────────────────────────────────────────────────────────────────────────
# DATASET
# ── ─────────────────────────────────────────────────────────────────────────

class RamanSpectralDataset(Dataset):
    """
    PyTorch Dataset wrapping a pre-processed (SG + StandardScaler) matrix
    of Raman spectra.

    Returns individual spectra as tensors of shape (1, L) -- channel-first,
    ready for Conv1d layers.  The channel dimension is added here so that
    the DataLoader stacks them into (B, 1, L) batches automatically.

    Parameters
    ----------
    spectra : np.ndarray  (N, L)  -- scaled, float32 spectra
    """

    def __init__(self, spectra: np.ndarray) -> None:
        # Store as float32 tensor; keep channel dim implicit -- added in __getitem__
        self._data = torch.from_numpy(spectra.astype(np.float32))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Add channel dimension: (L,) -> (1, L)
        return self._data[idx].unsqueeze(0)


# ── ─────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ── ─────────────────────────────────────────────────────────────────────────

def load_raman_from_csv(
    csv_path: str,
    n_raman:  int = N_RAMAN_COLS,
    start_idx: int = RAMAN_START_COL_IDX,
) -> np.ndarray:
    """
    Load raw Raman spectra from the IndPenSim V3 CSV.

    Only the Raman wavenumber columns are loaded; all other columns
    (process variables, targets, fault flag) are discarded.

    Parameters
    ----------
    csv_path  : path to 100_Batches_IndPenSim_V3.csv
    n_raman   : number of wavenumber channels to use
    start_idx : 0-based column index where Raman data begins (default 39)

    Returns
    -------
    X_raw : (N, n_raman)  float32 raw intensity matrix
            Rows where all channels are exactly 0 (instrument warm-up)
            are removed before returning.
    """
    logger.info("Loading Raman spectra from %s ...", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("  CSV shape: %d rows x %d cols", *df.shape)

    # Select n_raman channels by absolute column position (same as PLS / SVR scripts)
    all_cols = df.columns.tolist()
    avail    = len(all_cols) - start_idx
    n_use    = min(n_raman, avail)
    if n_use < n_raman:
        warnings.warn(
            f"CSV has only {avail} Raman columns from index {start_idx}; "
            f"using {n_use} instead of {n_raman}."
        )
    raman_cols = all_cols[start_idx : start_idx + n_use]
    X_raw = df[raman_cols].values.astype(np.float32)

    # Remove zero-energy rows (instrument warm-up rows at the start of each batch)
    energy_mask = np.abs(X_raw).sum(axis=1) > 0
    X_raw = X_raw[energy_mask]

    # Replace any NaN / Inf with 0 before filtering
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
    X_raw = np.clip(X_raw, a_min=0.0, a_max=None)

    logger.info(
        "  Raman matrix after removing zero rows: %d x %d", *X_raw.shape
    )
    return X_raw


def apply_sg_filter(X: np.ndarray) -> np.ndarray:
    """
    Apply a Savitzky-Golay first-derivative filter row-wise.

    This converts each raw intensity spectrum into a first-derivative
    spectrum, which suppresses additive baseline drift and fluorescence
    background while enhancing peak resolution.

    Parameters
    ----------
    X : (N, L) raw spectra

    Returns
    -------
    X_sg : (N, L) first-derivative spectra  (same shape, float64 -> cast to float32)
    """
    logger.info(
        "Applying SG filter (window=%d, poly=%d, deriv=%d) ...",
        SG_WINDOW, SG_POLY, SG_DERIV,
    )
    X_sg = savgol_filter(
        X.astype(np.float64),   # savgol_filter prefers float64
        window_length=SG_WINDOW,
        polyorder=SG_POLY,
        deriv=SG_DERIV,
        axis=1,                 # filter along wavenumber axis (columns)
    ).astype(np.float32)
    logger.info("  SG output shape: %s", X_sg.shape)
    return X_sg


def fit_and_scale(
    X_train_sg:    np.ndarray,
    X_val_sg:      np.ndarray,
    X_test_sg:     Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], StandardScaler]:
    """
    Fit a StandardScaler on the TRAINING data only and apply it to all splits.

    StandardScaler per-feature (per-wavenumber channel):
        X_scaled = (X - mean_train) / std_train

    This ensures the model sees data with approximately zero mean and unit
    variance per channel, which is critical for stable gradient flow through
    the deep encoder.

    IMPORTANT: The scaler is fitted on training data only.  Fitting on the
    full dataset would constitute data leakage.

    Parameters
    ----------
    X_train_sg : (N_train, L)
    X_val_sg   : (N_val,   L)
    X_test_sg  : (N_test,  L) or None

    Returns
    -------
    X_train_sc, X_val_sc, X_test_sc (or None), fitted StandardScaler
    """
    logger.info("Fitting StandardScaler on %d training spectra ...", len(X_train_sg))
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_sg).astype(np.float32)
    X_val_sc   = scaler.transform(X_val_sg).astype(np.float32)
    X_test_sc  = scaler.transform(X_test_sg).astype(np.float32) if X_test_sg is not None else None
    logger.info(
        "  Per-channel mean range: [%.4f, %.4f]",
        scaler.mean_.min(), scaler.mean_.max(),
    )
    return X_train_sc, X_val_sc, X_test_sc, scaler


def preprocess_spectra(
    csv_path:  str,
    val_frac:  float = VAL_FRAC,
    seed:      int   = SEED,
    n_raman:   int   = N_RAMAN_COLS,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Full preprocessing pipeline:
        CSV -> SG filter -> train/val split -> StandardScaler fit/transform

    Parameters
    ----------
    csv_path : path to CSV
    val_frac : fraction of spectra used for validation
    seed     : random seed for train/val split
    n_raman  : number of wavenumber channels

    Returns
    -------
    X_train : (N_train, n_raman)  scaled training spectra
    X_val   : (N_val,   n_raman)  scaled validation spectra
    scaler  : fitted StandardScaler (save alongside the checkpoint)
    """
    # 1. Load raw
    X_raw = load_raman_from_csv(csv_path, n_raman=n_raman)

    # 2. Savitzky-Golay first-derivative
    X_sg = apply_sg_filter(X_raw)

    # 3. Random train / val split (no batch structure needed here -- the
    #    autoencoder is trained on individual spectra, not sequences)
    rng     = np.random.default_rng(seed)
    indices = rng.permutation(len(X_sg))
    n_val   = max(1, int(len(X_sg) * val_frac))
    val_idx   = indices[:n_val]
    train_idx = indices[n_val:]

    X_train_sg = X_sg[train_idx]
    X_val_sg   = X_sg[val_idx]
    logger.info(
        "Split: train=%d  val=%d  (val_frac=%.2f)",
        len(X_train_sg), len(X_val_sg), val_frac,
    )

    # 4. StandardScaler (fit on train only)
    X_train_sc, X_val_sc, _, scaler = fit_and_scale(X_train_sg, X_val_sg)
    return X_train_sc, X_val_sc, scaler


# ── ─────────────────────────────────────────────────────────────────────────
# DATA LOADERS
# ── ─────────────────────────────────────────────────────────────────────────

def build_gpu_data_loaders(
    X_train:    np.ndarray,
    X_val:      np.ndarray,
    batch_size: int,
    device:     torch.device,
) -> Tuple[DataLoader, DataLoader]:
    """
    Move the entire dataset to GPU once and wrap in TensorDataset DataLoaders.

    Why this is the critical performance fix
    ----------------------------------------
    The original implementation used CPU tensors with pin_memory=True and
    num_workers=0.  On Windows, pin_memory allocates a new page-locked buffer
    per batch in the main thread, adding ~260 ms overhead per batch.
    With 750 batches per epoch, this alone accounts for ~3 minutes of dead
    time per epoch BEFORE any GPU computation starts.

    By moving all data to GPU upfront:
      * CPU->GPU transfer: once (at startup), not 750 times per epoch.
      * DataLoader indexing: GPU slice operation (< 1 ms per batch).
      * pin_memory=False: no pinned-memory allocation overhead.
      * num_workers=0: required -- worker processes cannot access CUDA tensors.

    Memory budget  (RTX 4050 6 GB VRAM)
    -------------------------------------
      X_train  95995 x 2001 x 4B = ~768 MB
      X_val    16940 x 2001 x 4B = ~135 MB
      Model + gradients           = ~  32 MB
      Activations (batch=256)     = ~  64 MB
      Total                       = ~1.0 GB  (<< 6 GB, safe)

    Returns
    -------
    train_loader, val_loader  (batches already on GPU -- no .to(device) needed)
    """
    vram_mb = (X_train.nbytes + X_val.nbytes) / 1e6
    logger.info(
        "Moving data to %s  (%.0f MB train + %.0f MB val = %.0f MB total) ...",
        device, X_train.nbytes / 1e6, X_val.nbytes / 1e6, vram_mb,
    )

    # (N, L) -> (N, 1, L)  channel-first, already float32
    t_train = torch.from_numpy(X_train).float().unsqueeze(1).to(device)
    t_val   = torch.from_numpy(X_val).float().unsqueeze(1).to(device)

    train_loader = DataLoader(
        TensorDataset(t_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,      # must be 0 -- workers cannot access CUDA tensors
        pin_memory=False,   # data is already on GPU; pin_memory would be a no-op
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(t_val),
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    logger.info(
        "DataLoaders ready: train=%d batches  val=%d batches  (batch_size=%d)",
        len(train_loader), len(val_loader), batch_size,
    )
    return train_loader, val_loader


# ── ─────────────────────────────────────────────────────────────────────────
# TRAINING LOOPS -- CDAE
# ── ─────────────────────────────────────────────────────────────────────────

def _train_one_epoch_cdae(
    model:      CDAE_Raman,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    scaler_amp: torch.cuda.amp.GradScaler,
    use_amp:    bool,
) -> float:
    """
    One training epoch for the CDAE with AMP and a tqdm progress bar.

    Key optimisations vs original
    ------------------------------
    * autocast:  forward pass runs in float16 on tensor cores (~2x faster conv)
    * GradScaler: prevents float16 gradient underflow
    * loss accumulated as a GPU tensor (no .item() per batch)
    * tqdm bar shows per-batch loss without blocking the GPU pipeline
    * no .to(device): data is already on GPU from build_gpu_data_loaders()
    """
    model.train()
    device     = next(model.parameters()).device
    running    = torch.zeros(1, device=device)   # GPU accumulator, no sync per batch

    pbar = tqdm(loader, desc="  train", leave=False, unit="b", dynamic_ncols=True)
    for (x_clean,) in pbar:                      # TensorDataset yields tuples
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            x_hat = model(x_clean)               # noise injected internally
            loss  = cdae_loss(x_hat, x_clean)

        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler_amp.step(optimizer)
        scaler_amp.update()

        running = running + loss.detach()        # stays on GPU, no sync
        pbar.set_postfix(loss=f"{loss.item():.5f}")   # one sync per bar refresh

    return (running / len(loader)).item()        # single GPU->CPU sync per epoch


@torch.no_grad()
def _val_one_epoch_cdae(
    model:   CDAE_Raman,
    loader:  DataLoader,
    use_amp: bool,
) -> float:
    """Validation epoch for CDAE (no noise, no gradients, AMP for speed)."""
    model.eval()
    device  = next(model.parameters()).device
    running = torch.zeros(1, device=device)

    for (x_clean,) in loader:
        with torch.cuda.amp.autocast(enabled=use_amp):
            x_hat = model(x_clean)
            running = running + cdae_loss(x_hat, x_clean).detach()

    return (running / len(loader)).item()


# ── ─────────────────────────────────────────────────────────────────────────
# TRAINING LOOPS -- CVAE
# ── ─────────────────────────────────────────────────────────────────────────

def _train_one_epoch_cvae(
    model:      CVAE_Raman,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    scaler_amp: torch.cuda.amp.GradScaler,
    use_amp:    bool,
    beta:       float,
) -> Tuple[float, float, float]:
    """Training epoch for CVAE with AMP + tqdm."""
    model.train()
    device = next(model.parameters()).device
    r_tot  = torch.zeros(1, device=device)
    r_rec  = torch.zeros(1, device=device)
    r_kl   = torch.zeros(1, device=device)

    pbar = tqdm(loader, desc="  train", leave=False, unit="b", dynamic_ncols=True)
    for (x,) in pbar:
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            x_hat, mu, log_var      = model(x)
            total, recon, kl        = vae_loss_function(x_hat, x, mu, log_var, beta=beta)

        scaler_amp.scale(total).backward()
        scaler_amp.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler_amp.step(optimizer)
        scaler_amp.update()

        r_tot = r_tot + total.detach()
        r_rec = r_rec + recon.detach()
        r_kl  = r_kl  + kl.detach()
        pbar.set_postfix(tot=f"{total.item():.4f}", kl=f"{kl.item():.3f}")

    n = len(loader)
    return (r_tot/n).item(), (r_rec/n).item(), (r_kl/n).item()


@torch.no_grad()
def _val_one_epoch_cvae(
    model:   CVAE_Raman,
    loader:  DataLoader,
    use_amp: bool,
    beta:    float,
) -> Tuple[float, float, float]:
    """Validation epoch for CVAE (deterministic: z = mu)."""
    model.eval()
    device = next(model.parameters()).device
    r_tot  = torch.zeros(1, device=device)
    r_rec  = torch.zeros(1, device=device)
    r_kl   = torch.zeros(1, device=device)

    for (x,) in loader:
        with torch.cuda.amp.autocast(enabled=use_amp):
            x_hat, mu, log_var = model(x)
            total, recon, kl   = vae_loss_function(x_hat, x, mu, log_var, beta=beta)
        r_tot = r_tot + total.detach()
        r_rec = r_rec + recon.detach()
        r_kl  = r_kl  + kl.detach()

    n = len(loader)
    return (r_tot/n).item(), (r_rec/n).item(), (r_kl/n).item()


# ── ─────────────────────────────────────────────────────────────────────────
# FULL TRAINING ORCHESTRATOR
# ── ─────────────────────────────────────────────────────────────────────────

def train_cdae(
    X_train:    np.ndarray,
    X_val:      np.ndarray,
    input_length: int   = N_RAMAN_COLS,
    latent_dim:   int   = LATENT_DIM,
    noise_std:    float = NOISE_STD,
    epochs:       int   = EPOCHS,
    batch_size:   int   = BATCH_SIZE,
    lr:           float = LR,
    weight_decay: float = WEIGHT_DECAY,
    es_patience:  int   = ES_PATIENCE,
    use_amp:      bool  = USE_AMP,
    device:       Optional[torch.device] = None,
    output_dir:   Path  = OUTPUT_DIR,
    ckpt_dir:     Path  = CKPT_DIR,
) -> Tuple[CDAE_Raman, Dict]:
    """
    Train the CDAE model end-to-end.

    Returns
    -------
    best_model : CDAE_Raman with the best validation weights loaded
    history    : dict of train/val loss curves
    """
    device  = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = use_amp and device.type == "cuda"   # AMP is CUDA-only
    logger.info("=== Training CDAE_Raman  device=%s  AMP=%s ===", device, use_amp)

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "cdae_best.pt"

    # -- Model, optimiser, scheduler, AMP scaler -----------------------------
    model = CDAE_Raman(
        input_length=input_length,
        latent_dim=latent_dim,
        noise_std=noise_std,
    ).to(device)

    optimizer  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=LR_MIN,
    )
    scaler_amp = torch.cuda.amp.GradScaler(enabled=use_amp)

    # -- GPU-native DataLoaders (data transferred to GPU once) ---------------
    train_loader, val_loader = build_gpu_data_loaders(
        X_train, X_val, batch_size, device
    )

    # -- Training loop -------------------------------------------------------
    history    = {"train_loss": [], "val_loss": [], "lr": []}
    best_val   = float("inf")
    no_improve = 0

    epoch_bar = tqdm(range(1, epochs + 1), desc="CDAE", unit="ep", dynamic_ncols=True)
    for epoch in epoch_bar:
        t0 = time.perf_counter()

        train_loss = _train_one_epoch_cdae(
            model, train_loader, optimizer, scaler_amp, use_amp
        )
        val_loss = _val_one_epoch_cdae(model, val_loader, use_amp)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        dt         = time.perf_counter() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        # Update the outer tqdm bar with epoch-level stats
        epoch_bar.set_postfix(
            train=f"{train_loss:.5f}",
            val=f"{val_loss:.5f}",
            lr=f"{current_lr:.1e}",
            s=f"{dt:.1f}",
        )
        logger.info(
            "Epoch %4d/%d  train=%.6f  val=%.6f  lr=%.2e  (%.1fs)",
            epoch, epochs, train_loss, val_loss, current_lr, dt,
        )

        # Checkpoint best model
        if val_loss < best_val - ES_MIN_DELTA:
            best_val   = val_loss
            no_improve = 0
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss":         best_val,
                    "input_length":     input_length,
                    "latent_dim":       latent_dim,
                    "noise_std":        noise_std,
                },
                ckpt_path,
            )
        else:
            no_improve += 1

        if no_improve >= es_patience:
            logger.info("Early stopping at epoch %d (best val=%.6f)", epoch, best_val)
            break

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    logger.info(
        "CDAE training complete.  Best val MSE=%.6f  checkpoint=%s",
        best_val, ckpt_path,
    )
    return model, history


def train_cvae(
    X_train:     np.ndarray,
    X_val:       np.ndarray,
    input_length: int   = N_RAMAN_COLS,
    latent_dim:   int   = LATENT_DIM,
    beta_max:     float = BETA_MAX,
    kl_warmup:    int   = KL_WARMUP,
    epochs:       int   = EPOCHS,
    batch_size:   int   = BATCH_SIZE,
    lr:           float = LR,
    weight_decay: float = WEIGHT_DECAY,
    es_patience:  int   = ES_PATIENCE,
    use_amp:      bool  = USE_AMP,
    device:       Optional[torch.device] = None,
    output_dir:   Path  = OUTPUT_DIR,
    ckpt_dir:     Path  = CKPT_DIR,
) -> Tuple[CVAE_Raman, Dict]:
    """
    Train the CVAE model with KL annealing to prevent posterior collapse.

    KL annealing schedule
    ---------------------
    beta starts at 0 (pure reconstruction) and ramps linearly to beta_max
    over the first kl_warmup epochs.  This allows the model to first learn
    a good reconstruction before the KL regularisation kicks in.

        beta(epoch) = min(beta_max, beta_max * epoch / kl_warmup)

    Returns
    -------
    best_model : CVAE_Raman with best validation weights
    history    : dict of train/val loss curves + component losses
    """
    device  = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = use_amp and device.type == "cuda"
    logger.info("=== Training CVAE_Raman  device=%s  AMP=%s ===", device, use_amp)

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "cvae_best.pt"

    model      = CVAE_Raman(input_length=input_length, latent_dim=latent_dim).to(device)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=LR_MIN,
    )
    scaler_amp = torch.cuda.amp.GradScaler(enabled=use_amp)

    train_loader, val_loader = build_gpu_data_loaders(
        X_train, X_val, batch_size, device
    )

    history = {
        "train_total": [], "train_recon": [], "train_kl": [],
        "val_total":   [], "val_recon":   [], "val_kl":   [],
        "beta": [],        "lr": [],
    }
    best_val   = float("inf")
    no_improve = 0

    epoch_bar = tqdm(range(1, epochs + 1), desc="CVAE", unit="ep", dynamic_ncols=True)
    for epoch in epoch_bar:
        t0   = time.perf_counter()
        beta = min(beta_max, beta_max * epoch / max(kl_warmup, 1))

        tr_tot, tr_rec, tr_kl = _train_one_epoch_cvae(
            model, train_loader, optimizer, scaler_amp, use_amp, beta
        )
        vl_tot, vl_rec, vl_kl = _val_one_epoch_cvae(
            model, val_loader, use_amp, beta
        )

        scheduler.step(vl_tot)
        current_lr = optimizer.param_groups[0]["lr"]
        dt         = time.perf_counter() - t0

        history["train_total"].append(tr_tot)
        history["train_recon"].append(tr_rec)
        history["train_kl"].append(tr_kl)
        history["val_total"].append(vl_tot)
        history["val_recon"].append(vl_rec)
        history["val_kl"].append(vl_kl)
        history["beta"].append(beta)
        history["lr"].append(current_lr)

        epoch_bar.set_postfix(
            vl=f"{vl_tot:.5f}",
            rec=f"{vl_rec:.5f}",
            kl=f"{vl_kl:.3f}",
            b=f"{beta:.4f}",
            s=f"{dt:.1f}",
        )
        logger.info(
            "Epoch %4d/%d  "
            "train[tot=%.5f rec=%.5f kl=%.4f]  "
            "val[tot=%.5f rec=%.5f kl=%.4f]  "
            "beta=%.5f  lr=%.2e  (%.1fs)",
            epoch, epochs,
            tr_tot, tr_rec, tr_kl,
            vl_tot, vl_rec, vl_kl,
            beta, current_lr, dt,
        )

        if vl_tot < best_val - ES_MIN_DELTA:
            best_val   = vl_tot
            no_improve = 0
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "val_total":        best_val,
                    "val_recon":        vl_rec,
                    "val_kl":           vl_kl,
                    "input_length":     input_length,
                    "latent_dim":       latent_dim,
                    "beta_max":         beta_max,
                },
                ckpt_path,
            )
        else:
            no_improve += 1

        if no_improve >= es_patience:
            logger.info("Early stopping at epoch %d (best ELBO=%.6f)", epoch, best_val)
            break

    # If no valid checkpoint was saved (e.g. all-NaN loss in early epochs),
    # save the final model state so torch.load() below never raises.
    if not ckpt_path.exists():
        logger.warning(
            "No checkpoint was saved during training (all losses were NaN or "
            "no improvement seen).  Saving final model state as fallback."
        )
        torch.save(
            {
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_total":        float("nan"),
                "input_length":     input_length,
                "latent_dim":       latent_dim,
                "beta_max":         beta_max,
            },
            ckpt_path,
        )

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    logger.info(
        "CVAE training complete.  Best val ELBO=%.6f  checkpoint=%s",
        best_val, ckpt_path,
    )
    return model, history


# ── ─────────────────────────────────────────────────────────────────────────
# SCALER PERSISTENCE
# ── ─────────────────────────────────────────────────────────────────────────

def save_scaler(scaler: StandardScaler, model_type: str, ckpt_dir: Path) -> Path:
    """Save the fitted StandardScaler alongside the model checkpoint."""
    path = ckpt_dir / f"{model_type}_scaler.joblib"
    joblib.dump(scaler, path)
    logger.info("Scaler saved -> %s", path)
    return path


def load_scaler(model_type: str, ckpt_dir: Path) -> StandardScaler:
    """Load a previously saved StandardScaler."""
    path = ckpt_dir / f"{model_type}_scaler.joblib"
    scaler = joblib.load(path)
    logger.info("Scaler loaded <- %s", path)
    return scaler


# ── ─────────────────────────────────────────────────────────────────────────
# MAIN
# ── ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CDAE or CVAE on IndPenSim Raman spectra.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv",        type=str,   default=CSV_PATH)
    parser.add_argument("--model",      type=str,   default=MODEL_TYPE,
                        choices=["cdae", "cvae"])
    parser.add_argument("--epochs",     type=int,   default=EPOCHS)
    parser.add_argument("--batch_size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LR)
    parser.add_argument("--latent_dim", type=int,   default=LATENT_DIM)
    parser.add_argument("--noise_std",  type=float, default=NOISE_STD,
                        help="CDAE Gaussian noise std")
    parser.add_argument("--beta_max",   type=float, default=BETA_MAX,
                        help="CVAE maximum KL beta weight")
    parser.add_argument("--kl_warmup",  type=int,   default=KL_WARMUP,
                        help="CVAE KL annealing warmup epochs")
    parser.add_argument("--val_frac",   type=float, default=VAL_FRAC)
    parser.add_argument("--seed",       type=int,   default=SEED)
    parser.add_argument("--device",     type=str,   default=None,
                        help="e.g. 'cuda:0' or 'cpu'")
    args = parser.parse_args()

    # Resolve device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Seeding for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # -- Preprocessing -------------------------------------------------------
    X_train, X_val, scaler = preprocess_spectra(
        csv_path=args.csv,
        val_frac=args.val_frac,
        seed=args.seed,
        n_raman=N_RAMAN_COLS,
    )
    logger.info(
        "Preprocessed: X_train=%s  X_val=%s",
        X_train.shape, X_val.shape,
    )

    # -- Training ------------------------------------------------------------
    if args.model == "cdae":
        model, history = train_cdae(
            X_train, X_val,
            input_length=N_RAMAN_COLS,
            latent_dim=args.latent_dim,
            noise_std=args.noise_std,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            use_amp=USE_AMP,
            device=device,
        )
    else:   # cvae
        model, history = train_cvae(
            X_train, X_val,
            input_length=N_RAMAN_COLS,
            latent_dim=args.latent_dim,
            beta_max=args.beta_max,
            kl_warmup=args.kl_warmup,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            use_amp=USE_AMP,
            device=device,
        )

    # -- Save scaler (must be saved alongside the checkpoint) ----------------
    save_scaler(scaler, args.model, CKPT_DIR)

    # -- Final summary -------------------------------------------------------
    logger.info("=== Training complete ===")
    if args.model == "cdae":
        best_val = min(history["val_loss"])
        logger.info("Best val MSE : %.6f  g/L^2", best_val)
    else:
        best_val = min(history["val_total"])
        best_ep  = history["val_total"].index(best_val) + 1
        logger.info(
            "Best val ELBO: %.6f  (epoch %d)  "
            "recon=%.6f  kl=%.6f",
            best_val, best_ep,
            history["val_recon"][best_ep - 1],
            history["val_kl"][best_ep - 1],
        )


if __name__ == "__main__":
    main()
