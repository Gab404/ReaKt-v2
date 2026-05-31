#!/usr/bin/env python3
"""
finetune_closed_loop.py
=======================
Closed-loop (autoregressive) fine-tuning for PI-LSTM and Neural ODE
bioreactor models via Backpropagation Through Time (BPTT).

WHY
---
Both model families were trained with teacher forcing: at every step the
model received ground-truth inputs/states. Under MPC the model's own
predictions are fed back recursively, compounding errors over the horizon.
This script re-trains the dynamic part of each model so it is robust to
its own prediction drift.

ARCHITECTURE-SPECIFIC CLOSED-LOOP MECHANISM
--------------------------------------------
PI-LSTM  (BioreactorLogiPINN)
  • Input: (B, T, 23|87)  — process features only, bio/pen NOT in input.
  • The LSTM hidden tuple (h, c) is the model's state; it IS the carrier
    of prediction history across time steps.
  • Closed-loop training: carry (h, c) continuously across `horizon` steps
    with BPTT.  The LSTM cell is called one timestep at a time, passing
    (h, c) from step t to step t+1.
  • Scheduled sampling: at each BPTT step, detach (h, c) with probability
    `tf_ratio` — this truncates the gradient, mimicking teacher-forced
    per-window training.  As tf_ratio→0 full BPTT is enabled.

Neural ODE  (NeuralODEModel)
  • Input: y0=(B,2) explicit state [X_n, P_n] + U_grid controls.
  • Closed-loop training: at each horizon step the ODE is integrated for
    one dt, then y_pred is fed back as y0 for the next integration.
  • Scheduled sampling: y0_next = tf_ratio * y_true[t].detach()
                                + (1-tf_ratio) * y_pred_t
    At tf_ratio=1 always ground-truth (teacher forcing).
    At tf_ratio=0 always prediction (fully autoregressive BPTT).

SCHEDULED SAMPLING SCHEDULE
----------------------------
  epoch 0          → tf_ratio = 0.90   (90 % teacher forcing)
  epoch n_epochs-1 → tf_ratio = 0.00   (100 % autoregressive)
  Linear decay over epochs.

FEATURE INDEX REFERENCE  (PROCESS_FEATURE_COLS, 0-based)
---------------------------------------------------------
  0   Time (h)
  1   Fg   — aeration rate         [MPC-controlled]
  2   Fs   — sugar feed rate       [MPC-controlled]
  3   Fa   — acid flow
  4   Fb   — base flow
  5   Fc   — heating/cooling water
  6   Fh   — heating water
  7   Fw   — water for injection
  8   pressure (bar)
  9   Fremoved
  10  S    — substrate conc.       [process state]
  11  DO2  — dissolved O2          [process state]
  12  V    — vessel volume
  13  Wt   — vessel weight
  14  pH                           [process state]
  15  T    — temperature           [process state]
  16  Q    — generated heat
  17  CO2outgas
  18  Fpaa — PAA flow              [MPC-controlled]
  19  Foil — oil flow              [MPC-controlled]
  20  OUR
  21  O2
  22  CER
  23-86  _raman_0 … _raman_63      [Raman models only, frozen encoder]

Bio (X) and Penicillin (P) are TARGET_COLS — never present in the input
feature vector.  Their normalised predictions are [bio_norm, pen_norm],
i.e. model output index 0 and 1 respectively.

USAGE
-----
  # Fine-tune PI-LSTM
  python finetune_closed_loop.py \\
      --model pi_lstm \\
      --ckpt  checkpoints/pi_lstm.pt \\
      --horizon 20 --epochs 10 --lr 1e-4

  # Fine-tune Neural ODE (Raman variant)
  python finetune_closed_loop.py \\
      --model neural_ode_raman \\
      --ckpt  checkpoints/neural_ode_raman.pt \\
      --horizon 20 --epochs 15 --lr 5e-5

  # Disable scheduled sampling (jump straight to 100 % autoregressive)
  python finetune_closed_loop.py --model pi_lstm --ckpt ... \\
      --no-scheduled-sampling --horizon 20 --epochs 5
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Project imports  (script lives at repo root)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.algorithms import REGISTRY, BaseAlgorithm, ScalerBundle  # noqa: E402
from src.config import Config  # noqa: E402
from src.data.dataset import (  # noqa: E402
    FEATURE_PRESETS,
    PROCESS_FEATURE_COLS,
    PenicillinDataModule,
    RAMAN_LATENT_COLS,
    TARGET_COLS,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAMES = ["pi_lstm", "neural_ode", "pi_lstm_raman", "neural_ode_raman"]

_CONFIG_MAP = {
    "pi_lstm":          "./configs/pi_lstm.yaml",
    "neural_ode":       "./configs/neural_ode.yaml",
    "pi_lstm_raman":    "./configs/pi_lstm_raman.yaml",
    "neural_ode_raman": "./configs/neural_ode_raman.yaml",
}

# Physics parameter names inside BioreactorLogiPINN (need separate LR group)
_PINN_PHYSICS_PARAMS = {"_r_raw", "_ymax_raw", "_alpha_raw", "_beta_raw"}


# ===========================================================================
# CLI
# ===========================================================================

def _str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Closed-loop BPTT fine-tuning for PI-LSTM and Neural ODE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    p.add_argument("--model", required=True, choices=MODEL_NAMES,
                   help="Model architecture to fine-tune.")
    p.add_argument("--ckpt", required=True,
                   help="Path to the pretrained checkpoint (.pt).")

    # Fine-tuning hyper-parameters
    p.add_argument("--horizon", type=int, default=20,
                   help="Autoregressive unroll length for BPTT.")
    p.add_argument("--epochs", type=int, default=10,
                   help="Number of fine-tuning epochs.")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Base fine-tuning learning rate.")
    p.add_argument("--scheduled-sampling", type=_str2bool, default=True,
                   metavar="BOOL",
                   help="Anneal teacher-forcing ratio 0.9→0.0 if True.")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Mini-batch size.")
    p.add_argument("--grad-clip", type=float, default=5.0,
                   help="Max gradient norm (0 = disabled).")

    # Physics loss (PI-LSTM only)
    p.add_argument("--lambda-physics", type=float, default=-1.0,
                   help="Physics loss weight (PI-LSTM only). "
                        "-1 = read from checkpoint config.")

    # Dataset / splitting
    p.add_argument("--stride", type=int, default=-1,
                   help="Sliding stride for the fine-tuning dataset windows. "
                        "-1 = auto (warmup_len//2 for PI-LSTM, horizon//2 for NODE).")
    p.add_argument("--val-horizon", type=int, default=-1,
                   help="Autoregressive horizon used for validation loss. "
                        "-1 = same as --horizon.")

    # Infrastructure
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto",
                   help="'cpu', 'cuda', 'cuda:N', or 'auto'.")
    p.add_argument("--out", default=None,
                   help="Output checkpoint path (default: <model>_finetuned.pt).")
    p.add_argument("--config", default=None,
                   help="Override config YAML path.")
    p.add_argument("--es-patience", type=int, default=5,
                   help="Early-stopping patience (val epochs without improvement).")

    return p.parse_args()


# ===========================================================================
# Utilities
# ===========================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def teacher_forcing_ratio(epoch: int, n_epochs: int, use_ss: bool) -> float:
    """
    Linearly anneal teacher-forcing ratio from 0.90 (epoch 0) to 0.00 (last epoch).

    If `use_ss` is False the ratio is always 0.0 (fully autoregressive from ep 0).
    """
    if not use_ss:
        return 0.0
    if n_epochs == 1:
        return 0.9
    return 0.9 * max(0.0, 1.0 - epoch / (n_epochs - 1))


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ===========================================================================
# Datasets
# ===========================================================================

class ClosedLoopDatasetPILSTM(Dataset):
    """
    Sliding-window dataset for PI-LSTM closed-loop fine-tuning.

    Each sample is a window of length (warmup_len + horizon):

        x : (warmup_len + horizon, n_feat)   normalised process (±Raman) features
        y : (warmup_len + horizon, 2)         normalised [bio_norm, pen_norm]

    The first `warmup_len` steps warm up the LSTM hidden state from ground
    truth (no BPTT loss applied there).  The subsequent `horizon` steps are
    the BPTT unroll window where the loss is computed.

    Parameters
    ----------
    batches    : preprocessed batch DataFrames from PenicillinDataModule
    scalers    : fitted ScalerBundle (feat / bio / pen)
    feat_cols  : ordered feature column names (len 23 or 87)
    warmup_len : LSTM warm-up length — must equal the model's seq_len
    horizon    : BPTT unroll length
    stride     : sliding stride (default = warmup_len // 2 ≈ 50 % overlap)
    """

    def __init__(
        self,
        batches:    List,
        scalers:    ScalerBundle,
        feat_cols:  List[str],
        warmup_len: int,
        horizon:    int,
        stride:     Optional[int] = None,
    ) -> None:
        self.warmup_len = warmup_len
        self.horizon    = horizon
        window_len      = warmup_len + horizon
        stride          = stride if (stride is not None and stride > 0) \
                          else max(1, warmup_len // 2)

        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []

        for b in batches:
            feats = scalers.feat.transform(
                b[feat_cols].values
            ).astype(np.float32)                                       # (T, n_feat)
            bio_n = scalers.bio.transform(
                b[TARGET_COLS[0]].values.reshape(-1, 1)
            ).flatten().astype(np.float32)
            pen_n = scalers.pen.transform(
                b[TARGET_COLS[1]].values.reshape(-1, 1)
            ).flatten().astype(np.float32)
            tgts = np.stack([bio_n, pen_n], axis=-1)                  # (T, 2)

            T = len(feats)
            if T < window_len:
                continue
            for start in range(0, T - window_len + 1, stride):
                xs.append(feats[start : start + window_len])
                ys.append(tgts [start : start + window_len])

        if not xs:
            raise RuntimeError(
                f"No windows produced (window_len={window_len}, stride={stride}). "
                "Try a smaller --horizon or check your data."
            )

        self._x = np.array(xs, dtype=np.float32)  # (N, window_len, n_feat)
        self._y = np.array(ys, dtype=np.float32)  # (N, window_len, 2)

        print(
            f"  [Dataset/PILSTM] {len(self._x):,} windows  "
            f"warmup={warmup_len}  horizon={horizon}  "
            f"stride={stride}  n_feat={self._x.shape[-1]}"
        )

    def __len__(self) -> int:
        return len(self._x)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self._x[idx]),  # (warmup+horizon, n_feat)
            torch.from_numpy(self._y[idx]),  # (warmup+horizon, 2)
        )


class ClosedLoopDatasetNeuralODE(Dataset):
    """
    Segment dataset for Neural ODE closed-loop fine-tuning.

    Each sample:

        U_seg : (horizon, n_ctrl)   normalised control/feature inputs
        y0    : (2,)                normalised initial state [X_n, P_n]
        y_tgt : (horizon, 2)        normalised target trajectory [X, P]

    Parameters
    ----------
    batches   : preprocessed batch DataFrames from PenicillinDataModule
    scalers   : fitted ScalerBundle
    feat_cols : ordered feature column names
    horizon   : segment length (= BPTT unroll length)
    stride    : sliding stride (default = horizon // 2 ≈ 50 % overlap)
    """

    def __init__(
        self,
        batches:   List,
        scalers:   ScalerBundle,
        feat_cols: List[str],
        horizon:   int,
        stride:    Optional[int] = None,
    ) -> None:
        self.horizon = horizon
        stride       = stride if (stride is not None and stride > 0) \
                       else max(1, horizon // 2)

        us:  List[np.ndarray] = []
        y0s: List[np.ndarray] = []
        ys:  List[np.ndarray] = []

        for b in batches:
            U = scalers.feat.transform(
                b[feat_cols].values
            ).astype(np.float32)                                       # (T, n_ctrl)
            bio_n = scalers.bio.transform(
                b[TARGET_COLS[0]].values.reshape(-1, 1)
            ).flatten().astype(np.float32)
            pen_n = scalers.pen.transform(
                b[TARGET_COLS[1]].values.reshape(-1, 1)
            ).flatten().astype(np.float32)
            y_all = np.stack([bio_n, pen_n], axis=-1)                 # (T, 2)

            T = len(b)
            if T < horizon:
                continue
            for start in range(0, T - horizon + 1, stride):
                us.append (U    [start : start + horizon])             # (horizon, n_ctrl)
                y0s.append(y_all[start])                               # (2,)
                ys.append (y_all[start : start + horizon])             # (horizon, 2)

        if not us:
            raise RuntimeError(
                f"No segments produced (horizon={horizon}, stride={stride}). "
                "Try a smaller --horizon or check your data."
            )

        self._U  = np.array(us,  dtype=np.float32)  # (N, horizon, n_ctrl)
        self._y0 = np.array(y0s, dtype=np.float32)  # (N, 2)
        self._y  = np.array(ys,  dtype=np.float32)  # (N, horizon, 2)

        print(
            f"  [Dataset/NODE]   {len(self._U):,} segments  "
            f"horizon={horizon}  stride={stride}  n_ctrl={self._U.shape[-1]}"
        )

    def __len__(self) -> int:
        return len(self._U)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self._U [idx]),   # (horizon, n_ctrl)
            torch.from_numpy(self._y0[idx]),   # (2,)
            torch.from_numpy(self._y [idx]),   # (horizon, 2)
        )


# ===========================================================================
# Physics residual helper  (PI-LSTM only)
# ===========================================================================

def pinn_physics_residual(
    model,
    preds: torch.Tensor,
    dt_n:  float,
) -> torch.Tensor:
    """
    Compute the Logi-PINN physics residual from a rollout prediction tensor.

    Replicates the physics terms from ``src/physics/losses.py`` but operates
    on an already-computed (B, horizon, 2) prediction tensor rather than
    calling model.forward() a second time.

    Parameters
    ----------
    model  : BioreactorLogiPINN  (must expose .r, .y_max, .alpha, .beta)
    preds  : (B, H, 2)  normalised [bio_hat, pen_hat] — must retain grad
    dt_n   : normalised dt  (= dt_physical / time_range)

    Returns
    -------
    Scalar physics MSE loss (differentiable w.r.t. model parameters).
    """
    if preds.shape[1] < 2:
        return preds.new_tensor(0.0)

    bio_seq = preds[:, :, 0]            # (B, H)
    pen_seq = preds[:, :, 1]            # (B, H)

    delta_bio = bio_seq[:, 1:] - bio_seq[:, :-1]   # (B, H-1)
    delta_pen = pen_seq[:, 1:] - pen_seq[:, :-1]   # (B, H-1)

    r_n   = model.r
    y_max = model.y_max
    alpha = model.alpha
    beta  = model.beta

    # Logistic growth ODE: ΔX ≈ r·X·(1 - X/X_max)·dt
    delta_bio_ode = (
        r_n * bio_seq[:, :-1] * (1.0 - bio_seq[:, :-1] / y_max) * dt_n
    )
    # Luedeking-Piret: ΔP ≈ α·ΔX + β·X·dt
    delta_pen_ode = alpha * delta_bio + beta * bio_seq[:, :-1] * dt_n

    return (
        F.mse_loss(delta_bio, delta_bio_ode)
        + F.mse_loss(delta_pen, delta_pen_ode)
    )


# ===========================================================================
# Training epoch — PI-LSTM
# ===========================================================================

def train_epoch_pilstm(
    model,
    loader:         DataLoader,
    optimizer:      torch.optim.Optimizer,
    device:         torch.device,
    warmup_len:     int,
    horizon:        int,
    tf_ratio:       float,
    grad_clip:      float,
    lambda_physics: float,
    dt_n:           float,
) -> Dict[str, float]:
    """
    One fine-tuning epoch for PI-LSTM.

    Autoregressive rollout
    ----------------------
    1. Warmup (no BPTT loss):
       Run model.lstm over the first `warmup_len` ground-truth steps to build
       a meaningful initial (h, c).  This phase uses torch.no_grad so the
       warmup itself does not consume GPU memory for backward.

    2. BPTT unroll over `horizon` steps:
       At each step t:
         a. Scheduled sampling  — detach (h, c) with probability `tf_ratio`.
            This truncates the gradient chain (simulating teacher forcing).
            At tf_ratio=0 gradients flow freely through the entire horizon.
         b. Forward one LSTM cell step:  lstm_out, (h,c) = model.lstm(x_t, (h,c))
         c. Pass through shared head and output heads to get y_hat_t.

    3. Loss: MSE over the full horizon, plus optional PINN physics term.
    """
    model.train()
    loss_totals: List[float] = []
    loss_data_l: List[float] = []
    loss_phys_l: List[float] = []

    for x_long, y_long in loader:
        # x_long : (B, warmup+horizon, n_feat)
        # y_long : (B, warmup+horizon, 2)
        x_long = x_long.to(device)
        y_long = y_long.to(device)

        optimizer.zero_grad(set_to_none=True)

        # ── Phase 1: Warmup — build initial (h, c) from ground truth ──────────
        # torch.no_grad so no backward graph is built for the warmup steps.
        with torch.no_grad():
            _, (h, c) = model.lstm(x_long[:, :warmup_len, :])
        # Detach to be explicit: warmup state is a fixed starting point.
        h = h.detach()
        c = c.detach()

        # ── Phase 2: BPTT over horizon steps ──────────────────────────────────
        preds: List[torch.Tensor] = []

        for t in range(horizon):
            # Scheduled sampling on hidden state:
            #   tf_ratio = 1.0 → always detach  (truncated BPTT, ≈ teacher forced)
            #   tf_ratio = 0.0 → never  detach  (full BPTT through all steps)
            if tf_ratio > 0.0 and torch.rand(1, device=device).item() < tf_ratio:
                h = h.detach()
                c = c.detach()

            # Single-step LSTM forward  (B, 1, n_feat) → (B, 1, hidden)
            x_t = x_long[:, warmup_len + t : warmup_len + t + 1, :]
            lstm_out, (h, c) = model.lstm(x_t, (h, c))

            # Shared projection + output heads
            z       = model.shared(lstm_out)                          # (B, 1, fc_hidden)
            bio_hat = model.head_bio(z)                               # (B, 1, 1)
            pen_hat = model.head_pen(z)                               # (B, 1, 1)
            y_hat   = torch.cat([bio_hat, pen_hat], dim=-1).squeeze(1)  # (B, 2)
            preds.append(y_hat)

        preds_t = torch.stack(preds, dim=1)       # (B, horizon, 2)
        targets = y_long[:, warmup_len:, :]       # (B, horizon, 2)

        # ── Loss ──────────────────────────────────────────────────────────────
        loss_data = F.mse_loss(preds_t, targets)

        if lambda_physics > 0.0 and horizon >= 2:
            loss_phys = pinn_physics_residual(model, preds_t, dt_n)
            loss      = loss_data + lambda_physics * loss_phys
        else:
            loss_phys = preds_t.new_tensor(0.0)
            loss      = loss_data

        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        loss_totals.append(loss.item())
        loss_data_l.append(loss_data.item())
        loss_phys_l.append(loss_phys.item())

    return {
        "train_loss": float(np.mean(loss_totals)),
        "train_data": float(np.mean(loss_data_l)),
        "train_phys": float(np.mean(loss_phys_l)),
    }


@torch.no_grad()
def val_epoch_pilstm(
    model,
    loader:     DataLoader,
    device:     torch.device,
    warmup_len: int,
    horizon:    int,
) -> float:
    """
    Validation for PI-LSTM.

    Always fully autoregressive (tf_ratio=0): (h, c) is never detached,
    giving the most pessimistic / honest estimate of closed-loop performance.
    """
    model.eval()
    total  = 0.0
    n_samp = 0

    for x_long, y_long in loader:
        x_long = x_long.to(device)
        y_long = y_long.to(device)
        B = x_long.shape[0]

        _, (h, c) = model.lstm(x_long[:, :warmup_len, :])

        preds: List[torch.Tensor] = []
        for t in range(horizon):
            x_t             = x_long[:, warmup_len + t : warmup_len + t + 1, :]
            lstm_out, (h, c) = model.lstm(x_t, (h, c))
            z               = model.shared(lstm_out)
            y_hat           = torch.cat(
                [model.head_bio(z), model.head_pen(z)], dim=-1
            ).squeeze(1)                                               # (B, 2)
            preds.append(y_hat)

        preds_t  = torch.stack(preds, dim=1)       # (B, horizon, 2)
        targets  = y_long[:, warmup_len:, :]
        total   += F.mse_loss(preds_t, targets).item() * B
        n_samp  += B

    return total / max(n_samp, 1)


# ===========================================================================
# Training epoch — Neural ODE
# ===========================================================================

def train_epoch_neural_ode(
    model,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    horizon:   int,
    tf_ratio:  float,
    grad_clip: float,
    dt:        float,
    solver:    str = "rk4",
) -> Dict[str, float]:
    """
    One fine-tuning epoch for Neural ODE.

    Autoregressive rollout
    ----------------------
    At each step t in [0, horizon):

      1. Integrate ODE for one dt step:
           traj = model(y0_current, [0, dt], U_t_repeated)
           y_pred_t = traj[:, -1, :]       # state at end of step

      2. Scheduled sampling for next y0:
           • with prob tf_ratio   : y0_next = y_true[t]   (teacher-forced, detached)
           • with prob 1-tf_ratio : y0_next = y_pred_t    (autoregressive, full BPTT)

    The two-point time grid [0, dt] with repeated control keeps the ODE
    integration self-consistent for single-step calls and matches the
    piecewise-linear interpolation in ODEFunc._interpolate_U.

    Loss: MSE over all horizon predictions vs ground-truth targets.
    """
    model.train()
    total_loss = 0.0
    n_samples  = 0

    # Reusable 2-point time span on the correct device
    t_step = torch.tensor([0.0, dt], dtype=torch.float32, device=device)

    for U_seg, y0_batch, y_tgt in loader:
        # U_seg    : (B, horizon, n_ctrl)
        # y0_batch : (B, 2)               — ground-truth initial state
        # y_tgt    : (B, horizon, 2)       — ground-truth target trajectory
        U_seg    = U_seg.to(device)
        y0_batch = y0_batch.to(device)
        y_tgt    = y_tgt.to(device)
        B        = U_seg.shape[0]

        optimizer.zero_grad(set_to_none=True)

        y0_current             = y0_batch        # start from ground truth
        preds: List[torch.Tensor] = []

        for t in range(horizon):
            # Replicate the control vector to fill the 2-point time grid.
            # Shape: (B, 2, n_ctrl) — same control at t=0 and t=dt.
            # ODEFunc interpolates linearly between them (constant control).
            U_t = U_seg[:, t, :].unsqueeze(1).expand(-1, 2, -1)      # (B, 2, n_ctrl)

            # Single-step ODE integration: y0_current → y at t=dt
            traj     = model(y0_current, t_step, U_t, method=solver)  # (B, 2, 2)
            y_pred_t = traj[:, -1, :]                                  # (B, 2)
            preds.append(y_pred_t)

            # ── Scheduled sampling for the next initial state ────────────────
            # Teacher forcing (detached ground truth) prevents gradients from
            # flowing back through y0 — the loss still drives model parameters
            # via the current step's computation graph.
            # Autoregressive (y_pred_t kept in graph) enables full BPTT.
            if tf_ratio > 0.0 and torch.rand(1, device=device).item() < tf_ratio:
                y0_current = y_tgt[:, t, :].detach()                  # teacher forced
            else:
                y0_current = y_pred_t                                  # autoregressive

        preds_t = torch.stack(preds, dim=1)                            # (B, horizon, 2)
        loss    = F.mse_loss(preds_t, y_tgt)

        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * B
        n_samples  += B

    return {"train_loss": total_loss / max(n_samples, 1)}


@torch.no_grad()
def val_epoch_neural_ode(
    model,
    loader:  DataLoader,
    device:  torch.device,
    horizon: int,
    dt:      float,
    solver:  str = "rk4",
) -> float:
    """
    Validation for Neural ODE — fully autoregressive (y_pred fed back as y0).
    """
    model.eval()
    total  = 0.0
    n_samp = 0

    t_step = torch.tensor([0.0, dt], dtype=torch.float32, device=device)

    for U_seg, y0_batch, y_tgt in loader:
        U_seg    = U_seg.to(device)
        y0_batch = y0_batch.to(device)
        y_tgt    = y_tgt.to(device)
        B        = U_seg.shape[0]

        y0_current = y0_batch
        preds: List[torch.Tensor] = []

        for t in range(horizon):
            U_t      = U_seg[:, t, :].unsqueeze(1).expand(-1, 2, -1)
            traj     = model(y0_current, t_step, U_t, method=solver)
            y_pred_t = traj[:, -1, :]
            preds.append(y_pred_t)
            y0_current = y_pred_t                                      # always autoregressive

        preds_t  = torch.stack(preds, dim=1)
        total   += F.mse_loss(preds_t, y_tgt).item() * B
        n_samp  += B

    return total / max(n_samp, 1)


# ===========================================================================
# Data loading
# ===========================================================================

def _load_batches(cfg: Config, device: torch.device):
    """
    Load and split the IndPenSim dataset.

    Raman encoder — if required — is loaded with frozen weights and used
    only to pre-compute latent columns; it is never added to the fine-tuning
    optimiser.

    Returns
    -------
    train_batches, val_batches : list[DataFrame]
    """
    use_raman     = bool(getattr(cfg.data, "use_raman", False))
    raman_encoder = None

    if use_raman:
        from src.data.raman_encoder import RamanEncoder

        raman_ckpt = str(cfg.data.raman_ckpt)
        print(f"  Loading Raman encoder (frozen): {raman_ckpt}")
        raman_encoder = RamanEncoder(raman_ckpt, device=device)
        # Verify encoder is frozen
        for p in raman_encoder.model.parameters():
            p.requires_grad_(False)

    dm = PenicillinDataModule(cfg.data, raman_encoder=raman_encoder)
    dm.load()

    strategy   = str(getattr(cfg.data, "split_strategy", "random"))
    train_frac = float(getattr(cfg.data, "train_frac",      0.80))
    seed       = int  (getattr(cfg.data, "seed",            42))

    train_b, val_b, _ = dm.get_splits(
        train_frac=train_frac, seed=seed, strategy=strategy
    )
    print(f"  Train batches: {len(train_b)}   Val batches: {len(val_b)}")
    return train_b, val_b


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    args       = parse_args()
    out_path   = args.out or f"{args.model}_finetuned.pt"
    cfg_path   = args.config or _CONFIG_MAP[args.model]
    val_horiz  = args.val_horizon if args.val_horizon > 0 else args.horizon

    set_seed(args.seed)
    device = resolve_device(args.device)

    is_pi_lstm    = "pi_lstm"    in args.model
    is_neural_ode = "neural_ode" in args.model

    # ── Banner ────────────────────────────────────────────────────────────────
    print("=" * 72)
    print("  Closed-Loop BPTT Fine-Tuning")
    print("=" * 72)
    print(f"  model             : {args.model}")
    print(f"  pretrained ckpt   : {args.ckpt}")
    print(f"  horizon           : {args.horizon}  (val_horizon={val_horiz})")
    print(f"  epochs            : {args.epochs}")
    print(f"  lr                : {args.lr}")
    print(f"  scheduled sampling: {args.scheduled_sampling}")
    print(f"  batch size        : {args.batch_size}")
    print(f"  grad clip         : {args.grad_clip}")
    print(f"  es patience       : {args.es_patience}")
    print(f"  device            : {device}")
    print(f"  output            : {out_path}")
    print()

    # ── 1. Load pretrained algorithm ──────────────────────────────────────────
    print("[1/5] Loading pretrained checkpoint ...")
    alg   = BaseAlgorithm.load(args.ckpt, device)
    model = alg.model
    model.to(device)

    # Feature columns — use the algorithm's own method so Raman flag is respected
    feat_cols = alg._feature_cols()
    n_feat    = len(feat_cols)
    use_raman = bool(getattr(alg.cfg.data, "use_raman", False))

    print(f"  Algorithm class   : {type(alg).__name__}  (alg.name='{alg.name}')")
    print(f"  Trainable params  : {count_params(model):,}")
    print(f"  Feature columns   : {n_feat}  "
          f"({'23 process + 64 Raman' if use_raman else '23 process only'})")

    # ── 2. Load config ────────────────────────────────────────────────────────
    # Use a freshly parsed YAML for clarity; the loaded alg.cfg is authoritative
    # for things like use_raman, seq_len, dt — these come from the checkpoint.
    cfg = alg.cfg   # Config object already in memory, guaranteed consistent

    # ── 3. Load dataset ───────────────────────────────────────────────────────
    print("\n[2/5] Loading dataset ...")
    train_batches, val_batches = _load_batches(cfg, device)

    # ── 4. Build fine-tuning datasets & loaders ───────────────────────────────
    print("\n[3/5] Building closed-loop datasets ...")

    user_stride = args.stride if args.stride > 0 else None

    if is_pi_lstm:
        warmup_len = int(cfg.training.seq_len)   # original training context length
        print(f"  PI-LSTM warmup_len = {warmup_len}  (= original seq_len)")

        train_ds = ClosedLoopDatasetPILSTM(
            train_batches, alg.scalers, feat_cols,
            warmup_len=warmup_len, horizon=args.horizon, stride=user_stride,
        )
        val_ds = ClosedLoopDatasetPILSTM(
            val_batches, alg.scalers, feat_cols,
            warmup_len=warmup_len, horizon=val_horiz, stride=user_stride,
        )

    else:  # Neural ODE
        dt = float(cfg.training.dt)
        print(f"  Neural ODE  dt = {dt}")

        train_ds = ClosedLoopDatasetNeuralODE(
            train_batches, alg.scalers, feat_cols,
            horizon=args.horizon, stride=user_stride,
        )
        val_ds = ClosedLoopDatasetNeuralODE(
            val_batches, alg.scalers, feat_cols,
            horizon=val_horiz, stride=user_stride,
        )

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=pin,
    )

    # ── 5. Optimiser & scheduler ──────────────────────────────────────────────
    print("\n[4/5] Building optimiser ...")
    wd = float(getattr(cfg.training, "weight_decay", 1e-4))

    if is_pi_lstm:
        # Preserve the two-group structure from original training.
        # Physics parameters (r, y_max, α, β) get 10× the network LR —
        # they are high-level scalars that need faster adaptation.
        physics_params = [
            p for n, p in model.named_parameters() if n in _PINN_PHYSICS_PARAMS
        ]
        network_params = [
            p for n, p in model.named_parameters() if n not in _PINN_PHYSICS_PARAMS
        ]
        optimizer = torch.optim.AdamW([
            {"params": network_params, "lr": args.lr,        "weight_decay": wd},
            {"params": physics_params, "lr": args.lr * 10.0, "weight_decay": 0.0},
        ])
        print(f"  Two param groups: network lr={args.lr:.1e}, "
              f"physics lr={args.lr*10:.1e}")

        # Physics loss configuration
        if args.lambda_physics < 0.0:
            lambda_physics = float(
                getattr(cfg.training, "lambda_physics_end", 0.1)
            )
        else:
            lambda_physics = args.lambda_physics

        # Normalised dt  (same formula as PILSTMAlgorithm._fit_scalers)
        t_range = float(alg.scalers.feat.data_range_[0])
        dt_n    = float(cfg.training.dt) / max(t_range, 1e-8)
        print(f"  Physics lambda={lambda_physics:.4f}  dt_n={dt_n:.6f}")

    else:  # Neural ODE — single param group
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=wd
        )
        print(f"  Single param group: lr={args.lr:.1e}")
        dt             = float(cfg.training.dt)
        train_solver   = str(getattr(cfg.training, "train_solver", "rk4"))
        lambda_physics = 0.0
        dt_n           = 0.0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-7,
    )

    # ── 6. Training loop ──────────────────────────────────────────────────────
    print("\n[5/5] Fine-tuning ...")
    hdr = (f"  {'Ep':>4}  {'TF':>5}  {'train_loss':>11}  {'val_loss':>10}  "
           f"{'Δval':>9}  {'lr':>9}  {'time':>7}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    best_val   = float("inf")
    best_state: Optional[dict] = None
    patience   = 0
    history: Dict[str, list] = {
        "train_loss": [], "val_loss": [], "tf_ratio": [], "lr": [],
    }

    for epoch in range(args.epochs):
        tf_ratio = teacher_forcing_ratio(epoch, args.epochs, args.scheduled_sampling)
        t0 = time.time()

        # ── Train ──
        if is_pi_lstm:
            train_m = train_epoch_pilstm(
                model, train_loader, optimizer, device,
                warmup_len=warmup_len, horizon=args.horizon,
                tf_ratio=tf_ratio, grad_clip=args.grad_clip,
                lambda_physics=lambda_physics, dt_n=dt_n,
            )
        else:
            train_m = train_epoch_neural_ode(
                model, train_loader, optimizer, device,
                horizon=args.horizon, tf_ratio=tf_ratio,
                grad_clip=args.grad_clip, dt=dt, solver=train_solver,
            )

        # ── Validate ──
        if is_pi_lstm:
            val_loss = val_epoch_pilstm(
                model, val_loader, device,
                warmup_len=warmup_len, horizon=val_horiz,
            )
        else:
            val_loss = val_epoch_neural_ode(
                model, val_loader, device,
                horizon=val_horiz, dt=dt, solver=train_solver,
            )

        scheduler.step(val_loss)
        lr_now  = float(optimizer.param_groups[0]["lr"])
        elapsed = time.time() - t0
        delta   = val_loss - best_val

        improved = val_loss < best_val
        if improved:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience   = 0
        else:
            patience += 1

        # Accumulate history
        history["train_loss"].append(train_m["train_loss"])
        history["val_loss"].append(val_loss)
        history["tf_ratio"].append(tf_ratio)
        history["lr"].append(lr_now)
        # Extra PI-LSTM metrics
        for k in ("train_data", "train_phys"):
            if k in train_m:
                history.setdefault(k, []).append(train_m[k])

        # Log row
        star = "★" if improved else " "
        extra = ""
        if is_pi_lstm and "train_phys" in train_m and lambda_physics > 0:
            extra = f"  phys={train_m['train_phys']:.5f}"
        print(
            f"  {star}{epoch+1:3d}/{args.epochs}"
            f"  {tf_ratio:5.3f}"
            f"  {train_m['train_loss']:11.6f}"
            f"  {val_loss:10.6f}"
            f"  {delta:+9.5f}"
            f"  {lr_now:9.2e}"
            f"  {elapsed:6.1f}s"
            + extra
        )

        # Early stopping
        if patience >= args.es_patience:
            print(f"\n  [Early Stop] No improvement for {args.es_patience} epochs "
                  f"— stopped at epoch {epoch+1}.")
            break

    # ── Restore best weights ──────────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"\n  Best weights restored  (val_loss = {best_val:.6f})")
    model.eval()

    # ── Save fine-tuned checkpoint ────────────────────────────────────────────
    # Format is identical to BaseAlgorithm.save() so the checkpoint can be
    # consumed by BaseAlgorithm.load() and simulator.py without modification.
    #
    # Key:  algorithm_name = alg.name ("pi_lstm" or "neural_ode")
    #       NOT args.model — the REGISTRY lookup uses the base name.
    #       The cfg (with use_raman=True/False) encodes the Raman variant.
    ckpt_out = {
        "algorithm_name": alg.name,           # "pi_lstm" or "neural_ode"
        "model_state":    model.state_dict(),
        "scalers":        alg.scalers.to_dict(),
        "cfg":            cfg.to_dict(),
        "history":        {
            **alg._history,                    # original training history
            "finetune": history,               # closed-loop fine-tuning history
        },
        "finetune_args":  vars(args),          # provenance
    }
    torch.save(ckpt_out, out_path)
    print(f"\n  Fine-tuned checkpoint saved → {out_path}")

    # ── Print final summary ───────────────────────────────────────────────────
    print()
    print("  Fine-tuning summary")
    print("  " + "─" * 40)
    print(f"  Epochs run       : {min(epoch+1, args.epochs)}")
    print(f"  Best val loss    : {best_val:.6f}")
    if len(history["train_loss"]) >= 2:
        first = history["train_loss"][0]
        last  = history["train_loss"][-1]
        print(f"  Train loss       : {first:.6f} → {last:.6f}  "
              f"({100*(last-first)/max(abs(first),1e-9):+.1f} %)")
    print(f"  Checkpoint       : {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
