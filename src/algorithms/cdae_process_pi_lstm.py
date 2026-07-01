"""
src/algorithms/cdae_process_pi_lstm.py
=======================================
CDAEProcessPILSTMAlgorithm — PI-LSTM driven by 64-dim CDAE Raman latents
**concatenated with all 23 process state variables** (temperature, pH,
dissolved O2, substrate, aeration, feeds, …).

Data flow
---------
  CSV  →  PenicillinDataModule (use_raman=True, encoder=CDAERamanEncoderV2)
       →  batch DataFrames with _raman_0 … _raman_63  (CDAE latents)
                            and PROCESS_FEATURE_COLS   (23 process vars)
       →  CDAEProcessPILSTMAlgorithm._build_dataset()
               MinMaxScale the 87-dim (64 CDAE + 23 process) feature vector
               → X windows  (seq_len, 87)
               MinMaxScale Penicillin → y windows  (seq_len,)
       →  CDAEPILSTMModel(input_size=87)  — same architecture, wider input
       →  compute_cdae_pinn_loss  (data MSE + λ · mass-balance residual)

Rationale
---------
  cdae_pi_lstm uses Raman latents only (pure spectroscopic sensor).
  This variant adds all 23 process variables so the LSTM can condition
  on the full bioreactor state: temperature, pH, dissolved O2, substrate
  concentration, feed rates, vessel volume, off-gas composition, etc.
  This tests whether process context on top of the Raman signal improves
  concentration tracking, especially on fault batches where state variables
  carry diagnostic information.

Input feature order (87-dim)
-----------------------------
  Columns 0–63  : _raman_0 … _raman_63  (CDAE latent, MinMaxScaled)
  Columns 64–86 : PROCESS_FEATURE_COLS  (23 process vars, MinMaxScaled)

  Both halves are scaled by a **single** joint MinMaxScaler (one fit,
  no information about which half is which at inference time — the LSTM
  figures out the structure from the data).

Physics / training settings
----------------------------
  Identical to cdae_pi_lstm: simplified penicillin mass-balance
  physics loss (ΔP ≈ k_prod · r_net · dt), same annealing schedule,
  same two AdamW parameter groups.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from src.algorithms.base import REGISTRY, BaseAlgorithm, ScalerBundle
from src.config import Config
from src.data.dataset import PROCESS_FEATURE_COLS, TARGET_COLS
from src.evaluation.metrics import compute_metrics
from src.models.cdae_pi_lstm import CDAEPILSTMModel
from src.physics.losses import compute_cdae_pinn_loss


class CDAEProcessPILSTMAlgorithm(BaseAlgorithm):
    """
    PI-LSTM with frozen CDAE Raman encoder + all 23 process state variables.

    Config sections expected
    ------------------------
    data.use_raman          : true
    data.raman_encoder_type : "cdae_v2"
    data.raman_ckpt         : path to checkpoints/cdae_best.pt
    data.raman_scaler       : path to checkpoints/cdae_scaler.joblib
    data.raman_latent_dim   : 64
    model.hidden_size       : 64
    model.num_lstm_layers   : 2
    model.fc_hidden         : 32
    model.lstm_dropout      : 0.2
    model.k_prod_init       : 1.0
    training.seq_len        : 24
    training.dt             : 0.2
    training.batch_size     : 64
    training.n_epochs       : 200
    training.lr             : 1e-3
    training.lr_physics     : 1e-2
    training.weight_decay   : 1e-4
    training.lr_factor      : 0.5
    training.lr_patience    : 15
    training.es_patience    : 30
    training.grad_clip      : 5.0
    training.vanilla_mode           : false
    training.lambda_physics_start   : 0.0
    training.lambda_physics_end     : 0.1
    training.lambda_anneal_delay    : 30
    training.lambda_anneal_epochs   : 80
    """

    name = "cdae_process_pi_lstm"

    def __init__(self, cfg: Config, device: torch.device) -> None:
        super().__init__(cfg, device)
        self._dt_n: float = 0.0

    # ── Feature columns ───────────────────────────────────────────────────────

    def _raman_cols(self) -> List[str]:
        n_raman = int(self.cfg.data.get("raman_latent_dim", 64))
        return [f"_raman_{i}" for i in range(n_raman)]

    def _feature_cols(self) -> List[str]:
        """
        87-dim feature vector: 64 CDAE latents followed by 23 process vars.

        Ordering is stable across all calls so the joint MinMaxScaler stays
        aligned with the model input.
        """
        return self._raman_cols() + PROCESS_FEATURE_COLS

    # ── Abstract implementations ──────────────────────────────────────────────

    def _build_model(self) -> CDAEPILSTMModel:
        """Reuse CDAEPILSTMModel with input_size=87 (wider input, same arch)."""
        m      = self.cfg.model
        n_feat = len(self._feature_cols())   # 64 + 23 = 87
        return CDAEPILSTMModel(
            input_size=n_feat,
            hidden_size=int(m.hidden_size),
            num_lstm_layers=int(m.num_lstm_layers),
            fc_hidden=int(m.fc_hidden),
            lstm_dropout=float(m.lstm_dropout),
            k_prod_init=float(m.get("k_prod_init", 1.0)),
        )

    def _fit_scalers(self, train_batches) -> ScalerBundle:
        """
        Fit a joint MinMaxScaler on the full 87-dim feature vector.

        Fitting jointly keeps the scaler simple (one object to save/load)
        and avoids any leakage between the two feature groups.
        """
        feat_cols = self._feature_cols()

        X_all   = np.vstack([b[feat_cols].values for b in train_batches])
        bio_all = np.concatenate(
            [b[TARGET_COLS[0]].values for b in train_batches]
        ).reshape(-1, 1)
        pen_all = np.concatenate(
            [b[TARGET_COLS[1]].values for b in train_batches]
        ).reshape(-1, 1)

        feat_scaler = MinMaxScaler().fit(X_all)
        bio_scaler  = MinMaxScaler().fit(bio_all)
        pen_scaler  = MinMaxScaler().fit(pen_all)

        # dt_n from the Time (h) column (index 64 in the joint feature vector,
        # but easier to read directly from the batch)
        t_all    = np.concatenate([b["Time (h)"].values for b in train_batches])
        t_range  = float(t_all.max() - t_all.min())
        self._dt_n = float(self.cfg.training.dt) / max(t_range, 1e-8)

        print(
            f"  Scalers fitted  |  "
            f"input_dim={X_all.shape[1]}  "
            f"(raman={len(self._raman_cols())} + process={len(PROCESS_FEATURE_COLS)})  "
            f"n_train_rows={X_all.shape[0]:,}  "
            f"dt_n={self._dt_n:.6f}"
        )
        return ScalerBundle(feat=feat_scaler, bio=bio_scaler, pen=pen_scaler)

    def _build_dataset(
        self,
        batches,
        scalers: ScalerBundle,
    ) -> TensorDataset:
        feat_cols = self._feature_cols()
        seq_len   = int(self.cfg.training.seq_len)
        X_list, y_list = [], []

        for b in batches:
            feats = scalers.feat.transform(
                b[feat_cols].values
            ).astype(np.float32)
            pen_n = scalers.pen.transform(
                b[TARGET_COLS[1]].values.reshape(-1, 1)
            ).flatten().astype(np.float32)

            T = len(feats)
            for i in range(T - seq_len + 1):
                X_list.append(feats[i : i + seq_len])
                y_list.append(pen_n[i : i + seq_len])

        X = torch.tensor(np.array(X_list, dtype=np.float32))
        y = torch.tensor(np.array(y_list, dtype=np.float32))
        print(
            f"  Sliding-window dataset: {len(X_list):,} windows  "
            f"(seq_len={seq_len}, n_feat={X.shape[-1]})"
        )
        return TensorDataset(X, y)

    def _train_epoch(
        self,
        loader: DataLoader,
        lambda_physics: float = 0.0,
    ) -> Dict[str, float]:
        grad_clip = float(self.cfg.training.grad_clip)
        ep_total, ep_data, ep_phys = [], [], []

        for x_b, y_b in loader:
            x_b = x_b.to(self.device)
            y_b = y_b.to(self.device)

            self._optimizer.zero_grad(set_to_none=True)

            pred = self.model(x_b)   # (B, T, 2)
            loss, ld = compute_cdae_pinn_loss(
                model=self.model,
                pred=pred,
                y_pen_norm=y_b,
                dt_n=self._dt_n,
                lambda_physics=lambda_physics,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), grad_clip
            )
            self._optimizer.step()

            ep_total.append(ld["loss"])
            ep_data.append(ld["loss_data"])
            ep_phys.append(ld["loss_phys"])

        return {
            "train_loss":  float(np.mean(ep_total)),
            "train_data":  float(np.mean(ep_data)),
            "train_phys":  float(np.mean(ep_phys)),
            "lambda_phys": float(lambda_physics),
            "k_prod":      float(self.model.k_prod.item()),
        }

    def _val_epoch(self, loader: DataLoader) -> float:
        val_mses: List[float] = []
        with torch.no_grad():
            for x_b, y_b in loader:
                x_b = x_b.to(self.device)
                y_b = y_b.to(self.device)
                pred = self.model(x_b)
                pen_pred = pred[:, :, 0]
                val_mses.append(F.mse_loss(pen_pred, y_b).item())
        return float(np.mean(val_mses))

    def evaluate(self, batches) -> Dict[str, Any]:
        """
        Identical evaluation protocol to cdae_pi_lstm — sliding-window
        vectorised inference, last-timestep prediction, sparse-point metrics.
        """
        self.model.eval()
        feat_cols = self._feature_cols()
        seq_len   = int(self.cfg.training.seq_len)

        pen_true:      List[np.ndarray] = []
        pen_pred_list: List[np.ndarray] = []

        y_preds_pen:   List[np.ndarray] = []
        y_sparses_pen: List[np.ndarray] = []
        y_preds_bio:   List[np.ndarray] = []
        y_sparses_bio: List[np.ndarray] = []

        with torch.no_grad():
            for b in batches:
                feats  = self.scalers.feat.transform(
                    b[feat_cols].values
                ).astype(np.float32)
                sp_pen = b["_penicillin_sparse"].values
                sp_bio = b["_biomass_sparse"].values
                T      = len(feats)

                if T < seq_len:
                    continue

                N_win   = T - seq_len + 1
                windows = np.stack([feats[i : i + seq_len] for i in range(N_win)])
                x_all   = torch.tensor(
                    windows, dtype=torch.float32, device=self.device
                )

                preds_pen_n = np.full(T, np.nan)
                for start in range(0, N_win, 512):
                    chunk   = x_all[start : start + 512]
                    out_pen = self.model(chunk)[:, -1, 0].cpu().numpy()
                    s = seq_len - 1 + start
                    preds_pen_n[s : s + len(out_pen)] = out_pen

                valid_mask    = ~np.isnan(preds_pen_n)
                pred_pen_full = np.full(T, np.nan)
                if valid_mask.any():
                    pred_pen_full[valid_mask] = self.scalers.pen.inverse_transform(
                        preds_pen_n[valid_mask].reshape(-1, 1)
                    ).flatten()

                y_preds_pen.append(pred_pen_full)
                y_sparses_pen.append(sp_pen.copy())

                mask_pen = ~np.isnan(sp_pen) & ~np.isnan(pred_pen_full)
                if mask_pen.sum() > 0:
                    pen_true.append(sp_pen[mask_pen])
                    pen_pred_list.append(pred_pen_full[mask_pen])

                y_preds_bio.append(np.full(T, np.nan))
                y_sparses_bio.append(sp_bio.copy())

        if pen_true:
            pen_metrics = compute_metrics(
                np.concatenate(pen_true), np.concatenate(pen_pred_list)
            )
        else:
            pen_metrics = {"RMSE": float("nan"), "MAE": float("nan"),
                           "R2": float("nan"), "n": 0}

        bio_metrics = {"RMSE": float("nan"), "MAE": float("nan"),
                       "R2": float("nan"), "n": 0}

        print(
            f"  [Penicillin] "
            f"RMSE={pen_metrics.get('RMSE', float('nan')):.3f} g/L  "
            f"MAE={pen_metrics.get('MAE', float('nan')):.3f} g/L  "
            f"R2={pen_metrics.get('R2', float('nan')):.4f}  "
            f"n={pen_metrics.get('n', 0)}"
        )
        print(
            "  [Biomass   ] N/A "
            "(CDAE-Process-PI-LSTM predicts Penicillin only)"
        )

        return {
            "pen":           pen_metrics,
            "bio":           bio_metrics,
            "y_preds_pen":   y_preds_pen,
            "y_sparses_pen": y_sparses_pen,
            "y_preds_bio":   y_preds_bio,
            "y_sparses_bio": y_sparses_bio,
        }

    # ── Overrides ─────────────────────────────────────────────────────────────

    def _make_optimizer(self) -> torch.optim.Optimizer:
        tr = self.cfg.training
        return torch.optim.AdamW([
            {
                "params":       self.model.network_parameters(),
                "lr":           float(tr.lr),
                "weight_decay": float(tr.weight_decay),
            },
            {
                "params":       self.model.physics_parameters(),
                "lr":           float(tr.lr_physics),
                "weight_decay": 0.0,
            },
        ])

    def _is_es_active(self, epoch: int) -> bool:
        tr      = self.cfg.training
        vanilla = bool(getattr(tr, "vanilla_mode", False))
        if vanilla:
            return True
        delay = int(getattr(tr, "lambda_anneal_delay", 0))
        return epoch > delay

    def _epoch_extra_kwargs(self, epoch: int) -> Dict[str, Any]:
        tr      = self.cfg.training
        vanilla = bool(getattr(tr, "vanilla_mode", False))
        if vanilla:
            return {"lambda_physics": 0.0}

        delay         = int(getattr(tr, "lambda_anneal_delay", 0))
        anneal_epochs = max(int(getattr(tr, "lambda_anneal_epochs", 1)), 1)
        lam_start     = float(getattr(tr, "lambda_physics_start", 0.0))
        lam_end       = float(getattr(tr, "lambda_physics_end",   0.1))

        epoch_after_delay = epoch - delay
        if epoch_after_delay <= 0:
            frac = 0.0
        elif anneal_epochs <= 1:
            frac = 1.0
        else:
            frac = min((epoch_after_delay - 1) / (anneal_epochs - 1), 1.0)

        return {"lambda_physics": lam_start + (lam_end - lam_start) * frac}


# ── Register ──────────────────────────────────────────────────────────────────
REGISTRY[CDAEProcessPILSTMAlgorithm.name] = CDAEProcessPILSTMAlgorithm
