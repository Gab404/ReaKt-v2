"""
src/algorithms/delta_cdae_pi_lstm.py
=====================================
DeltaCDAEPILSTMAlgorithm — fit/evaluate pipeline for the Delta-CDAE-PI-LSTM.

The model predicts the **Penicillin increment** ΔP[t] = P[t] − P[t−1] at
every time step instead of the absolute concentration P[t].

Data flow
---------
  CSV  →  PenicillinDataModule (use_raman=True, encoder=CDAERamanEncoderV2)
       →  batch DataFrames  with  _raman_0 … _raman_63  columns  (64D CDAE latents)
       →  DeltaCDAEPILSTMAlgorithm._build_dataset()
               MinMaxScale the 64D latents → X windows  (seq_len, 64)
               Compute delta_pen_norm[t] = pen_norm[t] − pen_norm[t−1]
               → y windows  (seq_len,)
       →  DeltaCDAEPILSTMModel (LSTM → [delta_pen_norm, r_net])
       →  compute_delta_cdae_pinn_loss  (data MSE + λ · direct mass-balance residual)

Training target
---------------
  delta_pen_norm[t] = pen_norm[t] − pen_norm[t−1]
  where pen_norm is the MinMaxScaler-normalised penicillin concentration
  (fitted on absolute P values from the training split).
  For t = 0 of each batch, delta_pen_norm[0] = 0 (no previous value known).

Evaluation reconstruction
--------------------------
  We predict delta_pen_norm[t] for t = seq_len−1 … T−1 (last-step of each
  sliding window). To recover absolute concentrations for metric computation:

    pen_norm_pred[seq_len−2]  ← actual pen_norm value (one-time anchor)
    pen_norm_pred[t]          ← pen_norm_pred[t−1] + delta_pred[t],  t ≥ seq_len−1

  Then inverse-transform back to g/L and compare against sparse offline
  measurement points (same evaluation protocol as all other models).

Physics loss
-------------
  Unlike the concentration-prediction model (CDAEPILSTMModel), where the
  physics loss finite-differences the predicted P trajectory, here the model
  directly outputs ΔP.  The constraint is therefore applied without shifting:
      delta_pred ≈ k_prod · r_net · dt_n
  This is a stricter, more direct enforcement of the mass balance.

Physics annealing
-----------------
  Same schedule as CDAEPILSTMAlgorithm:
  lambda ramps from lambda_start → lambda_end over lambda_anneal_epochs,
  starting after lambda_anneal_delay warmup epochs.
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
from src.data.dataset import TARGET_COLS
from src.evaluation.metrics import compute_metrics
from src.models.delta_cdae_pi_lstm import DeltaCDAEPILSTMModel
from src.physics.losses import compute_delta_cdae_pinn_loss


class DeltaCDAEPILSTMAlgorithm(BaseAlgorithm):
    """
    Physics-Informed LSTM predicting Penicillin increments (ΔP) from CDAE latents.

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
    model.lstm_dropout      : 0.0
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

    name = "delta_cdae_pi_lstm"

    def __init__(self, cfg: Config, device: torch.device) -> None:
        super().__init__(cfg, device)
        self._dt_n: float = 0.0

    # ── Feature columns ───────────────────────────────────────────────────────

    def _feature_cols(self) -> List[str]:
        n_raman = int(self.cfg.data.get("raman_latent_dim", 64))
        return [f"_raman_{i}" for i in range(n_raman)]

    # ── Abstract implementations ──────────────────────────────────────────────

    def _build_model(self) -> DeltaCDAEPILSTMModel:
        m      = self.cfg.model
        n_feat = len(self._feature_cols())
        return DeltaCDAEPILSTMModel(
            input_size=n_feat,
            hidden_size=int(m.hidden_size),
            num_lstm_layers=int(m.num_lstm_layers),
            fc_hidden=int(m.fc_hidden),
            lstm_dropout=float(m.lstm_dropout),
            k_prod_init=float(m.get("k_prod_init", 1.0)),
        )

    def _fit_scalers(self, train_batches) -> ScalerBundle:
        """
        Fit MinMaxScalers for CDAE latent features and penicillin targets.

        The pen_scaler is fitted on **absolute** P values (same as cdae_pi_lstm).
        During dataset construction, delta_pen_norm[t] is computed as:
            pen_norm[t] − pen_norm[t−1]
        where pen_norm = pen_scaler.transform(P).  This keeps both the anchor
        value and the cumsum reconstructed predictions in the same [0,1] space
        as the pen_scaler, simplifying inverse-transform at evaluation time.
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

        # Normalised time step
        t_all    = np.concatenate([b["Time (h)"].values for b in train_batches])
        t_range  = float(t_all.max() - t_all.min())
        self._dt_n = float(self.cfg.training.dt) / max(t_range, 1e-8)

        print(
            f"  Scalers fitted  |  "
            f"latent_dim={X_all.shape[1]}  "
            f"n_train_rows={X_all.shape[0]:,}  "
            f"dt_n={self._dt_n:.6f}"
        )
        return ScalerBundle(feat=feat_scaler, bio=bio_scaler, pen=pen_scaler)

    def _build_dataset(
        self,
        batches,
        scalers: ScalerBundle,
    ) -> TensorDataset:
        """
        Build sliding-window TensorDataset where y = delta_pen_norm.

        Each window:
          X : (seq_len, n_feat=64)  — MinMaxScaled CDAE latents
          y : (seq_len,)            — delta_pen_norm (differences of normalised P)

        delta_pen_norm[t] = pen_norm[t] − pen_norm[t−1]
        For t=0 of each batch, delta_pen_norm[0] = 0 (no prior value available).
        """
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

            # Compute deltas; first step has no predecessor → delta = 0
            delta_pen_n = np.empty_like(pen_n)
            delta_pen_n[0]  = 0.0
            delta_pen_n[1:] = pen_n[1:] - pen_n[:-1]

            T = len(feats)
            for i in range(T - seq_len + 1):
                X_list.append(feats[i : i + seq_len])
                y_list.append(delta_pen_n[i : i + seq_len])

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

            pred = self.model(x_b)   # (B, T, 2): [delta_pen_norm, r_net]
            loss, ld = compute_delta_cdae_pinn_loss(
                model=self.model,
                pred=pred,
                y_delta_pen_norm=y_b,
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
        """Validation MSE on delta_pen_norm predictions."""
        val_mses: List[float] = []
        with torch.no_grad():
            for x_b, y_b in loader:
                x_b = x_b.to(self.device)
                y_b = y_b.to(self.device)
                pred = self.model(x_b)               # (B, T, 2)
                delta_pred = pred[:, :, 0]           # (B, T)
                val_mses.append(F.mse_loss(delta_pred, y_b).item())
        return float(np.mean(val_mses))

    def evaluate(self, batches) -> Dict[str, Any]:
        """
        Sliding-window inference with cumsum reconstruction and sparse-point
        evaluation.

        Evaluation protocol
        -------------------
        1. Build all sliding windows for each batch (vectorised).
        2. Run inference in chunks of 512; take the LAST timestep of each window
           → delta_pred[t] for t = seq_len−1 … T−1.
        3. Reconstruct absolute penicillin:
             anchor:  pen_norm_pred[seq_len−2] = pen_norm_actual[seq_len−2]
             cumsum:  pen_norm_pred[t] = pen_norm_pred[t−1] + delta_pred[t]
        4. Inverse-transform to g/L.
        5. Compute RMSE / MAE / R² only at sparse offline measurement points.
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
                pen_n  = self.scalers.pen.transform(
                    b[TARGET_COLS[1]].values.reshape(-1, 1)
                ).flatten().astype(np.float32)   # actual pen_norm (for anchor)
                sp_pen = b["_penicillin_sparse"].values
                sp_bio = b["_biomass_sparse"].values
                T      = len(feats)

                if T < seq_len:
                    continue

                # ── Build all sliding windows ─────────────────────────────────
                N_win   = T - seq_len + 1
                windows = np.stack([feats[i : i + seq_len] for i in range(N_win)])
                x_all   = torch.tensor(
                    windows, dtype=torch.float32, device=self.device
                )

                # ── Chunked inference → last timestep delta_pred ──────────────
                delta_preds_n = np.full(T, np.nan)
                for start in range(0, N_win, 512):
                    chunk      = x_all[start : start + 512]
                    out_delta  = self.model(chunk)[:, -1, 0].cpu().numpy()
                    s = seq_len - 1 + start
                    delta_preds_n[s : s + len(out_delta)] = out_delta

                # ── Cumsum reconstruction ─────────────────────────────────────
                # Anchor at t = seq_len - 2 (actual pen_norm, last step before
                # the first prediction window ends).
                anchor_t       = seq_len - 2
                anchor_val     = pen_n[anchor_t]   # actual normalised P
                pen_norm_pred  = np.full(T, np.nan)
                pen_norm_pred[anchor_t] = anchor_val

                for t in range(seq_len - 1, T):
                    if not np.isnan(delta_preds_n[t]):
                        prev = pen_norm_pred[t - 1]
                        if not np.isnan(prev):
                            pen_norm_pred[t] = prev + delta_preds_n[t]

                # ── Inverse-transform to g/L ──────────────────────────────────
                valid_mask    = ~np.isnan(pen_norm_pred)
                pred_pen_full = np.full(T, np.nan)
                if valid_mask.any():
                    pred_pen_full[valid_mask] = self.scalers.pen.inverse_transform(
                        pen_norm_pred[valid_mask].reshape(-1, 1)
                    ).flatten()

                y_preds_pen.append(pred_pen_full)
                y_sparses_pen.append(sp_pen.copy())

                # ── Sparse-point metrics ──────────────────────────────────────
                mask_pen = ~np.isnan(sp_pen) & ~np.isnan(pred_pen_full)
                if mask_pen.sum() > 0:
                    pen_true.append(sp_pen[mask_pen])
                    pen_pred_list.append(pred_pen_full[mask_pen])

                # ── Biomass placeholder ───────────────────────────────────────
                y_preds_bio.append(np.full(T, np.nan))
                y_sparses_bio.append(sp_bio.copy())

        # ── Aggregate metrics ─────────────────────────────────────────────────
        if pen_true:
            pen_t_cat = np.concatenate(pen_true)
            pen_p_cat = np.concatenate(pen_pred_list)
            pen_metrics = compute_metrics(pen_t_cat, pen_p_cat)
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
        print("  [Biomass   ] N/A (Delta-CDAE-PI-LSTM predicts Penicillin increments only)")

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
        physics_params = self.model.physics_parameters()
        network_params = self.model.network_parameters()
        return torch.optim.AdamW([
            {
                "params":       network_params,
                "lr":           float(tr.lr),
                "weight_decay": float(tr.weight_decay),
            },
            {
                "params":       physics_params,
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

        lambda_physics = lam_start + (lam_end - lam_start) * frac
        return {"lambda_physics": lambda_physics}


# ── Register ──────────────────────────────────────────────────────────────────
REGISTRY[DeltaCDAEPILSTMAlgorithm.name] = DeltaCDAEPILSTMAlgorithm
