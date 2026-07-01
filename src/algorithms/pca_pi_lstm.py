"""
src/algorithms/pca_pi_lstm.py
=============================
PCAPILSTMAlgorithm -- full fit/evaluate pipeline for the PCA-PI-LSTM.

Data flow
---------
  CSV  ->  PenicillinDataModule (use_raman=True, encoder=PCARamanEncoderV1)
       ->  batch DataFrames  with  _raman_0 ... _raman_(K-1)  columns
           (K-dimensional PCA scores; default K=4)
       ->  PCAPILSTMAlgorithm._build_dataset()
              MinMaxScale the K-D PCA scores -> X windows  (seq_len, K)
              MinMaxScale Penicillin          -> y windows  (seq_len,)
       ->  CDAEPILSTMModel (LSTM -> [pen_norm, r_net])    [reused class]
       ->  compute_cdae_pinn_loss  (data MSE + lambda * mass-balance residual)

Why this exists
---------------
  This is the direct PCA counterpart of CDAEPILSTMAlgorithm.  It plugs a
  frozen PCA Raman encoder (linear, unsupervised) into the same LSTM +
  physics-loss head used by CDAE-PI-LSTM.  The intended scientific
  comparison is:

      static    : PLS  vs  PCA-PCR     -> supervised vs unsupervised
                                          linear reduction at K=4
      temporal  : CDAE-PI-LSTM  vs  PCA-PI-LSTM
                                       -> non-linear (CDAE) vs linear (PCA)
                                          unsupervised encoder feeding the
                                          same LSTM+physics head

Feature philosophy
------------------
  This model uses ONLY the K-D PCA scores extracted from the Raman spectra.
  No process variables (aeration, temperature, ...) are included.  This makes
  it a pure spectroscopic soft-sensor.

Sparse evaluation rule
----------------------
  Although the model is trained on DENSE interpolated targets (one value per
  0.2 h time step), evaluation metrics are computed ONLY at the sparse
  offline laboratory measurement time-points where _penicillin_sparse is
  not NaN (~20 measurements per 226 h batch, approx. every 12 h).
  This matches the evaluation convention for all other benchmark models.

Physics annealing
-----------------
  Lambda_physics ramps from lambda_physics_start to lambda_physics_end over
  lambda_anneal_epochs, starting after lambda_anneal_delay "data-only" warmup
  epochs.  Early stopping is frozen during the delay phase.

Model reuse
-----------
  The downstream temporal model is ``CDAEPILSTMModel`` from
  ``src.models.cdae_pi_lstm``, instantiated with ``input_size=K``
  (4 by default).  The class is encoder-agnostic; nothing CDAE-specific
  is baked in beyond the default value of input_size=64.  This keeps the
  PCA-vs-CDAE comparison strictly downstream-equivalent.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from src.algorithms.base import REGISTRY, BaseAlgorithm, ScalerBundle
from src.config import Config
from src.data.dataset import TARGET_COLS
from src.evaluation.metrics import compute_metrics
from src.models.cdae_pi_lstm import CDAEPILSTMModel   # reused: same LSTM+physics head
from src.physics.losses import compute_cdae_pinn_loss


class PCAPILSTMAlgorithm(BaseAlgorithm):
    """
    Physics-Informed LSTM algorithm driven exclusively by PCA Raman scores.

    Config sections expected
    ------------------------
    data.use_raman          : true        (required -- triggers PCA encoding)
    data.raman_encoder_type : "pca_v2"    (selects PCARamanEncoderV1)
    data.raman_ckpt         : path to checkpoints/pca_best.joblib
    data.raman_scaler       : path to checkpoints/pca_scaler.joblib
    data.raman_latent_dim   : 4           (number of PCA components)
    model.hidden_size       : 64
    model.num_lstm_layers   : 2
    model.fc_hidden         : 32
    model.lstm_dropout      : 0.0
    model.k_prod_init       : 1.0         (initial physics scale factor)
    training.seq_len        : 24
    training.dt             : 0.2
    training.batch_size     : 64
    training.n_epochs       : 200
    training.lr             : 1e-3        (network parameters)
    training.lr_physics     : 1e-2        (k_prod parameter)
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

    name = "pca_pi_lstm"

    def __init__(self, cfg: Config, device: torch.device) -> None:
        super().__init__(cfg, device)
        self._dt_n: float = 0.0   # normalised time step; filled in _fit_scalers

    # -- Feature columns -------------------------------------------------------

    def _feature_cols(self) -> List[str]:
        """
        Return the K PCA latent column names (K = raman_latent_dim, default 4).

        These columns are populated by PCARamanEncoderV1.encode_dataframe()
        during PenicillinDataModule.load() and are named _raman_0 ... _raman_(K-1).
        """
        n_raman = int(self.cfg.data.get("raman_latent_dim", 4))
        return [f"_raman_{i}" for i in range(n_raman)]

    # -- Abstract implementations ---------------------------------------------

    def _build_model(self) -> CDAEPILSTMModel:
        m      = self.cfg.model
        n_feat = len(self._feature_cols())
        return CDAEPILSTMModel(
            input_size=n_feat,        # K=4 for PCA-PI-LSTM (vs 64 for CDAE)
            hidden_size=int(m.hidden_size),
            num_lstm_layers=int(m.num_lstm_layers),
            fc_hidden=int(m.fc_hidden),
            lstm_dropout=float(m.lstm_dropout),
            k_prod_init=float(m.get("k_prod_init", 1.0)),
        )

    def _fit_scalers(self, train_batches) -> ScalerBundle:
        """
        Fit MinMaxScalers for PCA latent features and targets.

        The normalised time step (dt_n) cannot be derived from the feature
        scaler here (latent columns have no time index), so it is computed
        directly from the raw Time (h) column of the training batches.
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
        bio_scaler  = MinMaxScaler().fit(bio_all)   # kept for ScalerBundle compat
        pen_scaler  = MinMaxScaler().fit(pen_all)

        # Normalised time step: dt_n = dt_physical / time_range
        # (used in the mass-balance physics loss)
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
        Build sliding-window TensorDataset from a list of batch DataFrames.

        Each window:
          X : (seq_len, n_feat=K)  -- MinMaxScaled PCA scores
          y : (seq_len,)           -- MinMaxScaled Penicillin (dense)

        Both X and y are built from the dense (interpolated) targets so the
        LSTM learns continuous temporal dynamics.  Evaluation uses only the
        sparse measurement points (see evaluate()).
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
        """
        One training epoch.

        For each batch:
          1. Forward pass -> (B, T, 2) predictions [pen_norm, r_net]
          2. compute_cdae_pinn_loss -> data MSE + lambda * mass-balance residual
          3. Backward + gradient clip + optimizer step

        Returns epoch-level mean metrics for logging.
        """
        grad_clip = float(self.cfg.training.grad_clip)
        ep_total, ep_data, ep_phys = [], [], []

        for x_b, y_b in loader:
            # x_b : (B, seq_len, K)    -- MinMaxScaled PCA scores
            # y_b : (B, seq_len)       -- MinMaxScaled Penicillin (dense)
            x_b = x_b.to(self.device)
            y_b = y_b.to(self.device)

            self._optimizer.zero_grad(set_to_none=True)

            pred = self.model(x_b)   # (B, T, 2): [pen_norm, r_net]
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
        """Validation MSE on Penicillin predictions (no physics loss)."""
        val_mses: List[float] = []
        with torch.no_grad():
            for x_b, y_b in loader:
                x_b = x_b.to(self.device)
                y_b = y_b.to(self.device)
                pred = self.model(x_b)          # (B, T, 2)
                pen_pred = pred[:, :, 0]        # (B, T)   pen_norm only
                val_mses.append(F.mse_loss(pen_pred, y_b).item())
        return float(np.mean(val_mses))

    def evaluate(self, batches) -> Dict[str, Any]:
        """
        Sliding-window vectorised inference with sparse-point evaluation.

        Evaluation protocol
        -------------------
        1. Build ALL sliding windows for each batch at once (vectorised).
        2. Run inference in chunks of 512 to fit VRAM.
        3. Reconstruct the full-length prediction by taking the LAST timestep
           of each window -- this is the prediction for time step t, conditioned
           on the preceding seq_len steps.
        4. Inverse-transform Penicillin predictions back to g/L.
        5. Compute RMSE / MAE / R2 ONLY at the sparse offline measurement
           time-points (where _penicillin_sparse is not NaN).

        Biomass metrics are not computed (the model does not predict biomass)
        and are returned as NaN for benchmark table completeness.

        Returns
        -------
        dict with keys:
          "pen"          : {RMSE, MAE, R2, n}
          "bio"          : {RMSE=nan, MAE=nan, R2=nan, n=0}
          "y_preds_pen"  : list of per-batch full-length pen predictions (g/L)
          "y_sparses_pen": list of per-batch sparse pen labels (g/L)
          "y_preds_bio"  : list of per-batch full-length nan arrays (placeholder)
          "y_sparses_bio": list of per-batch sparse bio labels (g/L)
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
                sp_pen = b["_penicillin_sparse"].values       # NaN where unmeasured
                sp_bio = b["_biomass_sparse"].values          # for plot compat
                T      = len(feats)

                if T < seq_len:
                    continue

                # -- Build all sliding windows at once (vectorised) --------
                N_win   = T - seq_len + 1
                windows = np.stack([feats[i : i + seq_len] for i in range(N_win)])
                x_all   = torch.tensor(
                    windows, dtype=torch.float32, device=self.device
                )

                # -- Chunked inference -> take last timestep of each window
                preds_pen_n = np.full(T, np.nan)
                for start in range(0, N_win, 512):
                    chunk   = x_all[start : start + 512]
                    out_pen = self.model(chunk)[:, -1, 0].cpu().numpy()  # pen only
                    s = seq_len - 1 + start
                    preds_pen_n[s : s + len(out_pen)] = out_pen

                # -- Inverse-transform Penicillin to g/L -------------------
                valid_mask    = ~np.isnan(preds_pen_n)
                pred_pen_full = np.full(T, np.nan)
                if valid_mask.any():
                    pred_pen_full[valid_mask] = self.scalers.pen.inverse_transform(
                        preds_pen_n[valid_mask].reshape(-1, 1)
                    ).flatten()

                y_preds_pen.append(pred_pen_full)
                y_sparses_pen.append(sp_pen.copy())

                # -- Sparse-point metrics for Penicillin -------------------
                mask_pen = ~np.isnan(sp_pen) & ~np.isnan(pred_pen_full)
                if mask_pen.sum() > 0:
                    pen_true.append(sp_pen[mask_pen])
                    pen_pred_list.append(pred_pen_full[mask_pen])

                # -- Biomass placeholder (not predicted) -------------------
                y_preds_bio.append(np.full(T, np.nan))
                y_sparses_bio.append(sp_bio.copy())

        # -- Aggregate metrics --------------------------------------------
        if pen_true:
            pen_t_cat = np.concatenate(pen_true)
            pen_p_cat = np.concatenate(pen_pred_list)
            pen_metrics = compute_metrics(pen_t_cat, pen_p_cat)
        else:
            pen_metrics = {"RMSE": float("nan"), "MAE": float("nan"),
                           "R2": float("nan"), "n": 0}

        bio_metrics = {"RMSE": float("nan"), "MAE": float("nan"),
                       "R2": float("nan"), "n": 0}

        # -- Print for interactive use ------------------------------------
        print(
            f"  [Penicillin] "
            f"RMSE={pen_metrics.get('RMSE', float('nan')):.3f} g/L  "
            f"MAE={pen_metrics.get('MAE', float('nan')):.3f} g/L  "
            f"R2={pen_metrics.get('R2', float('nan')):.4f}  "
            f"n={pen_metrics.get('n', 0)}"
        )
        print("  [Biomass   ] N/A (PCA-PI-LSTM predicts Penicillin only)")

        return {
            "pen":           pen_metrics,
            "bio":           bio_metrics,
            "y_preds_pen":   y_preds_pen,
            "y_sparses_pen": y_sparses_pen,
            "y_preds_bio":   y_preds_bio,
            "y_sparses_bio": y_sparses_bio,
        }

    # -- Overrides ------------------------------------------------------------

    def _make_optimizer(self) -> torch.optim.Optimizer:
        """
        Two AdamW parameter groups:
          network  : lr=1e-3, weight_decay=1e-4
          physics  : lr=1e-2, weight_decay=0  (k_prod only)

        The higher learning rate for k_prod mirrors the PINN convention where
        physics parameters are updated more aggressively than neural weights.
        """
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
        """
        Freeze early-stopping patience during the physics delay phase.

        Avoids premature stopping before the physics loss has had any effect.
        In vanilla_mode (lambda_physics always 0) ES runs from epoch 1.
        """
        tr      = self.cfg.training
        vanilla = bool(getattr(tr, "vanilla_mode", False))
        if vanilla:
            return True
        delay = int(getattr(tr, "lambda_anneal_delay", 0))
        return epoch > delay

    def _epoch_extra_kwargs(self, epoch: int) -> Dict[str, Any]:
        """
        Compute and return the annealed lambda_physics for this epoch.

        Schedule:
          epoch <= delay                   -> lambda = lambda_start (usually 0)
          delay < epoch <= delay + anneal  -> linear ramp to lambda_end
          epoch > delay + anneal           -> lambda = lambda_end (constant)
        """
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


# -- Register ---------------------------------------------------------------
REGISTRY[PCAPILSTMAlgorithm.name] = PCAPILSTMAlgorithm
