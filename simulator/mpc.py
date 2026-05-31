"""
simulator/mpc.py
================
BioreactorMPC  — gradient-based MPC for PI-LSTM and Neural ODE models.
MPCBatchController — drop-in IndPenSim control callback that overlays MPC
                     actions on top of recipe + PID baseline.
"""

from __future__ import annotations

import gc
from collections import deque
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.data.dataset import PROCESS_FEATURE_COLS

# ── MPC-controllable variables ────────────────────────────────────────────────
#
# Physical bounds taken from dataset statistics.  The simulator keys map
# to the override dict returned by MPCBatchController.

CTRL_SETTINGS: List[Dict] = [
    # name (PROCESS_FEATURE_COLS entry), sim_key, col_idx, physical min/max
    # Bounds taken from training-set scaler data_min_ / data_max_, ensuring the
    # MPC never drives controls outside the distribution the model was trained on.
    {"name": "Aeration rate(Fg:L/h)",          "sim_key": "Fg",   "idx": 1,  "min": 30.0, "max": 75.0},
    {"name": "Sugar feed rate(Fs:L/h)",         "sim_key": "Fs",   "idx": 2,  "min": 8.0,  "max": 150.0},
    {"name": "PAA flow(Fpaa:PAA flow (L/h))",   "sim_key": "Fpaa", "idx": 18, "min": 0.0,  "max": 15.0},
    {"name": "Oil flow(Foil:L/hr)",             "sim_key": "Foil", "idx": 19, "min": 22.0, "max": 35.0},
]


# ── BioreactorMPC ─────────────────────────────────────────────────────────────

class BioreactorMPC:
    """
    Gradient-based MPC that optimises the next ``horizon`` control actions
    to maximise a weighted combination of predicted penicillin and biomass.

    Adapted from ReaKt/MPC.py for our PI-LSTM / Neural ODE algorithms which:
      - Use MinMaxScaler (not StandardScaler)
      - Do NOT append output columns to the input feature vector
      - Require different rolling-forward logic per model type

    Parameters
    ----------
    algorithm  : loaded BaseAlgorithm (PILSTMAlgorithm or NeuralODEAlgorithm)
    horizon    : prediction horizon in steps (default 5)
    steps      : Adam optimisation iterations per MPC call (default 10)
    bio_weight : weight for biomass in the reward (0.0 = penicillin only,
                 1.0 = biomass only, 0.5 = equal weight).  Default 0.5.
    """

    def __init__(self, algorithm, horizon: int = 5, steps: int = 10, bio_weight: float = 0.5):
        self.algorithm   = algorithm
        self.model       = algorithm.model
        self.scalers     = algorithm.scalers
        self.horizon     = horizon
        self.steps       = steps
        self.bio_weight  = float(np.clip(bio_weight, 0.0, 1.0))
        self.model_type  = algorithm.name   # "pi_lstm", "neural_ode", etc.
        self.dt          = float(algorithm.cfg.training.dt)

        try:
            self.device = next(self.model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")

        feat_scaler = self.scalers.feat

        # Column indices of the 4 MPC-controlled variables inside the full
        # normalised feature vector (may be 23-dim or 87-dim for Raman models)
        self.ctrl_indices  = [c["idx"] for c in CTRL_SETTINGS]
        self.ctrl_sim_keys = [c["sim_key"] for c in CTRL_SETTINGS]

        # Convert physical bounds → normalised space via MinMaxScaler
        data_min = feat_scaler.data_min_[self.ctrl_indices]
        scale    = feat_scaler.scale_[self.ctrl_indices]   # 1/(max-min)

        phys_min = np.array([c["min"] for c in CTRL_SETTINGS], dtype=np.float32)
        phys_max = np.array([c["max"] for c in CTRL_SETTINGS], dtype=np.float32)

        norm_min = (phys_min - data_min) * scale
        norm_max = (phys_max - data_min) * scale

        self.min_t = torch.tensor(norm_min, dtype=torch.float32, device=self.device)
        self.max_t = torch.tensor(norm_max, dtype=torch.float32, device=self.device)

    # ── Public API ────────────────────────────────────────────────────────────

    def optimize(
        self,
        current_seq_np: np.ndarray,
        y0_np: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Run gradient-based MPC optimisation.

        Parameters
        ----------
        current_seq_np : (seq_len, n_feat) float32 — normalised feature window
        y0_np          : (2,) float32 — normalised [X, P] initial state,
                         required for Neural ODE; ignored for PI-LSTM.

        Returns
        -------
        actions_phys : (4,) float32 — physical-space first-step actions
                       ordered as [Fg, Fs, Fpaa, Foil]
        """
        was_training = self.model.training
        self.model.train()
        # Keep dropout in eval mode for stability during gradient rollout
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.eval()

        u_future   = None
        optimizer  = None
        loss       = None

        try:
            seq = torch.tensor(
                current_seq_np, dtype=torch.float32, device=self.device
            ).unsqueeze(0)  # (1, seq_len, n_feat)

            u_future = torch.zeros(
                self.horizon, len(self.ctrl_indices),
                device=self.device, requires_grad=True,
            )
            optimizer = optim.Adam([u_future], lr=0.1)

            # Prepare y0 for Neural ODE
            if "neural_ode" in self.model_type and y0_np is not None:
                y0 = torch.tensor(
                    y0_np[None, :], dtype=torch.float32, device=self.device
                )   # (1, 2)
            else:
                y0 = None

            for _ in range(self.steps):
                optimizer.zero_grad()

                if "pi_lstm" in self.model_type:
                    rewards = self._rollout_pi_lstm(seq, u_future)
                else:
                    rewards = self._rollout_neural_ode(seq, u_future, y0)

                loss = (
                    -torch.mean(torch.stack(rewards))
                    + 0.1 * torch.sum((u_future[1:] - u_future[:-1]) ** 2)
                )
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    for i in range(len(self.ctrl_indices)):
                        u_future[:, i].clamp_(self.min_t[i], self.max_t[i])

            # ── Convert first-step action from normalised → physical ──────────
            first_norm = u_future.detach().cpu().numpy()[0, :]   # (4,)

            feat_scaler = self.scalers.feat
            data_min    = feat_scaler.data_min_[self.ctrl_indices]
            scale       = feat_scaler.scale_[self.ctrl_indices]

            first_phys = first_norm / scale + data_min
            return first_phys.astype(np.float32)

        finally:
            self.model.train(was_training)
            del u_future, optimizer, loss
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    # ── Rollout helpers ───────────────────────────────────────────────────────

    def _rollout_pi_lstm(
        self,
        seq: torch.Tensor,
        u_future: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Roll out PI-LSTM for ``horizon`` steps.

        At each step:
          1. Forward pass → last-step prediction `pred[:, -1, :]`  (shape 1×2)
          2. Penicillin (index 1) is collected as reward
          3. Window shifted: last row updated with u_future[t], then shift left
        """
        curr    = seq.clone()
        rewards = []

        for t in range(self.horizon):
            pred = self.model(curr)[:, -1, :]   # (1, 2) normalised [X, P]
            # Weighted reward: bio_weight * X + (1 - bio_weight) * P
            reward = self.bio_weight * pred[0, 0] + (1.0 - self.bio_weight) * pred[0, 1]
            rewards.append(reward)

            # Advance rolling window
            last_in = curr[0, -1, :].clone()
            for i, c_idx in enumerate(self.ctrl_indices):
                last_in[c_idx] = u_future[t, i]
            curr = torch.cat(
                (curr[:, 1:, :], last_in.view(1, 1, -1)), dim=1
            )

        return rewards

    def _rollout_neural_ode(
        self,
        seq: torch.Tensor,
        u_future: torch.Tensor,
        y0: Optional[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Roll out Neural ODE for ``horizon`` steps using RK4.

        At each step:
          1. Build a 2-point U_grid from the last seq row with overridden ctrl cols
          2. Integrate one step (t=0 → t=dt)
          3. Predicted state becomes y0 for the next step
          4. Penicillin (index 1) is collected as reward
        """
        if y0 is None:
            # Fallback: start from a zero state (should not occur in normal use)
            y0 = torch.zeros(1, 2, device=self.device)

        t_step = torch.tensor(
            [0.0, self.dt], dtype=torch.float32, device=self.device
        )

        rewards = []
        curr_y0 = y0.clone()

        for t in range(self.horizon):
            # Build control row for this step (last seq row + ctrl override)
            u_row = seq[:, -1:, :].clone()   # (1, 1, n_feat)
            for i, c_idx in enumerate(self.ctrl_indices):
                u_row[0, 0, c_idx] = u_future[t, i]

            # Replicate to 2 time points (piecewise-linear interp in model)
            U_grid = u_row.expand(-1, 2, -1)   # (1, 2, n_feat)

            traj   = self.model(curr_y0, t_step, U_grid, method="rk4")
            curr_y0 = traj[:, -1, :]            # (1, 2) — state at t=dt

            # Weighted reward: bio_weight * X + (1 - bio_weight) * P
            reward = self.bio_weight * curr_y0[0, 0] + (1.0 - self.bio_weight) * curr_y0[0, 1]
            rewards.append(reward)

        return rewards


# ── MPCBatchController ────────────────────────────────────────────────────────

class MPCBatchController:
    """
    Drop-in control callback for IndPenSim's ``f_input`` argument.

    Wraps ``fctrl_indpensim`` (recipe + PIDs + faults) and overrides the 4
    MPC-controlled variables (Fg, Fs, Fpaa, Foil) when the rolling window has
    accumulated enough history for the MPC to act.

    Parameters
    ----------
    algorithm     : loaded BaseAlgorithm
    mpc_optimizer : BioreactorMPC instance
    knn_raman     : KNNRamanSelector (optional; used with Raman models)
    seq_len       : history window length (must match model's training seq_len)
    """

    def __init__(
        self,
        algorithm,
        mpc_optimizer: BioreactorMPC,
        knn_raman=None,
        seq_len: int = 24,
    ):
        self.algorithm     = algorithm
        self.mpc           = mpc_optimizer
        self.knn_raman     = knn_raman
        self.seq_len       = seq_len
        self._feat_cols    = algorithm._feature_cols()
        self._scalers      = algorithm.scalers
        self._window: deque = deque(maxlen=seq_len)
        self._cached_action: Optional[np.ndarray] = None  # physical-space (4,)

        # Import here to avoid circular imports at module load time
        from .fctrl_indpensim import fctrl_indpensim as _base_ctrl
        self._base_ctrl = _base_ctrl

        # Import SIM_TO_FEAT for feature extraction
        from .variable_map import SIM_TO_FEAT
        self._sim_to_feat = SIM_TO_FEAT

    def __call__(
        self,
        X: dict,
        Xd: dict,
        k: int,
        h: float,
        T: float,
        Ctrl_flags: dict,
    ):
        """
        Called at each simulation step k.

        1. Extract normalised feature vector for the current state
        2. Append to rolling window
        3. If window full: run MPC → cache first-step action
        4. Call base recipe controller
        5. Override MPC-controlled variables with cached action (if available)
        """
        # ── 1. Build process feature vector ──────────────────────────────────
        feat_dict = self._extract_feature_dict(X, k, h)

        if self.knn_raman is not None:
            raman_latent = self.knn_raman.query(feat_dict)   # (64,)
            # Build row in the order expected by Raman model (23 + 64 = 87)
            from src.data.dataset import RAMAN_LATENT_COLS
            feat_row = np.array(
                [feat_dict[c] for c in PROCESS_FEATURE_COLS]
                + list(raman_latent),
                dtype=np.float32,
            )
        else:
            feat_row = np.array(
                [feat_dict[c] for c in PROCESS_FEATURE_COLS], dtype=np.float32
            )

        # ── 2. Normalise and push to window ──────────────────────────────────
        feat_row_norm = self._scalers.feat.transform(feat_row[None, :])[0].astype(np.float32)
        self._window.append(feat_row_norm)

        # ── 3. Run MPC if window is full ──────────────────────────────────────
        if len(self._window) == self.seq_len:
            seq_np = np.stack(list(self._window), axis=0)   # (seq_len, n_feat)

            # y0 for Neural ODE: normalised [X, P] from last known state
            y0_np = None
            if "neural_ode" in self.algorithm.name:
                prev = max(k - 1, 0)
                x_phys = float(X["X"]["y"][prev])
                p_phys = float(X["P"]["y"][prev])
                x_norm = float(self._scalers.bio.transform([[x_phys]])[0, 0])
                p_norm = float(self._scalers.pen.transform([[p_phys]])[0, 0])
                y0_np  = np.array([x_norm, p_norm], dtype=np.float32)

            self._cached_action = self.mpc.optimize(seq_np, y0_np)

        # ── 4. Call base recipe + PID controller ─────────────────────────────
        u, X = self._base_ctrl(X, Xd, k, h, T, Ctrl_flags)

        # ── 5. Override MPC-controlled variables ──────────────────────────────
        if self._cached_action is not None:
            for i, c in enumerate(CTRL_SETTINGS):
                u[c["sim_key"]] = float(self._cached_action[i])

        return u, X

    # ── Feature extraction ────────────────────────────────────────────────────

    def _extract_feature_dict(self, X: dict, k: int, h: float) -> dict:
        """
        Build a {col_name: value} dict for the 23 PROCESS_FEATURE_COLS at
        step k, using the previous step's states (available to the controller).
        """
        prev = max(k - 1, 0)
        feat = {}

        # Time
        feat["Time (h)"] = float(k * h)

        # 22 process variables via SIM_TO_FEAT
        for sim_key, col_name in self._sim_to_feat.items():
            arr = X.get(sim_key, {}).get("y", None)
            if arr is not None and len(arr) > prev:
                feat[col_name] = float(arr[prev])
            else:
                feat[col_name] = 0.0

        return feat
