"""
src/models/neural_ode.py
========================
Neural ODE model for dual-output prediction — pure nn.Module definitions.

No data loading, training, or evaluation logic here.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchdiffeq import odeint


class ODEFunc(nn.Module):
    """
    The right-hand side  f_θ  of the Neural ODE:

        d[X, P] / dt = f_θ([X(t), P(t)], U(t))

    where U(t) is the time-varying control input (process variables ± Raman
    latents) interpolated piecewise-linearly from the discrete grid.

    Parameters
    ----------
    n_state : dimension of state vector (default 2: [X, P])
    n_ctrl  : dimension of control vector (default 23 for process-only)
    hidden  : MLP hidden width (default 64)
    """

    def __init__(self, n_state: int = 2, n_ctrl: int = 23, hidden: int = 64):
        super().__init__()
        self.n_state = n_state
        self.net = nn.Sequential(
            nn.Linear(n_state + n_ctrl, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_state),
            # No final activation — derivative can be negative
        )
        self._t_grid: torch.Tensor = None   # (N,)  time points
        self._U_grid: torch.Tensor = None   # (B, N, n_ctrl) control trajectories

        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def set_controls(self, t_grid: torch.Tensor, U_grid: torch.Tensor):
        """Cache control trajectory before calling odeint."""
        self._t_grid = t_grid    # (N,)
        self._U_grid = U_grid    # (B, N, n_ctrl)

    def _interpolate_U(self, t: torch.Tensor) -> torch.Tensor:
        """
        Piecewise-linear interpolation of U at scalar time t.

        Returns
        -------
        U_t : (B, n_ctrl)
        """
        idx  = torch.searchsorted(self._t_grid, t.unsqueeze(0)).squeeze(0)
        idx  = idx.clamp(1, len(self._t_grid) - 1)
        t0   = self._t_grid[idx - 1]
        t1   = self._t_grid[idx]
        w    = ((t - t0) / (t1 - t0 + 1e-8)).clamp(0.0, 1.0)
        U0   = self._U_grid[:, idx - 1, :]    # (B, n_ctrl)
        U1   = self._U_grid[:, idx,     :]    # (B, n_ctrl)
        return U0 + w * (U1 - U0)             # (B, n_ctrl)

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        t : scalar time
        y : (B, n_state)

        Returns
        -------
        dydt : (B, n_state)
        """
        U_t = self._interpolate_U(t)              # (B, n_ctrl)
        inp = torch.cat([y, U_t], dim=-1)         # (B, n_state + n_ctrl)
        return self.net(inp)


class NeuralODEModel(nn.Module):
    """
    Neural ODE integrator for penicillin fermentation state prediction.

    Usage
    -----
        model = NeuralODEModel(n_state=2, n_ctrl=23, hidden=64)
        traj  = model(y0, t_span, U_grid, method="rk4")
        # traj : (B, T, n_state)
    """

    def __init__(self, n_state: int = 2, n_ctrl: int = 23, hidden: int = 64):
        super().__init__()
        self.odefunc = ODEFunc(n_state=n_state, n_ctrl=n_ctrl, hidden=hidden)

    def forward(
        self,
        y0:     torch.Tensor,   # (B, n_state)
        t_span: torch.Tensor,   # (T,)  time points
        U_grid: torch.Tensor,   # (B, T, n_ctrl)
        method: str = "rk4",
    ) -> torch.Tensor:
        """
        Integrate the ODE from y0 over t_span.

        Returns
        -------
        traj : (B, T, n_state)
        """
        self.odefunc.set_controls(t_span, U_grid)
        traj = odeint(self.odefunc, y0, t_span, method=method)
        # odeint output shape: (T, B, n_state) → permute to (B, T, n_state)
        return traj.permute(1, 0, 2)
