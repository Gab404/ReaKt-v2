"""
simulator/variable_map.py
=========================
Mapping from IndPenSim simulator output dict keys to the exact column names
expected by our model's PROCESS_FEATURE_COLS, plus a helper that assembles
a pandas DataFrame from a raw simulator output dict.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from src.data.dataset import PROCESS_FEATURE_COLS, TARGET_COLS, FAULT_COL

# ── Simulator key → DataFrame column name ────────────────────────────────────
#
# Maps the 22 process-variable keys present in the IndPenSim output dict to the
# exact column names used in PROCESS_FEATURE_COLS (excluding "Time (h)" which is
# handled separately from X['Fg']['t']).

SIM_TO_FEAT: Dict[str, str] = {
    "Fg":        "Aeration rate(Fg:L/h)",
    "Fs":        "Sugar feed rate(Fs:L/h)",
    "Fa":        "Acid flow rate(Fa:L/h)",
    "Fb":        "Base flow rate(Fb:L/h)",
    "Fc":        "Heating/cooling water flow rate(Fc:L/h)",
    "Fh":        "Heating water flow rate(Fh:L/h)",
    "Fw":        "Water for injection/dilution(Fw:L/h)",
    "pressure":  "Air head pressure(pressure:bar)",
    "Fremoved":  "Dumped broth flow(Fremoved:L/h)",
    "S":         "Substrate concentration(S:g/L)",
    "DO2":       "Dissolved oxygen concentration(DO2:mg/L)",
    "V":         "Vessel Volume(V:L)",
    "Wt":        "Vessel Weight(Wt:Kg)",
    "pH":        "pH(pH:pH)",
    "T":         "Temperature(T:K)",
    "Q":         "Generated heat(Q:kJ)",
    "CO2outgas": "carbon dioxide percent in off-gas(CO2outgas:%)",
    "Fpaa":      "PAA flow(Fpaa:PAA flow (L/h))",
    "Foil":      "Oil flow(Foil:L/hr)",
    "OUR":       "Oxygen Uptake Rate(OUR:(g min^{-1}))",
    "O2":        "Oxygen in percent in off-gas(O2:O2  (%))",
    "CER":       "Carbon evolution rate(CER:g/h)",
}


def sim_to_dataframe(X: dict, fault_code: int = 0) -> pd.DataFrame:
    """
    Convert a raw IndPenSim simulation output dict to a pandas DataFrame with
    the same column layout used by PenicillinDataModule.

    Parameters
    ----------
    X          : dict returned by indpensim() / indpensim_run()
    fault_code : fault reference integer (0 = no fault); used to fill the
                 Fault_ref column when the simulator didn't inject faults.

    Returns
    -------
    df : pd.DataFrame with columns:
         - "Time (h)"
         - all 22 PROCESS_FEATURE_COLS (excluding Time)
         - FAULT_COL
         - "_biomass_sparse"    (NaN except offline-measurement rows)
         - "_penicillin_sparse" (NaN except offline-measurement rows)
         - TARGET_COLS[0]  (dense biomass  — linearly interpolated + ffill/bfill)
         - TARGET_COLS[1]  (dense penicillin — linearly interpolated + ffill/bfill)
    """

    # ── Time vector (from Fg channel which is always present) ────────────────
    t_vec = X["Fg"]["t"]
    n = len(t_vec)

    data: Dict[str, np.ndarray] = {"Time (h)": t_vec.copy()}

    # ── 22 process variables ─────────────────────────────────────────────────
    for sim_key, col_name in SIM_TO_FEAT.items():
        arr = X[sim_key]["y"]
        # Trim / pad to match n (defensive; should always be equal)
        if len(arr) > n:
            arr = arr[:n]
        elif len(arr) < n:
            arr = np.concatenate([arr, np.full(n - len(arr), arr[-1])])
        data[col_name] = arr.copy()

    # ── Fault reference ───────────────────────────────────────────────────────
    if "Fault_ref" in X:
        fault_arr = X["Fault_ref"]["y"][:n].copy()
    else:
        fault_arr = np.full(n, float(fault_code))
    data[FAULT_COL] = fault_arr

    # ── Sparse offline measurements ──────────────────────────────────────────
    #   IndPenSim stores NaN at non-sampling timesteps and the delayed
    #   measurement value at sampling times (every 12 h with 4 h delay).
    sparse_bio = X["X_offline"]["y"][:n].copy()
    sparse_pen = X["P_offline"]["y"][:n].copy()

    data["_biomass_sparse"]    = sparse_bio
    data["_penicillin_sparse"] = sparse_pen

    # ── Dense target columns (interpolated sparse → dense) ───────────────────
    #   Matches PenicillinDataModule's per-batch interpolation logic:
    #     b[tc] = b[tc].interpolate(method="linear").ffill().bfill()
    df = pd.DataFrame(data)

    for tc, sparse_col in [
        (TARGET_COLS[0], "_biomass_sparse"),
        (TARGET_COLS[1], "_penicillin_sparse"),
    ]:
        df[tc] = df[sparse_col].interpolate(method="linear").ffill().bfill()

    # ── Reorder to match PROCESS_FEATURE_COLS + extras ───────────────────────
    ordered = (
        PROCESS_FEATURE_COLS
        + [FAULT_COL, "_biomass_sparse", "_penicillin_sparse"]
        + TARGET_COLS
    )
    # Keep only columns we have, in the right order
    present = [c for c in ordered if c in df.columns]
    df = df[present].reset_index(drop=True)

    return df
