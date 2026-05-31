#!/usr/bin/env python
"""
simulator.py  — IndPenSim + Model Evaluation CLI
=================================================
Runs any of the 4 trained models on newly-simulated bioreactor batches.

Examples
--------
# Basic prediction evaluation (recipe control)
python simulator.py --model pi_lstm --n-batches 2 --fault 0 1

# With MPC control
python simulator.py --model pi_lstm --mpc --n-batches 2

# With KNN Raman retrieval (required for Raman model variants)
python simulator.py --model pi_lstm_raman --knn-raman --n-batches 1

# MPC + KNN Raman
python simulator.py --model neural_ode_raman --mpc --knn-raman --n-batches 2

# All options
python simulator.py --model pi_lstm --n-batches 3 --fault 0 1 0 \\
    --mpc --mpc-horizon 5 --mpc-steps 10 \\
    --seed 42 --device cuda \\
    --plots --show --save ./outputs/sim_run/
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

# ── Default checkpoint paths ──────────────────────────────────────────────────

DEFAULT_CKPTS = {
    "pi_lstm":         "checkpoints/pi_lstm.pt",
    "pi_lstm_raman":   "checkpoints/pi_lstm_raman.pt",
    "neural_ode":      "checkpoints/neural_ode.pt",
    "neural_ode_raman":"checkpoints/neural_ode_raman.pt",
}

RAMAN_ENCODER_CKPT = "checkpoints/cdae_best_model.pth"


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="IndPenSim bioreactor simulator + model evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model
    p.add_argument(
        "--model", required=True,
        choices=["pi_lstm", "neural_ode", "pi_lstm_raman", "neural_ode_raman"],
        help="Which model to evaluate",
    )
    p.add_argument(
        "--ckpt", default=None,
        help="Override checkpoint path (default: from DEFAULT_CKPTS)",
    )

    # Simulation
    p.add_argument("--n-batches", type=int, default=1, help="Number of batches to simulate")
    p.add_argument(
        "--fault", type=int, nargs="*", default=[0],
        help="Per-batch fault codes 0-8 (repeats last if fewer than n-batches)",
    )
    p.add_argument("--seed", type=int, default=42, help="Global RNG seed")
    p.add_argument("--device", default="auto", help="'cpu', 'cuda', or 'auto'")

    # MPC
    p.add_argument("--mpc",           action="store_true", help="Enable MPC control")
    p.add_argument("--mpc-horizon",   type=int,   default=5,   help="MPC prediction horizon (steps)")
    p.add_argument("--mpc-steps",     type=int,   default=10,  help="Adam iterations per MPC call")
    p.add_argument("--mpc-bio-weight",type=float, default=0.5,
                   help="Reward weight for biomass [0=pen only, 1=bio only, 0.5=equal]. Default 0.5")

    # KNN Raman
    p.add_argument("--knn-raman",  action="store_true", help="Enable KNN Raman latent retrieval")
    p.add_argument("--knn-k",      type=int,   default=5,                         help="Number of KNN neighbours")
    p.add_argument("--knn-cache",  default="./outputs/raman_knn_cache.npz",       help="KNN cache path")

    # Output
    p.add_argument("--plots",  action="store_true", help="Save PNG plots to --save dir")
    p.add_argument("--show",   action="store_true", help="Display interactive matplotlib window")
    p.add_argument("--save",   default="./outputs/simulator/", help="Output directory")

    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def pad_fault_list(fault_list: List[int], n: int) -> List[int]:
    """Extend fault list to length n by repeating the last entry."""
    if not fault_list:
        fault_list = [0]
    while len(fault_list) < n:
        fault_list.append(fault_list[-1])
    return fault_list[:n]


def build_batch_run_flags(n_batches: int, fault_list: List[int]) -> dict:
    """Construct the Batch_run_flags dict expected by indpensim_run."""
    return {
        "Control_strategy":             np.zeros(n_batches, dtype=int),   # SBC=0 (recipe)
        "Batch_length":                 np.zeros(n_batches, dtype=int),   # 0=fixed 230h
        "Batch_fault_order_reference":  np.array(fault_list, dtype=int),
        "Raman_spec":                   np.zeros(n_batches, dtype=int),   # 0=no Raman sensor
    }


def load_algorithm(model_name: str, ckpt_path: str, device: torch.device):
    """Load a trained algorithm from checkpoint."""
    # Ensure all algorithm subclasses are registered
    import src.algorithms  # noqa: F401

    from src.algorithms.base import BaseAlgorithm
    print(f"  Loading checkpoint: {ckpt_path}")
    alg = BaseAlgorithm.load(ckpt_path, device)
    print(f"  Algorithm: {alg.name}  |  device: {device}")
    return alg


def load_raman_encoder(device: torch.device):
    """Load the frozen CDAE Raman encoder."""
    from src.data.raman_encoder import RamanEncoder
    enc = RamanEncoder(RAMAN_ENCODER_CKPT, device)
    return enc


# ── Per-batch simulation + evaluation ────────────────────────────────────────

def run_batch_recipe(
    batch_no: int,
    fault_code: int,
    alg,
    knn_raman,
    n_batches: int,
    fault_list: List[int],
) -> dict:
    """
    Run one batch with recipe control (no MPC).
    Returns a results dict.
    """
    from simulator.indpensim_run import indpensim_run
    from simulator.variable_map import sim_to_dataframe
    from src.data.dataset import RAMAN_LATENT_COLS

    flags = build_batch_run_flags(n_batches, fault_list)
    print(f"\n  Simulating batch {batch_no} (fault={fault_code}) ...")
    t0   = time.time()
    Xref = indpensim_run(batch_no, flags)
    sim_time = time.time() - t0
    print(f"  Simulation done in {sim_time:.1f}s")

    # ── Build DataFrame ───────────────────────────────────────────────────────
    df = sim_to_dataframe(Xref, fault_code=fault_code)

    # If Raman model: append KNN-retrieved latents row-by-row
    if knn_raman is not None:
        from src.data.dataset import PROCESS_FEATURE_COLS
        latents = np.zeros((len(df), 64), dtype=np.float32)
        for row_i, row in df[PROCESS_FEATURE_COLS].iterrows():
            latents[row_i] = knn_raman.query(row.to_dict())
        for j, col in enumerate(RAMAN_LATENT_COLS):
            df[col] = latents[:, j]

        # Forward-fill Raman latents (first rows may be zero)
        for col in RAMAN_LATENT_COLS:
            df[col] = df[col].replace(0.0, float("nan")).ffill().bfill()

    # ── Evaluate ──────────────────────────────────────────────────────────────
    metrics = alg.evaluate([df])

    return {
        "Xref":    Xref,
        "df":      df,
        "metrics": metrics,
        "batch_no": batch_no,
        "fault":   fault_code,
    }


def run_batch_mpc(
    batch_no: int,
    fault_code: int,
    alg,
    mpc_ctrl: "MPCBatchController",
    knn_raman,
    n_batches: int,
    fault_list: List[int],
) -> dict:
    """
    Run one batch with MPC control.
    Also runs a recipe baseline for comparison.
    Returns a results dict.
    """
    from simulator.indpensim import indpensim
    from simulator.indpensim_run import indpensim_run
    from simulator.variable_map import sim_to_dataframe
    from simulator.parameter_list import parameter_list
    from scipy.signal import lfilter
    from src.data.dataset import RAMAN_LATENT_COLS, PROCESS_FEATURE_COLS

    # ── Recipe baseline ───────────────────────────────────────────────────────
    flags = build_batch_run_flags(n_batches, fault_list)
    print(f"\n  Simulating batch {batch_no} recipe baseline (fault={fault_code}) ...")
    Xref_recipe = indpensim_run(batch_no, flags)

    # ── MPC-driven run ────────────────────────────────────────────────────────
    # Re-initialize the MPC controller's rolling window for this batch
    from collections import deque
    mpc_ctrl._window       = deque(maxlen=mpc_ctrl.seq_len)
    mpc_ctrl._cached_action = None

    # We need to re-generate initial conditions with the same seed as batch_no.
    # The simplest approach: replicate the init logic from indpensim_run.
    print(f"  Simulating batch {batch_no} with MPC control ...")

    # Use the same seeding logic as indpensim_run
    np.random.seed()  # allow fresh randomisation (same as recipe baseline seed)
    Random_seed_ref = int(np.ceil(np.random.rand() * 1000))
    Seed_ref = 31 + Random_seed_ref
    Rand_ref = 1

    x0 = {}
    np.random.seed(Seed_ref + batch_no + Rand_ref); Rand_ref += 1
    intial_conds = 0.5 + 0.05 * np.random.randn()
    np.random.seed(Seed_ref + batch_no + Rand_ref); Rand_ref += 1
    x0['mux'] = 0.41 + 0.025 * np.random.randn()
    np.random.seed(Seed_ref + batch_no + Rand_ref); Rand_ref += 1
    x0['mup'] = 0.041 + 0.0025 * np.random.randn()
    h = 0.2
    T = 230.0
    np.random.seed(Seed_ref + batch_no + Rand_ref); Rand_ref += 1
    x0['S']   = 1 + 0.1 * np.random.randn()
    np.random.seed(Seed_ref + batch_no + Rand_ref); Rand_ref += 1
    x0['DO2'] = 15 + 0.5 * np.random.randn()
    np.random.seed(Seed_ref + batch_no + Rand_ref); Rand_ref += 1
    x0['X']   = intial_conds + 0.1 * np.random.randn()
    x0['P']   = 0
    np.random.seed(Seed_ref + batch_no + Rand_ref); Rand_ref += 1
    x0['V']   = 5.800e+04 + 500 * np.random.randn()
    np.random.seed(Seed_ref + batch_no + Rand_ref); Rand_ref += 1
    x0['Wt']  = 6.2e+04 + 500 * np.random.randn()
    np.random.seed(Seed_ref + batch_no + Rand_ref); Rand_ref += 1
    x0['CO2outgas'] = 0.038 + 0.001 * np.random.randn()
    np.random.seed(Seed_ref + batch_no + Rand_ref); Rand_ref += 1
    x0['O2']  = 0.20 + 0.05 * np.random.randn()
    np.random.seed(Seed_ref + batch_no + Rand_ref); Rand_ref += 1
    x0['pH']  = 6.5 + 0.1 * np.random.randn()
    np.random.seed(Seed_ref + batch_no + Rand_ref); Rand_ref += 1
    x0['T']   = 297 + 0.5 * np.random.randn()
    x0['a0']  = intial_conds * (1/3)
    x0['a1']  = intial_conds * (2/3)
    x0['a3']  = 0; x0['a4'] = 0; x0['Culture_age'] = 0
    np.random.seed(Seed_ref + batch_no + Rand_ref); Rand_ref += 1
    x0['PAA'] = 1400 + 50 * np.random.randn()
    np.random.seed(Seed_ref + batch_no + Rand_ref); Rand_ref += 1
    x0['NH3'] = 1700 + 50 * np.random.randn()
    np.random.seed(Seed_ref + batch_no + Rand_ref); Rand_ref += 1
    alpha_kla = 85 + 10 * np.random.randn()
    np.random.seed(Seed_ref + batch_no + Rand_ref); Rand_ref += 1
    PAA_c     = 530000 + 20000 * np.random.randn()
    np.random.seed(Seed_ref + batch_no + Rand_ref)
    N_conc_paa = 2 * 75000 + 2000 * np.random.randn()

    Batch_time = np.arange(0, T + h, h)
    if Batch_time[-1] > T:
        Batch_time = Batch_time[:-1]

    # Disturbances
    np.random.seed(Random_seed_ref + batch_no)
    b1 = np.array([1 - 0.995]); a1_f = np.array([1, -0.995])
    ns = len(Batch_time)

    def gd(sf):
        return lfilter(b1, a1_f, sf * np.random.randn(ns))

    Xinterp = {
        "distMuP":   {"y": gd(0.03),      "t": Batch_time},
        "distMuX":   {"y": gd(0.25),      "t": Batch_time},
        "distcs":    {"y": gd(1500),       "t": Batch_time},
        "distcoil":  {"y": gd(300),        "t": Batch_time},
        "distabc":   {"y": gd(0.2),        "t": Batch_time},
        "distPAA":   {"y": gd(300000),     "t": Batch_time},
        "distTcin":  {"y": gd(100),        "t": Batch_time},
        "distO_2in": {"y": gd(0.02),       "t": Batch_time},
    }

    par = parameter_list(x0, alpha_kla, N_conc_paa, PAA_c)

    Ctrl_flags = {
        "SBC": 0, "PRBS": 0, "Fixed_Batch_length": 0,
        "IC": 0, "Inhib": 2, "Dis": 1,
        "Faults": fault_code, "Vis": 0, "Raman_spec": 0,
        "Batch_Num": batch_no,
        "Off_line_m": 12, "Off_line_delay": 4, "plots": 0,
        "T_sp": 298, "pH_sp": 6.5,
    }

    t0 = time.time()
    Xref_mpc = indpensim(mpc_ctrl, Xinterp, x0, h, T, 2, par, Ctrl_flags)
    print(f"  MPC simulation done in {time.time()-t0:.1f}s")

    # ── Build DataFrames and evaluate ─────────────────────────────────────────
    df_mpc = sim_to_dataframe(Xref_mpc, fault_code=fault_code)

    if knn_raman is not None:
        latents = np.zeros((len(df_mpc), 64), dtype=np.float32)
        for row_i, row in df_mpc[PROCESS_FEATURE_COLS].iterrows():
            latents[row_i] = knn_raman.query(row.to_dict())
        from src.data.dataset import RAMAN_LATENT_COLS
        for j, col in enumerate(RAMAN_LATENT_COLS):
            df_mpc[col] = latents[:, j]
        for col in RAMAN_LATENT_COLS:
            df_mpc[col] = df_mpc[col].replace(0.0, float("nan")).ffill().bfill()

    metrics_mpc    = alg.evaluate([df_mpc])
    df_recipe      = sim_to_dataframe(Xref_recipe, fault_code=fault_code)

    if knn_raman is not None:
        latents_r = np.zeros((len(df_recipe), 64), dtype=np.float32)
        for row_i, row in df_recipe[PROCESS_FEATURE_COLS].iterrows():
            latents_r[row_i] = knn_raman.query(row.to_dict())
        from src.data.dataset import RAMAN_LATENT_COLS
        for j, col in enumerate(RAMAN_LATENT_COLS):
            df_recipe[col] = latents_r[:, j]
        for col in RAMAN_LATENT_COLS:
            df_recipe[col] = df_recipe[col].replace(0.0, float("nan")).ffill().bfill()

    metrics_recipe = alg.evaluate([df_recipe])

    return {
        "Xref_mpc":      Xref_mpc,
        "Xref_recipe":   Xref_recipe,
        "df_mpc":        df_mpc,
        "df_recipe":     df_recipe,
        "metrics_mpc":   metrics_mpc,
        "metrics_recipe": metrics_recipe,
        "batch_no":      batch_no,
        "fault":         fault_code,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_batch(result: dict, model_name: str, save_dir: Optional[Path], show: bool, use_mpc: bool):
    """Generate a 5-row figure for one simulated batch."""
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARNING] matplotlib not available; skipping plot.")
        return

    batch_no  = result["batch_no"]
    fault     = result["fault"]

    if use_mpc:
        df      = result["df_mpc"]
        df_base = result["df_recipe"]
        metrics = result["metrics_mpc"]
        preds_bio = metrics["y_preds_bio"][0]
        preds_pen = metrics["y_preds_pen"][0]
        preds_bio_base = result["metrics_recipe"]["y_preds_bio"][0]
        preds_pen_base = result["metrics_recipe"]["y_preds_pen"][0]
    else:
        df      = result["df"]
        df_base = None
        metrics = result["metrics"]
        preds_bio = metrics["y_preds_bio"][0]
        preds_pen = metrics["y_preds_pen"][0]

    t_vec = df["Time (h)"].values

    bio_rmse = metrics["bio"]["RMSE"]
    pen_rmse = metrics["pen"]["RMSE"]
    title    = (
        f"Batch {batch_no}  [{model_name}]  fault={fault}  "
        f"Bio RMSE={bio_rmse:.3f}  Pen RMSE={pen_rmse:.3f}"
    )

    fig, axes = plt.subplots(5, 1, figsize=(14, 20))
    fig.suptitle(title, fontsize=11, y=0.995)

    # Row 0: Biomass X
    ax = axes[0]
    sparse_bio = df["_biomass_sparse"].values
    ax.plot(t_vec, preds_bio, "b--", lw=1.5, label="Model pred.")
    if df_base is not None:
        t_base = df_base["Time (h)"].values if "Time (h)" in df_base.columns else t_vec
        ax.plot(t_base, preds_bio_base, "g--", lw=1.2, alpha=0.7, label="Recipe pred.")
    mask_b = ~np.isnan(sparse_bio)
    ax.scatter(t_vec[mask_b], sparse_bio[mask_b], c="k", s=15, zorder=5, label="Measurements")
    ax.set_ylabel("Biomass X (g/L)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Row 1: Penicillin P
    ax = axes[1]
    sparse_pen = df["_penicillin_sparse"].values
    ax.plot(t_vec, preds_pen, "r--", lw=1.5, label="Model pred.")
    if df_base is not None:
        ax.plot(t_base, preds_pen_base, "g--", lw=1.2, alpha=0.7, label="Recipe pred.")
    mask_p = ~np.isnan(sparse_pen)
    ax.scatter(t_vec[mask_p], sparse_pen[mask_p], c="k", s=15, zorder=5, label="Measurements")
    ax.set_ylabel("Penicillin P (g/L)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Row 2: Substrate S + DO2
    ax  = axes[2]
    ax2 = ax.twinx()
    ax.plot(t_vec,  df["Substrate concentration(S:g/L)"].values,            "b-",  lw=1.2, label="S")
    ax2.plot(t_vec, df["Dissolved oxygen concentration(DO2:mg/L)"].values,  "c-",  lw=1.2, label="DO2")
    ax.set_ylabel("S (g/L)", color="b"); ax2.set_ylabel("DO2 (mg/L)", color="c")
    ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # Row 3: pH + Temperature
    ax  = axes[3]
    ax2 = ax.twinx()
    ax.plot(t_vec,  df["pH(pH:pH)"].values,             "m-",  lw=1.2, label="pH")
    ax2.plot(t_vec, df["Temperature(T:K)"].values,      "orange", lw=1.2, label="T (K)")
    ax.set_ylabel("pH", color="m"); ax2.set_ylabel("T (K)", color="orange")
    ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # Row 4: Fg + Fs
    ax  = axes[4]
    ax2 = ax.twinx()
    ax.plot(t_vec,  df["Aeration rate(Fg:L/h)"].values,   "b-",  lw=1.2, label="Fg")
    ax2.plot(t_vec, df["Sugar feed rate(Fs:L/h)"].values, "g-",  lw=1.2, label="Fs")
    if df_base is not None:
        t_b = df_base["Time (h)"].values if "Time (h)" in df_base.columns else t_vec
        ax.plot(t_b,  df_base["Aeration rate(Fg:L/h)"].values,   "b--", lw=1.0, alpha=0.5)
        ax2.plot(t_b, df_base["Sugar feed rate(Fs:L/h)"].values, "g--", lw=1.0, alpha=0.5)
    ax.set_ylabel("Fg (L/h)", color="b"); ax2.set_ylabel("Fs (L/h)", color="g")
    ax.set_xlabel("Time (h)")
    ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fname = save_dir / f"batch_{batch_no:03d}_{model_name}_fault{fault}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"  Plot saved → {fname}")

    if show:
        plt.show()

    plt.close(fig)


# ── Metrics table printing ────────────────────────────────────────────────────

def print_metrics_table(results: list, use_mpc: bool):
    """Print a summary table of all batch results."""
    header = f"{'Batch':>5}  {'Fault':>5}  {'Bio RMSE':>10}  {'Bio MAE':>9}  {'Pen RMSE':>10}  {'Pen MAE':>9}"
    if use_mpc:
        header += f"  {'[Recipe] Bio':>14}  {'[Recipe] Pen':>14}"
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))

    for r in results:
        if use_mpc:
            m  = r["metrics_mpc"]
            mr = r["metrics_recipe"]
            print(
                f"{r['batch_no']:>5}  {r['fault']:>5}  "
                f"{m['bio']['RMSE']:>10.4f}  {m['bio']['MAE']:>9.4f}  "
                f"{m['pen']['RMSE']:>10.4f}  {m['pen']['MAE']:>9.4f}  "
                f"{mr['bio']['RMSE']:>14.4f}  {mr['pen']['RMSE']:>14.4f}"
            )
        else:
            m = r["metrics"]
            print(
                f"{r['batch_no']:>5}  {r['fault']:>5}  "
                f"{m['bio']['RMSE']:>10.4f}  {m['bio']['MAE']:>9.4f}  "
                f"{m['pen']['RMSE']:>10.4f}  {m['pen']['MAE']:>9.4f}"
            )

    # Aggregate averages
    if use_mpc:
        bio_rmse_avg = np.mean([r["metrics_mpc"]["bio"]["RMSE"] for r in results])
        pen_rmse_avg = np.mean([r["metrics_mpc"]["pen"]["RMSE"] for r in results])
        bio_rmse_base = np.mean([r["metrics_recipe"]["bio"]["RMSE"] for r in results])
        pen_rmse_base = np.mean([r["metrics_recipe"]["pen"]["RMSE"] for r in results])
        print("─" * len(header))
        print(
            f"{'MEAN':>5}  {'─':>5}  "
            f"{bio_rmse_avg:>10.4f}  {'─':>9}  {pen_rmse_avg:>10.4f}  {'─':>9}  "
            f"{bio_rmse_base:>14.4f}  {pen_rmse_base:>14.4f}"
        )

        # Penicillin yield gain table
        print("\nMPC vs Recipe penicillin yield comparison:")
        print(f"  {'Batch':>5}  {'Pen RMSE (MPC)':>16}  {'Pen RMSE (recipe)':>18}  {'Gain':>8}")
        for r in results:
            mpc_r  = r["metrics_mpc"]["pen"]["RMSE"]
            rec_r  = r["metrics_recipe"]["pen"]["RMSE"]
            gain   = (rec_r - mpc_r) / (rec_r + 1e-12) * 100
            sign   = "+" if gain >= 0 else ""
            print(f"  {r['batch_no']:>5}  {mpc_r:>16.4f}  {rec_r:>18.4f}  {sign}{gain:>7.1f}%")
    else:
        bio_rmse_avg = np.mean([r["metrics"]["bio"]["RMSE"] for r in results])
        pen_rmse_avg = np.mean([r["metrics"]["pen"]["RMSE"] for r in results])
        print("─" * len(header))
        print(
            f"{'MEAN':>5}  {'─':>5}  "
            f"{bio_rmse_avg:>10.4f}  {'─':>9}  "
            f"{pen_rmse_avg:>10.4f}  {'─':>9}"
        )

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Validation ────────────────────────────────────────────────────────────
    if args.model in ("pi_lstm_raman", "neural_ode_raman") and not args.knn_raman:
        print(
            f"[ERROR] Model '{args.model}' requires --knn-raman flag to supply "
            "Raman latent vectors. Re-run with --knn-raman."
        )
        sys.exit(1)

    # ── Setup ─────────────────────────────────────────────────────────────────
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device     = resolve_device(args.device)
    fault_list = pad_fault_list(args.fault or [0], args.n_batches)
    ckpt_path  = args.ckpt or DEFAULT_CKPTS[args.model]
    save_dir   = Path(args.save) if (args.plots or args.show) else None

    print(f"\nIndPenSim Simulator")
    print(f"  model      : {args.model}")
    print(f"  checkpoint : {ckpt_path}")
    print(f"  n_batches  : {args.n_batches}")
    print(f"  faults     : {fault_list}")
    print(f"  mpc        : {args.mpc}" +
          (f"  (horizon={args.mpc_horizon}, steps={args.mpc_steps}, bio_weight={args.mpc_bio_weight})"
           if args.mpc else ""))
    print(f"  knn_raman  : {args.knn_raman}")
    print(f"  device     : {device}")
    print(f"  seed       : {args.seed}")

    # ── Load algorithm ────────────────────────────────────────────────────────
    alg = load_algorithm(args.model, ckpt_path, device)

    # ── Load KNN Raman (if requested) ─────────────────────────────────────────
    knn_raman = None
    if args.knn_raman:
        print("\nBuilding KNN Raman selector ...")
        raman_enc = load_raman_encoder(device)
        from simulator.knn_raman import KNNRamanSelector
        knn_raman = KNNRamanSelector(
            csv_path   = "100_Batches_IndPenSim_V3.csv",
            raman_encoder = raman_enc,
            device     = device,
            cache_path = args.knn_cache,
            n_neighbors = args.knn_k,
        )

    # ── Build MPC (if requested) ──────────────────────────────────────────────
    mpc_ctrl = None
    if args.mpc:
        print("\nBuilding MPC controller ...")
        from simulator.mpc import BioreactorMPC, MPCBatchController
        seq_len  = int(alg.cfg.training.seq_len) if hasattr(alg.cfg.training, "seq_len") else 24
        mpc_opt  = BioreactorMPC(alg, horizon=args.mpc_horizon, steps=args.mpc_steps,
                                  bio_weight=args.mpc_bio_weight)
        mpc_ctrl = MPCBatchController(
            algorithm     = alg,
            mpc_optimizer = mpc_opt,
            knn_raman     = knn_raman,
            seq_len       = seq_len,
        )

    # ── Run batches ───────────────────────────────────────────────────────────
    results = []

    for i in range(args.n_batches):
        batch_no   = i + 1
        fault_code = fault_list[i]

        if args.mpc:
            r = run_batch_mpc(
                batch_no, fault_code, alg, mpc_ctrl, knn_raman,
                args.n_batches, fault_list,
            )
        else:
            r = run_batch_recipe(
                batch_no, fault_code, alg, knn_raman,
                args.n_batches, fault_list,
            )

        results.append(r)

        # Per-batch quick print
        if args.mpc:
            m = r["metrics_mpc"]
        else:
            m = r["metrics"]
        print(
            f"  Batch {batch_no}: Bio RMSE={m['bio']['RMSE']:.4f}  "
            f"Pen RMSE={m['pen']['RMSE']:.4f}"
        )

    # ── Summary table ─────────────────────────────────────────────────────────
    print_metrics_table(results, use_mpc=args.mpc)

    # ── Plots ─────────────────────────────────────────────────────────────────
    if args.plots or args.show:
        for r in results:
            plot_batch(
                result    = r,
                model_name = args.model,
                save_dir  = Path(args.save) if args.plots else None,
                show      = args.show,
                use_mpc   = args.mpc,
            )

    print("Done.")


if __name__ == "__main__":
    main()
