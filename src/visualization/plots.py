"""
src/visualization/plots.py
==========================
Plotting utilities shared across all algorithms.

Functions
---------
plot_training_history   : train/val loss curves (+ optional lambda_physics)
plot_predictions_grid   : per-batch trajectory subplots
plot_scatter_comparison : predicted vs. true scatter for multiple models
"""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")   # headless

import matplotlib.pyplot as plt
import numpy as np


# ── Training history ──────────────────────────────────────────────────────────

def plot_training_history(
    history:   Dict[str, list],
    save_path: str,
    title:     str = "Training history",
) -> None:
    """
    Plot train / val loss on a log scale.

    If ``history`` contains a ``"lambda_phys"`` key, a twin-axis subplot
    shows the physics weight schedule below the loss curves.
    """
    has_phys = "lambda_phys" in history and any(v > 0 for v in history["lambda_phys"])
    n_panels = 2 if has_phys else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3.5 * n_panels))
    if n_panels == 1:
        axes = [axes]

    epochs = range(1, len(history.get("val_loss", [])) + 1)

    ax0 = axes[0]
    train_key = next(
        (k for k in ["train_loss", "train_total"] if k in history), None
    )
    if train_key:
        ax0.plot(epochs, history[train_key], label="Train loss", color="steelblue")
    ax0.plot(epochs, history["val_loss"], label="Val loss",   color="darkorange")
    ax0.set_yscale("log")
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("MSE loss (log scale)")
    ax0.set_title(title)
    ax0.legend()
    ax0.grid(True, alpha=0.3)

    if has_phys:
        ax1 = axes[1]
        ax1.plot(epochs, history["lambda_phys"], color="green")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("λ_physics")
        ax1.set_title("Physics loss weight schedule")
        ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {save_path}")


# ── Per-batch prediction grid ─────────────────────────────────────────────────

def plot_predictions_grid(
    batches:      list,
    y_preds:      List[np.ndarray],
    y_sparses:    List[np.ndarray],
    target_label: str,
    title:        str,
    save_path:    str,
    max_plots:    int = 12,
    time_col:     str = "Time (h)",
) -> None:
    """
    Grid of subplots — one per batch.

    Blue line   = full predicted trajectory [g/L]
    Orange dots = sparse offline measurements [g/L]
    """
    n     = min(len(y_preds), max_plots)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = np.array(axes).flatten()

    for i in range(n):
        ax   = axes[i]
        t_ph = batches[i][time_col].values

        ax.plot(t_ph, y_preds[i], lw=1.0, color="steelblue", label="Pred")

        sp   = y_sparses[i]
        mask = ~np.isnan(sp)
        ax.scatter(t_ph[mask], sp[mask], s=14, color="darkorange", zorder=5, label="Measured")

        if mask.sum() > 1:
            err  = y_preds[i][mask] - sp[mask]
            rmse = float(np.sqrt(np.mean(err ** 2)))
            ss_r = float(np.sum(err ** 2))
            ss_t = float(np.sum((sp[mask] - sp[mask].mean()) ** 2))
            r2   = 1.0 - ss_r / ss_t if ss_t > 0 else 0.0
            ax.set_title(f"Batch {i + 1}  RMSE={rmse:.2f}  R²={r2:.3f}", fontsize=8)
        else:
            ax.set_title(f"Batch {i + 1}", fontsize=8)

        ax.set_xlabel("Time (h)", fontsize=7)
        ax.set_ylabel(target_label, fontsize=7)
        ax.tick_params(labelsize=6)

    if n > 0:
        axes[0].legend(fontsize=7)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


# ── Scatter comparison ────────────────────────────────────────────────────────

def plot_scatter_comparison(
    models_results: Dict[str, Dict],
    save_path:      str,
    max_points:     int = 500,
) -> None:
    """
    Two-row scatter plot (biomass / penicillin) for each model.

    ``models_results`` : {model_label: evaluate() return dict}
    Each evaluate() dict must contain keys ``y_sparses_bio``, ``y_preds_bio``,
    ``y_sparses_pen``, ``y_preds_pen``.
    """
    n_models = len(models_results)
    fig, axes = plt.subplots(2, n_models, figsize=(4 * n_models, 8))
    if n_models == 1:
        axes = axes.reshape(2, 1)

    rng = np.random.default_rng(0)

    for col, (label, res) in enumerate(models_results.items()):
        for row, (sp_key, pd_key, tgt_name) in enumerate([
            ("y_sparses_bio", "y_preds_bio", "Biomass (g/L)"),
            ("y_sparses_pen", "y_preds_pen", "Penicillin (g/L)"),
        ]):
            ax = axes[row, col]

            # Gather all sparse measurement points
            yt_list, yp_list = [], []
            for sp_arr, pd_arr in zip(res[sp_key], res[pd_key]):
                mask = ~np.isnan(sp_arr)
                yt_list.append(sp_arr[mask])
                yp_list.append(pd_arr[mask])

            yt = np.concatenate(yt_list) if yt_list else np.array([])
            yp = np.concatenate(yp_list) if yp_list else np.array([])

            if len(yt) > max_points:
                idx = rng.choice(len(yt), max_points, replace=False)
                yt, yp = yt[idx], yp[idx]

            ax.scatter(yt, yp, s=12, alpha=0.5, color="steelblue")

            # Identity line
            lo = min(yt.min(), yp.min()) if len(yt) else 0
            hi = max(yt.max(), yp.max()) if len(yt) else 1
            ax.plot([lo, hi], [lo, hi], "r--", lw=1)

            # Metrics annotation
            if len(yt) > 1:
                err  = yp - yt
                rmse = float(np.sqrt(np.mean(err ** 2)))
                ss_r = float(np.sum(err ** 2))
                ss_t = float(np.sum((yt - yt.mean()) ** 2))
                r2   = 1.0 - ss_r / ss_t if ss_t > 0 else 0.0
                ax.text(
                    0.05, 0.93,
                    f"RMSE={rmse:.3f}\nR²={r2:.4f}",
                    transform=ax.transAxes,
                    fontsize=8,
                    verticalalignment="top",
                )

            ax.set_xlabel(f"Measured {tgt_name}", fontsize=8)
            ax.set_ylabel(f"Predicted {tgt_name}", fontsize=8)
            if row == 0:
                ax.set_title(label, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {save_path}")
