#!/usr/bin/env python3
"""
visualize_latent_space.py
=========================
Generate 2-D latent-space projections for all three frozen Raman encoders
(CDAE, FusionModel V4, CoAtNet V5).

Each encoder produces one figure with three side-by-side scatter plots:
  1. Colored by fault type  (fault-free vs. faulty batches)
  2. Colored by biomass concentration (g/L)
  3. Colored by viscosity (centPoise)

Usage
-----
    python visualize_latent_space.py                        # UMAP, 8 000 pts
    python visualize_latent_space.py --method pca           # force PCA
    python visualize_latent_space.py --n-samples 0          # use all rows
    python visualize_latent_space.py --out outputs/latent_space/
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Column names ──────────────────────────────────────────────────────────────

BIOMASS_COL   = "Offline Biomass concentratio(X_offline:X(g L^{-1}))"
VISCOSITY_COL = "Viscosity(Viscosity_offline:centPoise)"
FAULT_COL     = "Fault reference(Fault_ref:Fault ref)"
BATCH_COL     = "Batch ID"

RAMAN_START   = 39    # first Raman column index in V3 CSV
RAMAN_N       = 2200


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_csv(csv_path: str) -> pd.DataFrame:
    print(f"  Reading {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def valid_raman_mask(df: pd.DataFrame) -> np.ndarray:
    """Return boolean mask of rows that have a non-zero Raman spectrum."""
    raman_cols = df.columns[RAMAN_START : RAMAN_START + RAMAN_N]
    energy = np.nansum(np.abs(df[raman_cols].values.astype(np.float32)), axis=1)
    return energy > 0


def interpolate_within_batches(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Linearly interpolate sparse offline measurements within each batch.
    Remaining NaNs at the edges of each batch are filled outward (ffill+bfill).
    The original DataFrame index is preserved so subsequent row-selection works.
    """
    df = df.copy()
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    result_parts = []
    for _, group in df.groupby(BATCH_COL, sort=False):
        g = group.copy()
        for col in cols:
            g[col] = g[col].interpolate(method="linear", limit_direction="both")
        result_parts.append(g)

    return pd.concat(result_parts).sort_index()


def subsample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    idx.sort()
    return df.iloc[idx].reset_index(drop=True)


# ── Encoder loader ────────────────────────────────────────────────────────────

ENCODER_CONFIGS = {
    "cdae": {
        "label": "CDAE",
        "dim":   "64-D",
        "color": "#4C72B0",
    },
    "v4": {
        "label": "FusionModel V4",
        "dim":   "512-D",
        "color": "#DD8452",
    },
    "v5": {
        "label": "CoAtNet V5",
        "dim":   "32-D",
        "color": "#55A868",
    },
}


def load_encoder(key: str, device: torch.device):
    if key == "cdae":
        from src.data.raman_encoder import RamanEncoder
        return RamanEncoder("checkpoints/cdae_best_model.pth", device)
    elif key == "v4":
        from src.data.reakt_encoders import FusionModelV4Encoder
        return FusionModelV4Encoder(
            "checkpoints/reakt_fusion_v4_best.pth",
            "checkpoints/scaler_raman.pkl",
            device,
        )
    elif key == "v5":
        from src.data.reakt_encoders import CoAtNetV5Encoder
        return CoAtNetV5Encoder(
            "checkpoints/reakt_coatnet_v5_best.pth",
            "checkpoints/scaler_raman.pkl",
            device,
        )
    raise ValueError(key)


# ── Dimensionality reduction ──────────────────────────────────────────────────

def project_2d(latents: np.ndarray, method: str) -> tuple[np.ndarray, str]:
    """
    Project (N, D) latents to (N, 2).  Returns (embedding, method_used).
    """
    if method == "auto":
        try:
            import umap  # noqa: F401
            method = "umap"
        except ImportError:
            method = "pca"

    print(f"    Projecting {latents.shape[0]:,} × {latents.shape[1]} → 2D  [{method.upper()}] ...")

    if method == "umap":
        import umap as _umap
        reducer = _umap.UMAP(
            n_components=2,
            n_neighbors=30,
            min_dist=0.05,
            metric="euclidean",
            random_state=42,
            verbose=False,
        )
        return reducer.fit_transform(latents), "UMAP"

    elif method == "tsne":
        from sklearn.manifold import TSNE
        emb = TSNE(
            n_components=2,
            perplexity=40,
            random_state=42,
        ).fit_transform(latents)
        return emb, "t-SNE"

    else:  # pca
        from sklearn.decomposition import PCA
        emb = PCA(n_components=2, random_state=42).fit_transform(latents)
        return emb, "PCA"


# ── Plotting ──────────────────────────────────────────────────────────────────

def _scatter_continuous(ax, emb, values, cmap, label, method_label):
    """Scatter with continuous colormap; NaN rows drawn in light grey first."""
    valid = ~np.isnan(values)

    if (~valid).any():
        ax.scatter(
            emb[~valid, 0], emb[~valid, 1],
            c="#E0E0E0", s=4, alpha=0.25, linewidths=0,
            rasterized=True, zorder=1,
        )

    sc = ax.scatter(
        emb[valid, 0], emb[valid, 1],
        c=values[valid], cmap=cmap,
        s=6, alpha=0.65, linewidths=0,
        rasterized=True, zorder=2,
    )
    cb = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label(label, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax.set_xlabel(f"{method_label} 1", fontsize=9)


def make_figure(
    emb:          np.ndarray,
    fault:        np.ndarray,
    biomass:      np.ndarray,
    viscosity:    np.ndarray,
    enc_key:      str,
    method_label: str,
    save_path:    Path,
) -> None:
    cfg = ENCODER_CONFIGS[enc_key]
    title = f"{cfg['label']}  ({cfg['dim']})  —  Latent Space  [{method_label}]"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    # ── Panel 1 : Fault type ─────────────────────────────────────────────────
    ax = axes[0]
    ff_mask    = fault == 0
    fault_mask = fault != 0

    ax.scatter(
        emb[ff_mask, 0], emb[ff_mask, 1],
        c="#4C72B0", s=5, alpha=0.45, linewidths=0,
        rasterized=True, label=f"Fault-free  (n={ff_mask.sum():,})",
    )
    if fault_mask.any():
        ax.scatter(
            emb[fault_mask, 0], emb[fault_mask, 1],
            c="#C44E52", s=7, alpha=0.65, linewidths=0,
            rasterized=True, label=f"Faulty  (n={fault_mask.sum():,})",
        )

    ax.legend(fontsize=8, markerscale=3, framealpha=0.7)
    ax.set_title("Fault type", fontsize=11)
    ax.set_xlabel(f"{method_label} 1", fontsize=9)
    ax.set_ylabel(f"{method_label} 2", fontsize=9)

    # ── Panel 2 : Biomass ────────────────────────────────────────────────────
    ax = axes[1]
    _scatter_continuous(ax, emb, biomass, "viridis",
                        "Biomass (g/L)", method_label)
    ax.set_title("Biomass concentration", fontsize=11)
    ax.set_ylabel(f"{method_label} 2", fontsize=9)

    # ── Panel 3 : Viscosity ──────────────────────────────────────────────────
    ax = axes[2]
    _scatter_continuous(ax, emb, viscosity, "plasma",
                        "Viscosity (cP)", method_label)
    ax.set_title("Viscosity", fontsize=11)
    ax.set_ylabel(f"{method_label} 2", fontsize=9)

    # Uniform axis style
    for ax in axes:
        ax.tick_params(labelsize=8)
        ax.set_aspect("auto")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Latent space visualizations for CDAE / V4 / V5 Raman encoders."
    )
    parser.add_argument(
        "--csv", default="./100_Batches_IndPenSim_V3.csv",
        help="Path to IndPenSim V3 CSV.",
    )
    parser.add_argument(
        "--method", default="auto",
        choices=["auto", "umap", "pca", "tsne"],
        help="Dimensionality-reduction method (default: UMAP if installed, else PCA).",
    )
    parser.add_argument(
        "--n-samples", type=int, default=8000,
        help="Max rows to project. 0 = use all valid rows (slow for UMAP). Default: 8000.",
    )
    parser.add_argument(
        "--encoders", nargs="+", default=["cdae", "v4", "v5"],
        choices=["cdae", "v4", "v5"],
        help="Which encoder(s) to visualise.",
    )
    parser.add_argument(
        "--out", default="./outputs/latent_space/",
        help="Output directory for PNG figures.",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"\nDevice: {device}")

    # ── Load and filter data ──────────────────────────────────────────────────
    print("\n[1] Loading data ...")
    df = load_csv(args.csv)

    print("  Interpolating biomass & viscosity within each batch ...")
    df = interpolate_within_batches(df, [BIOMASS_COL, VISCOSITY_COL])

    vmask = valid_raman_mask(df)
    df_valid = df[vmask].reset_index(drop=True)
    print(f"  Valid Raman rows: {vmask.sum():,} / {len(df):,}")

    if args.n_samples > 0 and len(df_valid) > args.n_samples:
        df_sub = subsample(df_valid, args.n_samples)
        print(f"  Subsampled to {len(df_sub):,} rows (seed=42)")
    else:
        df_sub = df_valid

    # Extract labels (aligned to df_sub)
    fault     = pd.to_numeric(df_sub[FAULT_COL],     errors="coerce").fillna(0).values
    biomass   = pd.to_numeric(df_sub[BIOMASS_COL],   errors="coerce").values.astype(float)
    viscosity = pd.to_numeric(df_sub[VISCOSITY_COL], errors="coerce").values.astype(float)

    print(f"  Biomass  — non-NaN: {(~np.isnan(biomass)).sum():,} / {len(biomass):,}")
    print(f"  Viscosity — non-NaN: {(~np.isnan(viscosity)).sum():,} / {len(viscosity):,}")
    print(f"  Faulty rows: {(fault != 0).sum():,} / {len(fault):,}")

    out_dir = Path(args.out)

    # ── Encode + visualise each encoder ──────────────────────────────────────
    for enc_key in args.encoders:
        cfg = ENCODER_CONFIGS[enc_key]
        print(f"\n[{enc_key.upper()}]  {cfg['label']}  ({cfg['dim']})")

        print(f"  Loading encoder ...")
        enc = load_encoder(enc_key, device)

        print(f"  Encoding {len(df_sub):,} rows ...")
        latents = enc.encode_dataframe(df_sub)
        print(f"  Latents shape: {latents.shape}")

        emb, method_label = project_2d(latents, args.method)

        save_path = out_dir / f"{enc_key}_latent_{method_label.lower().replace('-', '')}.png"
        make_figure(
            emb=emb,
            fault=fault,
            biomass=biomass,
            viscosity=viscosity,
            enc_key=enc_key,
            method_label=method_label,
            save_path=save_path,
        )

    print(f"\nDone. All figures saved to {out_dir}")


if __name__ == "__main__":
    main()
