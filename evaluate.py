#!/usr/bin/env python
"""
evaluate.py
===========
CLI entry point for evaluating trained model checkpoints.

Usage
-----
    # Evaluate a single model
    python evaluate.py --model pi_lstm

    # Evaluate all four models and print a comparison table
    python evaluate.py --model all

    # Point to a custom checkpoint
    python evaluate.py --model neural_ode --ckpt ./checkpoints/neural_ode.pt

    # Save scatter plots
    python evaluate.py --model all --scatter ./outputs/scatter.png

    # Compare Raman encoders (CDAE vs V4-CNN vs V5-CoAtNet) via linear probe
    python evaluate.py --compare-encoders
    python evaluate.py --compare-encoders --raman-encoder v4
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, Optional

import torch

warnings.filterwarnings("ignore")

ALL_MODELS = [
    "pi_lstm", "pi_lstm_raman", "pi_lstm_v4", "pi_lstm_v5",
    "neural_ode", "neural_ode_raman", "neural_ode_v4", "neural_ode_v5",
]

DEFAULT_CKPTS = {
    "pi_lstm":          "./checkpoints/pi_lstm.pt",
    "pi_lstm_raman":    "./checkpoints/pi_lstm_raman.pt",
    "pi_lstm_v4":       "./checkpoints/pi_lstm_v4.pt",
    "pi_lstm_v5":       "./checkpoints/pi_lstm_v5.pt",
    "neural_ode":       "./checkpoints/neural_ode.pt",
    "neural_ode_raman": "./checkpoints/neural_ode_raman.pt",
    "neural_ode_v4":    "./checkpoints/neural_ode_v4.pt",
    "neural_ode_v5":    "./checkpoints/neural_ode_v5.pt",
}


def _maybe_build_raman_encoder(cfg, device: torch.device):
    """Return the appropriate frozen Raman encoder, or None if use_raman=False."""
    use_raman = cfg.data.get("use_raman", False)
    if not use_raman:
        return None

    enc_type    = cfg.data.get("raman_encoder_type", "cdae")
    ckpt_path   = cfg.data.get("raman_ckpt",   None)
    scaler_path = cfg.data.get("raman_scaler",  None)

    if enc_type == "cdae":
        if ckpt_path is None:
            ckpt_path = "./checkpoints/cdae_best_model.pth"
        from src.data.raman_encoder import RamanEncoder
        print(f"  Loading CDAE RamanEncoder from {ckpt_path} ...")
        return RamanEncoder(ckpt_path, device)

    elif enc_type == "v4":
        if ckpt_path is None:
            ckpt_path = "./checkpoints/reakt_fusion_v4_best.pth"
        if scaler_path is None:
            scaler_path = "./checkpoints/scaler_raman.pkl"
        from src.data.reakt_encoders import FusionModelV4Encoder
        print(f"  Loading FusionModelV4Encoder from {ckpt_path} ...")
        return FusionModelV4Encoder(ckpt_path, scaler_path, device)

    elif enc_type == "v5":
        if ckpt_path is None:
            ckpt_path = "./checkpoints/reakt_coatnet_v5_best.pth"
        if scaler_path is None:
            scaler_path = "./checkpoints/scaler_raman.pkl"
        from src.data.reakt_encoders import CoAtNetV5Encoder
        print(f"  Loading CoAtNetV5Encoder from {ckpt_path} ...")
        return CoAtNetV5Encoder(ckpt_path, scaler_path, device)

    else:
        raise ValueError(
            f"Unknown raman_encoder_type '{enc_type}'. Choose 'cdae', 'v4', or 'v5'."
        )


def _load_data(cfg, device: torch.device):
    """Return (train_b, val_b, test_b) using the split stored in the checkpoint cfg."""
    import random
    import numpy as np
    seed = int(cfg.data.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    raman_enc = _maybe_build_raman_encoder(cfg, device)

    from src.data.dataset import PenicillinDataModule
    dm = PenicillinDataModule(cfg.data, raman_encoder=raman_enc)
    dm.load()

    strategy   = cfg.data.get("split_strategy", "random")
    train_frac = float(cfg.data.get("train_frac", 0.80))
    return dm.get_splits(train_frac=train_frac, seed=seed, strategy=strategy)


def _evaluate_one(
    model_name: str,
    ckpt_path:  str,
    device:     torch.device,
    save_plots: bool = False,
) -> Optional[Dict]:
    """Load checkpoint, reload data, run evaluate(), print results."""
    if not Path(ckpt_path).exists():
        print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
        return None

    print(f"\n{'─' * 60}")
    print(f"  Model : {model_name}")
    print(f"  Ckpt  : {ckpt_path}")

    from src.algorithms import get_algorithm, BaseAlgorithm

    # Load algorithm from checkpoint
    alg = BaseAlgorithm.load(ckpt_path, device)
    cfg = alg.cfg

    # Reload the dataset with the same split settings
    _, val_b, test_b = _load_data(cfg, device)

    from src.evaluation.metrics import print_metrics_table

    print("  Val (fault-free):")
    val_m = alg.evaluate(val_b)
    print_metrics_table(val_m, label="Val")

    print("  Test (faulty):")
    test_m = alg.evaluate(test_b)
    print_metrics_table(test_m, label="Test")

    if save_plots:
        from src.visualization.plots import plot_predictions_grid
        plots_dir = Path(str(cfg.output.plots_dir))
        plots_dir.mkdir(parents=True, exist_ok=True)

        for split_name, batches, res in [
            ("val",  val_b,  val_m),
            ("test", test_b, test_m),
        ]:
            for tgt in ["bio", "pen"]:
                tgt_label = "Biomass (g/L)" if tgt == "bio" else "Penicillin (g/L)"
                m = res.get(tgt, {})
                plot_predictions_grid(
                    batches=batches,
                    y_preds=res[f"y_preds_{tgt}"],
                    y_sparses=res[f"y_sparses_{tgt}"],
                    target_label=tgt_label,
                    title=(
                        f"{model_name}  [{split_name}]  {tgt_label}  "
                        f"RMSE={m.get('RMSE', float('nan')):.3f}  "
                        f"R²={m.get('R2', float('nan')):.4f}"
                    ),
                    save_path=str(plots_dir / f"eval_{split_name}_{tgt}.png"),
                )

    return {"val": val_m, "test": test_m}


def _print_comparison_table(all_results: Dict[str, Dict]) -> None:
    """Print a side-by-side comparison table for all evaluated models."""
    print(f"\n{'=' * 75}")
    print(f"  {'Model':<22}  {'Biomass RMSE':>12}  {'Bio R²':>8}  "
          f"{'Pen RMSE':>10}  {'Pen R²':>8}  {'Split':>6}")
    print(f"{'─' * 75}")
    for model_name, res in all_results.items():
        if res is None:
            continue
        for split_key, label in [("val", "val"), ("test", "test")]:
            m = res.get(split_key, {})
            bio = m.get("bio", {})
            pen = m.get("pen", {})
            print(
                f"  {model_name:<22}  "
                f"{bio.get('RMSE', float('nan')):>12.3f}  "
                f"{bio.get('R2',   float('nan')):>8.4f}  "
                f"{pen.get('RMSE', float('nan')):>10.3f}  "
                f"{pen.get('R2',   float('nan')):>8.4f}  "
                f"{label:>6}"
            )
    print(f"{'=' * 75}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def _compare_raman_encoders(args: argparse.Namespace, device: torch.device) -> None:
    """
    Standalone Raman encoder quality comparison via linear probe.

    For each selected encoder (CDAE, V4-CNN, V5-CoAtNet), freezes the
    encoder, extracts per-spectrum features from the full dataset, then
    trains a Ridge regression probe and evaluates it on the held-out faulty
    batches.  No downstream PI-LSTM / Neural ODE model is involved.

    Train : fault-free batches (same split used for model training)
    Test  : faulty batches
    """
    import math

    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler as SKStdScaler

    from src.data.dataset import FAULT_COL, TARGET_COLS

    # ── 1. Load and prepare raw CSV ───────────────────────────────────────────
    csv_path = args.csv
    print(f"\nLoading {csv_path} ...")
    df = pd.read_csv(csv_path)

    # Detect batch boundaries by time reset (same logic as PenicillinDataModule)
    df["_bid"] = (df["Time (h)"].diff().fillna(0) < 0).cumsum()

    # Interpolate sparse targets within each batch
    for col in TARGET_COLS:
        df[col] = df.groupby("_bid")[col].transform(
            lambda s: s.interpolate(method="linear", limit_direction="both")
        )
    df = df.dropna(subset=TARGET_COLS).reset_index(drop=True)

    # Fault-free batches → probe training | Faulty batches → probe evaluation
    fault_max  = df.groupby("_bid")[FAULT_COL].max()
    train_bids = fault_max[fault_max == 0].index.tolist()
    test_bids  = fault_max[fault_max >  0].index.tolist()

    df_train = df[df["_bid"].isin(train_bids)].reset_index(drop=True)
    df_test  = df[df["_bid"].isin(test_bids)].reset_index(drop=True)

    print(f"  Batches — train (fault-free): {len(train_bids)}  |  "
          f"test (faulty): {len(test_bids)}")
    print(f"  Rows    — train: {len(df_train):,}  |  test: {len(df_test):,}")

    y_train = df_train[TARGET_COLS].values.astype(np.float32)
    y_test  = df_test[TARGET_COLS].values.astype(np.float32)

    # ── 2. Build encoder registry ─────────────────────────────────────────────
    from src.data.raman_encoder  import RamanEncoder
    from src.data.reakt_encoders import CoAtNetV5Encoder, FusionModelV4Encoder

    sel = args.raman_encoder
    enc_specs: dict = {}
    if sel in ("cdae", "all"):
        enc_specs["CDAE (64-d)"] = lambda: RamanEncoder(
            args.cdae_ckpt, device
        )
    if sel in ("v4", "all"):
        enc_specs["V4-CNN (512-d)"] = lambda: FusionModelV4Encoder(
            args.v4_ckpt, args.raman_scaler, device
        )
    if sel in ("v5", "all"):
        enc_specs["V5-CoAtNet (32-d)"] = lambda: CoAtNetV5Encoder(
            args.v5_ckpt, args.raman_scaler, device
        )

    # ── 3. Run linear probe for each encoder ──────────────────────────────────
    results: dict = {}
    for enc_name, build_fn in enc_specs.items():
        print(f"\n{'─' * 60}")
        print(f"  Encoder: {enc_name}")
        enc = build_fn()

        feat_train = enc.encode_dataframe(df_train)
        feat_test  = enc.encode_dataframe(df_test)

        # Use only rows where the encoder returned a non-zero output
        ok_tr = feat_train.any(axis=1)
        ok_te = feat_test.any(axis=1)
        X_tr, Y_tr = feat_train[ok_tr], y_train[ok_tr]
        X_te, Y_te = feat_test[ok_te],  y_test[ok_te]
        print(f"  Valid rows — train: {ok_tr.sum():,}  |  test: {ok_te.sum():,}")

        # Standardise features for Ridge
        sc     = SKStdScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        fdim = enc.latent_dim if hasattr(enc, "latent_dim") else enc.feature_dim
        row: dict = {"feature_dim": fdim}
        for ti, tname in enumerate(["bio", "pen"]):
            probe = Ridge(alpha=1.0).fit(X_tr_s, Y_tr[:, ti])
            pred  = probe.predict(X_te_s)
            rmse  = math.sqrt(float(np.mean((pred - Y_te[:, ti]) ** 2)))
            r2    = float(r2_score(Y_te[:, ti], pred))
            row[f"{tname}_rmse"] = rmse
            row[f"{tname}_r2"]   = r2
            label = "Biomass" if tname == "bio" else "Penicillin"
            print(f"  {label:<10}  RMSE={rmse:.4f}  R²={r2:.4f}")

        results[enc_name] = row

    # ── 4. Summary comparison table ───────────────────────────────────────────
    print(f"\n{'=' * 74}")
    print("  Raman Encoder Comparison — Linear Probe  (Ridge, α=1.0)")
    print("  Raman features only — no process variables.  Test = faulty batches.")
    print(f"{'=' * 74}")
    hdr = (
        f"  {'Encoder':<22}  {'Dim':>4}  "
        f"{'Bio RMSE':>9}  {'Bio R²':>7}  "
        f"{'Pen RMSE':>9}  {'Pen R²':>7}"
    )
    print(hdr)
    print(f"  {'─' * 70}")
    for enc_name, r in results.items():
        print(
            f"  {enc_name:<22}  {r['feature_dim']:>4}  "
            f"{r['bio_rmse']:>9.4f}  {r['bio_r2']:>7.4f}  "
            f"{r['pen_rmse']:>9.4f}  {r['pen_r2']:>7.4f}"
        )
    print(f"{'=' * 74}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained PI-LSTM / Neural ODE checkpoints.")
    parser.add_argument(
        "--model",
        default="none" if "--compare-encoders" in __import__("sys").argv else "all",
        choices=ALL_MODELS + ["all", "none"],
        help="Which model to evaluate. Use 'none' to skip model evaluation (default when --compare-encoders is set).",
    )
    parser.add_argument(
        "--ckpt", default=None,
        help="Override checkpoint path (only valid when --model is a single model).",
    )
    parser.add_argument(
        "--device", default=None,
        help="Force device ('cpu' or 'cuda').",
    )
    parser.add_argument(
        "--plots", action="store_true",
        help="Save per-batch prediction grid plots.",
    )
    parser.add_argument(
        "--scatter", default=None,
        help="Path to save a scatter comparison PNG (all models).",
    )
    # ── Raman encoder comparison ──────────────────────────────────────────────
    parser.add_argument(
        "--compare-encoders", action="store_true",
        help=(
            "Compare CDAE, V4-CNN, and V5-CoAtNet Raman encoders via a linear "
            "probe (Ridge regression).  Can be combined with --model or run alone."
        ),
    )
    parser.add_argument(
        "--raman-encoder",
        default="all",
        choices=["cdae", "v4", "v5", "all"],
        help="Which encoder(s) to include in --compare-encoders (default: all).",
    )
    parser.add_argument(
        "--cdae-ckpt",
        default="./checkpoints/cdae_best_model.pth",
        help="CDAE checkpoint path (used by --compare-encoders).",
    )
    parser.add_argument(
        "--v4-ckpt",
        default="./checkpoints/reakt_fusion_v4_best.pth",
        help="FusionModel V4 checkpoint path (used by --compare-encoders).",
    )
    parser.add_argument(
        "--v5-ckpt",
        default="./checkpoints/reakt_coatnet_v5_best.pth",
        help="CoAtNetFusion V5 checkpoint path (used by --compare-encoders).",
    )
    parser.add_argument(
        "--raman-scaler",
        default="./checkpoints/scaler_raman.pkl",
        help="StandardScaler .pkl for V4 / V5 (used by --compare-encoders).",
    )
    parser.add_argument(
        "--csv",
        default="./100_Batches_IndPenSim_V3.csv",
        help="Dataset CSV path (used by --compare-encoders).",
    )
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"\nDevice: {device}")

    models_to_run = [] if args.model == "none" else (
        ALL_MODELS if args.model == "all" else [args.model]
    )

    all_results: Dict[str, Optional[Dict]] = {}
    # For scatter plot we need the raw per-batch prediction arrays
    scatter_inputs: Dict[str, Dict] = {}

    for model_name in models_to_run:
        if args.ckpt and len(models_to_run) == 1:
            ckpt_path = args.ckpt
        else:
            ckpt_path = DEFAULT_CKPTS[model_name]

        res = _evaluate_one(model_name, ckpt_path, device, save_plots=args.plots)
        all_results[model_name] = res
        if res is not None and args.scatter:
            # Merge val + test for scatter
            scatter_inputs[model_name] = {
                "y_preds_bio":    (res["val"]["y_preds_bio"]  + res["test"]["y_preds_bio"]),
                "y_preds_pen":    (res["val"]["y_preds_pen"]  + res["test"]["y_preds_pen"]),
                "y_sparses_bio":  (res["val"]["y_sparses_bio"] + res["test"]["y_sparses_bio"]),
                "y_sparses_pen":  (res["val"]["y_sparses_pen"] + res["test"]["y_sparses_pen"]),
            }

    if len(models_to_run) > 1:
        _print_comparison_table(all_results)

    if args.scatter and scatter_inputs:
        from src.visualization.plots import plot_scatter_comparison
        Path(args.scatter).parent.mkdir(parents=True, exist_ok=True)
        plot_scatter_comparison(scatter_inputs, args.scatter)

    if args.compare_encoders:
        _compare_raman_encoders(args, device)


if __name__ == "__main__":
    main()
