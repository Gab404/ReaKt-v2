#!/usr/bin/env python
"""
train.py
========
CLI entry point for training any benchmark temporal model.

Available models
----------------
    pi_lstm          PI-LSTM, 23 process features only
    neural_ode       Neural ODE, 23 process features only
    cdae_pi_lstm     PI-LSTM, 23 process feats + 64 CDAE Raman latents
    cvae_pi_lstm     PI-LSTM, 23 process feats + 64 CVAE Raman latents
    pca_pi_lstm      PI-LSTM, 23 process feats +  4 PCA Raman latents
    pls_pi_lstm      PI-LSTM, 23 process feats +  4 PLS Raman latents

Usage
-----
    python train.py --model pi_lstm
    python train.py --model cdae_pi_lstm

    # Override epochs and checkpoint path at the command line:
    python train.py --model pi_lstm --epochs 2 --ckpt /tmp/test.pt

    # Force CPU even when a GPU is available:
    python train.py --model neural_ode --device cpu
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


MODEL_NAMES = [
    "pi_lstm", "neural_ode",
    "cdae_pi_lstm", "cvae_pi_lstm", "pca_pi_lstm", "pls_pi_lstm",
    "delta_cdae_pi_lstm", "cdae_process_pi_lstm",
]


def _resolve_config_path(model_name: str) -> Path:
    """Return path to configs/<model_name>.yaml, raising if not found."""
    root = Path(__file__).parent
    p    = root / "configs" / f"{model_name}.yaml"
    if not p.exists():
        raise FileNotFoundError(
            f"Config not found: {p}\n"
            f"Expected one of: {', '.join(MODEL_NAMES)}"
        )
    return p


def _maybe_build_raman_encoder(cfg, device: torch.device):
    """
    Return the appropriate frozen Raman encoder, or ``None`` if
    ``use_raman=False`` in the config.

    Supported ``raman_encoder_type`` values
    ----------------------------------------
        cdae_v2  -> src.data.cdae_encoder.CDAERamanEncoderV2  (64-d)
        cvae_v2  -> src.data.cvae_encoder.CVAERamanEncoderV2  (64-d, posterior mean)
        pca_v2   -> src.data.pca_encoder.PCARamanEncoderV1    (K-d, fit by pca_baseline.py)
        pls_v2   -> src.data.pls_encoder.PLSRamanEncoderV1    (K-d, fit by pls_baseline.py)

    All four share the SAME preprocessing pipeline:
        Savitzky-Golay first derivative (window=15, polyorder=2)
        -> StandardScaler.transform()
        -> frozen encoder
    """
    use_raman = cfg.data.get("use_raman", False)
    if not use_raman:
        return None

    enc_type    = cfg.data.get("raman_encoder_type", None)
    ckpt_path   = cfg.data.get("raman_ckpt",   None)
    scaler_path = cfg.data.get("raman_scaler", None)

    if enc_type == "cdae_v2":
        ckpt_path   = ckpt_path   or "./checkpoints/cdae_best.pt"
        scaler_path = scaler_path or "./checkpoints/cdae_scaler.joblib"
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"CDAE-V2 checkpoint not found: {ckpt_path}")
        if not Path(scaler_path).exists():
            raise FileNotFoundError(f"CDAE-V2 scaler not found: {scaler_path}")
        from src.data.cdae_encoder import CDAERamanEncoderV2
        print(f"  Loading CDAERamanEncoderV2 from {ckpt_path}  (scaler: {scaler_path}) ...")
        return CDAERamanEncoderV2(ckpt_path, scaler_path, device)

    if enc_type == "cvae_v2":
        ckpt_path   = ckpt_path   or "./checkpoints/cvae_best.pt"
        scaler_path = scaler_path or "./checkpoints/cvae_scaler.joblib"
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"CVAE-V2 checkpoint not found: {ckpt_path}")
        if not Path(scaler_path).exists():
            raise FileNotFoundError(f"CVAE-V2 scaler not found: {scaler_path}")
        from src.data.cvae_encoder import CVAERamanEncoderV2
        print(f"  Loading CVAERamanEncoderV2 from {ckpt_path}  (scaler: {scaler_path}) ...")
        return CVAERamanEncoderV2(ckpt_path, scaler_path, device)

    if enc_type == "pca_v2":
        ckpt_path   = ckpt_path   or "./checkpoints/pca_best.joblib"
        scaler_path = scaler_path or "./checkpoints/pca_scaler.joblib"
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(
                f"PCA encoder checkpoint not found: {ckpt_path}\n"
                "Run `python pca_baseline.py` first to fit and save the PCA encoder."
            )
        if not Path(scaler_path).exists():
            raise FileNotFoundError(
                f"PCA scaler not found: {scaler_path}\n"
                "Run `python pca_baseline.py` first to fit and save the PCA scaler."
            )
        n_components = int(cfg.data.get("raman_latent_dim", 4))
        from src.data.pca_encoder import PCARamanEncoderV1
        print(
            f"  Loading PCARamanEncoderV1 from {ckpt_path}  "
            f"(scaler: {scaler_path}, n_components={n_components}) ..."
        )
        return PCARamanEncoderV1(
            pca_model_path=ckpt_path,
            scaler_path=scaler_path,
            n_components=n_components,
        )

    if enc_type == "pls_v2":
        ckpt_path   = ckpt_path   or "./checkpoints/pls_best.joblib"
        scaler_path = scaler_path or "./checkpoints/pls_scaler.joblib"
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(
                f"PLS encoder checkpoint not found: {ckpt_path}\n"
                "Run `python pls_baseline.py` first to fit and save the PLS encoder."
            )
        if not Path(scaler_path).exists():
            raise FileNotFoundError(
                f"PLS scaler not found: {scaler_path}\n"
                "Run `python pls_baseline.py` first to fit and save the PLS scaler."
            )
        n_components = int(cfg.data.get("raman_latent_dim", 4))
        from src.data.pls_encoder import PLSRamanEncoderV1
        print(
            f"  Loading PLSRamanEncoderV1 from {ckpt_path}  "
            f"(scaler: {scaler_path}, n_components={n_components}) ..."
        )
        return PLSRamanEncoderV1(
            pls_model_path=ckpt_path,
            scaler_path=scaler_path,
            n_components=n_components,
        )

    raise ValueError(
        f"Unknown raman_encoder_type '{enc_type}'. "
        f"Choose one of: 'cdae_v2', 'cvae_v2', 'pca_v2', 'pls_v2'."
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a PI-LSTM or Neural ODE benchmark model."
    )
    parser.add_argument(
        "--model", required=True,
        choices=MODEL_NAMES,
        help="Which model variant to train.",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to a YAML config (overrides default configs/<model>.yaml).",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override n_epochs from config.",
    )
    parser.add_argument(
        "--ckpt", default=None,
        help="Override output checkpoint path from config.",
    )
    parser.add_argument(
        "--device", default=None,
        help="Force device ('cpu' or 'cuda'). Defaults to auto-detect.",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip saving prediction / history plots.",
    )
    args = parser.parse_args()

    # ── Device ───────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 65}")
    print(f"  Model  : {args.model}")
    print(f"  Device : {device}")
    print(f"{'=' * 65}\n")

    # ── Config ────────────────────────────────────────────────────────────────
    from src.config import Config
    cfg_path = Path(args.config) if args.config else _resolve_config_path(args.model)
    cfg = Config.from_yaml(cfg_path)

    # Command-line overrides
    if args.epochs is not None:
        cfg.training.n_epochs = args.epochs
    if args.ckpt is not None:
        cfg.output.checkpoint = args.ckpt

    seed = int(cfg.data.get("seed", 42))
    _set_seed(seed)

    # ── Raman encoder (if needed) ─────────────────────────────────────────────
    raman_encoder = _maybe_build_raman_encoder(cfg, device)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("[1] Loading data ...")
    from src.data.dataset import PenicillinDataModule
    dm = PenicillinDataModule(cfg.data, raman_encoder=raman_encoder)
    dm.load()

    strategy   = cfg.data.get("split_strategy", "random")
    train_frac = float(cfg.data.get("train_frac", 0.80))
    train_b, val_b, test_b = dm.get_splits(
        train_frac=train_frac,
        seed=seed,
        strategy=strategy,
    )
    print(
        f"  Train: {len(train_b)}  |  Val: {len(val_b)}  |  "
        f"Test (faulty): {len(test_b)}"
    )

    # ── Algorithm ────────────────────────────────────────────────────────────
    print(f"\n[2] Initialising algorithm '{args.model}' ...")
    from src.algorithms import get_algorithm
    AlgoCls = get_algorithm(args.model)
    alg     = AlgoCls(cfg, device)

    # ── Train ────────────────────────────────────────────────────────────────
    print("\n[3] Training ...")
    history = alg.fit(train_b, val_b, verbose=True)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    print("\n[4] Evaluating ...")
    from src.evaluation.metrics import print_metrics_table

    print("  --- Val (fault-free) ---")
    val_metrics = alg.evaluate(val_b)
    print_metrics_table(val_metrics, label="Val")

    print("  --- Test (faulty) ---")
    test_metrics = alg.evaluate(test_b)
    print_metrics_table(test_metrics, label="Test")

    # ── Save checkpoint ───────────────────────────────────────────────────────
    print("\n[5] Saving checkpoint ...")
    ckpt_path = str(cfg.output.checkpoint)
    os.makedirs(Path(ckpt_path).parent, exist_ok=True)
    alg.save(
        ckpt_path,
        extra={"val_metrics": val_metrics, "test_metrics": test_metrics},
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.no_plots:
        print("\n[6] Saving plots ...")
        plots_dir = Path(str(cfg.output.plots_dir))
        os.makedirs(plots_dir, exist_ok=True)

        from src.visualization.plots import (
            plot_training_history,
            plot_predictions_grid,
        )

        plot_training_history(
            history,
            save_path=str(plots_dir / "training_history.png"),
            title=f"{args.model} — training history",
        )

        for split_name, batches, res in [
            ("val",  val_b,  val_metrics),
            ("test", test_b, test_metrics),
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
                        f"{args.model}  [{split_name}]  {tgt_label}  "
                        f"RMSE={m.get('RMSE', float('nan')):.3f}  "
                        f"R²={m.get('R2', float('nan')):.4f}"
                    ),
                    save_path=str(plots_dir / f"{split_name}_{tgt}.png"),
                )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"  {args.model} training complete.")
    print_metrics_table(val_metrics,  label="Val ")
    print_metrics_table(test_metrics, label="Test")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
