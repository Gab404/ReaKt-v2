# ReaKt-v2 — Penicillin Soft-Sensor Benchmark on IndPenSim V3

A clean, reproducible benchmark suite for **online penicillin concentration
prediction from Raman spectra**, comparing classical chemometric baselines
(PCA-PCR, PLS, SVR) against modern temporal deep-learning models
(Physics-Informed LSTM, Neural ODE) augmented with **Raman encoders** — either
frozen pre-trained (CDAE, CVAE, PCA, PLS), combined with process state
variables (CDAE + process), or trained end-to-end jointly with the LSTM
(Joint CDAE-PI-LSTM).

All eleven models share the **identical preprocessing pipeline, train/val/test
split, and evaluation protocol** (sparse offline measurement points only) so
that reported numbers are strictly apples-to-apples.

---

## Table of Contents

1. [Dataset](#1-dataset)
2. [Repository Layout](#2-repository-layout)
3. [Setup](#3-setup)
4. [The Eleven Models](#4-the-eleven-models)
5. [Preprocessing Pipeline](#5-preprocessing-pipeline)
6. [How to Use](#6-how-to-use)
7. [Baseline Results](#7-baseline-results)
8. [Reproducibility Notes](#8-reproducibility-notes)

---

## 1. Dataset

**IndPenSim V3** — `100_Batches_IndPenSim_V3.csv` (~2.6 GB, not versioned).

| Property                                | Value                                |
| --------------------------------------- | ------------------------------------ |
| Batches                                 | 100 (90 fault-free + 10 faulty)      |
| Time step (`dt`)                        | 0.2 h                                |
| Steps per batch                         | ~1,135                               |
| Process variables                       | 23 (flows, pressures, temperatures, …) |
| Raman channels                          | 2,001 wavenumbers (400–2,400 cm⁻¹)   |
| Target — Biomass `X`                    | sparse (~every 12 h)                 |
| Target — Penicillin `P`                 | sparse (~every 12 h)                 |

**Fixed split** (seed = 42):

| Split                | Source                                  | # batches |
| -------------------- | --------------------------------------- | --------- |
| **Train**            | 80 % of fault-free batches (random perm.) | 72        |
| **Val (FF)**         | 20 % of fault-free batches              | 18        |
| **Test (fault)**     | all faulty batches                      | 10        |

> The Penicillin column predicted by this benchmark is the **offline laboratory
> assay** (`P_offline`). Models are trained against the linearly-interpolated
> dense series but **metrics are only computed at the sparse measurement
> points** (~20 per batch). This mirrors the operating constraint of a real
> bioreactor where ground-truth assays arrive ~1×/shift.

Each batch's first ~10 Raman spectra are zero rows (instrument warm-up); they
are detected by an all-zero row mask and forward-filled with the first valid
encoding within the batch.

---

## 2. Repository Layout

```
ReaKt-v2/
├── 100_Batches_IndPenSim_V3.csv        # Dataset (not versioned)
│
├── train.py                            # Train any of 6 temporal models
├── evaluate.py                         # Evaluate 1 or all 6 checkpoints
│
├── pca_baseline.py                     # Static PCR baseline + PCA encoder fit
├── pls_baseline.py                     # Static PLS baseline + PLS encoder fit
├── svr_baseline.py                     # Static SVR (RBF) baseline (RandomSearch)
│
├── configs/                            # YAML configs (one per temporal model)
│   ├── pi_lstm.yaml                    # process-only PI-LSTM
│   ├── neural_ode.yaml                 # process-only Neural ODE
│   ├── cdae_pi_lstm.yaml               # PI-LSTM + frozen CDAE (64-d)
│   ├── cvae_pi_lstm.yaml               # PI-LSTM + frozen CVAE (64-d, posterior μ)
│   ├── pca_pi_lstm.yaml                # PI-LSTM + frozen PCA  (4-d)
│   └── pls_pi_lstm.yaml                # PI-LSTM + frozen PLS  (4-d)
│
├── checkpoints/                        # Trained weights, scalers, fitted encoders
│   ├── pi_lstm.pt                      # Temporal models
│   ├── neural_ode.pt
│   ├── cdae_pi_lstm.pt
│   ├── cvae_pi_lstm.pt
│   ├── pca_pi_lstm.pt
│   ├── pls_pi_lstm.pt
│   ├── cdae_best.pt    + cdae_scaler.joblib    # Frozen Raman encoders
│   ├── cvae_best.pt    + cvae_scaler.joblib
│   ├── pca_best.joblib + pca_scaler.joblib     # (fit by pca_baseline.py)
│   ├── pls_best.joblib + pls_scaler.joblib     # (fit by pls_baseline.py)
│   ├── pca_baseline.joblib                     # Static baselines
│   ├── pls_baseline.joblib
│   └── svr_baseline.joblib
│
├── outputs/                            # Per-model training plots + metrics PNGs
│   ├── pca_baseline/   pls_baseline/   svr_baseline/
│   └── cdae_pi_lstm/   cvae_pi_lstm/   pca_pi_lstm/   pls_pi_lstm/
│
└── src/
    ├── config.py                       # YAML → nested Config object
    │
    ├── autoencoder/
    │   └── model.py                    # CDAE_Raman, CVAE_Raman (1-D conv backbones)
    │
    ├── data/
    │   ├── dataset.py                  # PenicillinDataModule (CSV → batches → split)
    │   ├── cdae_encoder.py             # CDAERamanEncoderV2  (64-d)
    │   ├── cvae_encoder.py             # CVAERamanEncoderV2  (64-d, posterior μ)
    │   ├── pca_encoder.py              # PCARamanEncoderV1   ( K-d, K=4)
    │   └── pls_encoder.py              # PLSRamanEncoderV1   ( K-d, K=4)
    │
    ├── models/
    │   ├── pi_lstm.py                  # BioreactorLogiPINN  (process-only)
    │   ├── cdae_pi_lstm.py             # PI-LSTM with Raman-latent input
    │   └── neural_ode.py               # NeuralODEModel  (MLP RHS + torchdiffeq)
    │
    ├── algorithms/
    │   ├── base.py                     # BaseAlgorithm ABC, ScalerBundle, REGISTRY
    │   ├── pi_lstm.py                  # PILSTMAlgorithm
    │   ├── neural_ode.py               # NeuralODEAlgorithm
    │   ├── cdae_pi_lstm.py             # CDAEPILSTMAlgorithm
    │   ├── cvae_pi_lstm.py             # CVAEPILSTMAlgorithm
    │   ├── pca_pi_lstm.py              # PCAPILSTMAlgorithm
    │   └── pls_pi_lstm.py              # PLSPILSTMAlgorithm
    │
    ├── physics/losses.py               # Mass-balance + logistic / Luedeking-Piret terms
    ├── evaluation/metrics.py           # compute_metrics, print_metrics_table
    └── visualization/plots.py          # training history + prediction grid plots
```

The **algorithm classes** (`src/algorithms/*.py`) are the public training
entry points. Each holds its own model, scalers, and physics parameters and
implements `fit()`, `evaluate()`, `save()`, `load()` against a common
`BaseAlgorithm` contract.

---

## 3. Setup

### Requirements

- Python 3.10+
- PyTorch 2.1+ (with CUDA 12.1 for GPU support)
- A GPU is **not strictly required** (everything runs on CPU), but training
  the LSTM / Neural ODE models is ~10× faster on GPU.

### Install

```bash
python3.10 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### External assets

| File                                  | Path                                  | Required for                                |
| ------------------------------------- | ------------------------------------- | ------------------------------------------- |
| IndPenSim V3 dataset (~2.6 GB)        | `./100_Batches_IndPenSim_V3.csv`      | every model                                 |
| CDAE Raman autoencoder weights        | `./checkpoints/cdae_best.pt`          | `cdae_pi_lstm`                              |
| CDAE input StandardScaler             | `./checkpoints/cdae_scaler.joblib`    | `cdae_pi_lstm`                              |
| CVAE Raman autoencoder weights        | `./checkpoints/cvae_best.pt`          | `cvae_pi_lstm`                              |
| CVAE input StandardScaler             | `./checkpoints/cvae_scaler.joblib`    | `cvae_pi_lstm`                              |

PCA and PLS encoder artefacts are **produced on the fly** by running
`python pca_baseline.py` and `python pls_baseline.py`; you do **not** need
to obtain them separately.

---

## 4. The Eleven Models

### Static, Raman-only (chemometric baselines)

| Model       | Encoder dim | Pipeline                                                |
| ----------- | ----------- | ------------------------------------------------------- |
| **PCA-PCR** | K = 4       | SG d=1 → StandardScaler → PCA(4) → LinearRegression      |
| **PLS**     | K = 4       | SG d=1 →                       PLS(4, scale=True)        |
| **SVR**     | RBF kernel  | SG d=1 → StandardScaler → PCA(50) → StandardScaler → SVR(RBF) |

Hyperparameters for SVR are tuned by `RandomizedSearchCV` with
`GroupKFold(10)` by `batch_id` (12 iterations × 10 folds).

### Temporal, process-only (no Raman)

| Model          | Architecture                                              | Physics                                                                              |
| -------------- | --------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| **pi_lstm**    | 1 × LSTM(64) → FC(64→32) → 2 heads `(X̂, P̂)`            | Logistic growth + Luedeking-Piret ODEs annealed into the loss after epoch 30        |
| **neural_ode** | MLP(64) right-hand-side `d[X,P]/dt = f([X,P,U(t)])`, RK4-train / Dopri5-eval | learned dynamics (no explicit closure)                                              |

### Temporal, with frozen Raman encoder (the 4 variants)

All four share the **same downstream architecture** (2-layer LSTM(64) +
FC(32), penicillin-only head, mass-balance physics) — the only thing that
changes is the upstream Raman encoder:

| Model              | Raman encoder            | Latent | Encoder kind             |
| ------------------ | ------------------------ | ------ | ------------------------ |
| **cdae_pi_lstm**   | CDAE (CNN denoising AE)  | 64     | non-linear, unsupervised |
| **cvae_pi_lstm**   | CVAE (CNN variational AE)| 64     | non-linear, unsupervised |
| **pca_pi_lstm**    | PCA                      |  4     | linear, unsupervised     |
| **pls_pi_lstm**    | PLS (supervised by P)    |  4     | linear, supervised       |

The encoder is **frozen** (`requires_grad=False`); only the LSTM, FC, and the
physics scale `k_prod` are trained.

> **Note** — the four `*_pi_lstm` variants predict **Penicillin only** (no
> biomass head). The mass-balance physics enforces
> `ΔP ≈ k_prod · r_net · dt`.

### Temporal, frozen CDAE encoder + process state variables

| Model                    | Input                        | Input dim | Architecture |
| ------------------------ | ---------------------------- | --------- | ------------ |
| **cdae_process_pi_lstm** | 64-dim CDAE latent + 23 process vars | 87 | 2-layer LSTM(87→64) + FC(32) + pen head |

The 23 process variables (temperature, pH, dissolved O₂, substrate
concentration, aeration rate, feed rates, vessel volume, off-gas CO₂/O₂,
OUR, CER, …) are concatenated to the CDAE Raman latent before the LSTM.
Both halves are jointly MinMaxScaled. The frozen CDAE encoder and the physics
loss (`ΔP ≈ k_prod · r_net · dt`) are identical to `cdae_pi_lstm`.

This variant tests whether adding process state context on top of the pure
spectroscopic signal improves fault-batch generalisation.

### End-to-end (joint encoder + LSTM)

| Model                    | Raman encoder           | Latent | Training            |
| ------------------------ | ----------------------- | ------ | ------------------- |
| **joint_cdae_pi_lstm**   | CDAE Conv encoder (same backbone as CDAE_Raman) | 64 | encoder + LSTM trained jointly from scratch |

The Conv encoder (3× Conv1d+BN+ReLU+MaxPool → Linear(32 000→64)) is
initialised randomly and trained **end-to-end** together with the 2-layer LSTM,
FC head, and physics scale `k_prod`. No pre-training and no weight freezing.
Gaussian denoising noise (σ = 0.1) is injected on the raw Raman input during
training only. Training used a standalone script (`train_joint_cdae_pi_lstm.py`,
now removed) with `torch.compile`, AMP, and early stopping (patience = 50);
best val MSE reached at epoch 74, training stopped at epoch 124.

> Because this model was trained via a standalone script outside the main
> `train.py` pipeline, it is **not** reproducible through `python train.py
> --model joint_cdae_pi_lstm`; results are archived in the table below.

---

## 5. Preprocessing Pipeline

Identical across **all eleven models**:

```
Raw Raman spectrum (2001 wavenumbers)
        │
        ▼
Savitzky-Golay 1st derivative  (window=15, polyorder=2)
        │
        ▼
StandardScaler  (per-channel μ=0, σ=1, fit on train split)
        │
        ▼
┌──────────────┬─────────────┬─────────────┬──────────────┐
│   PCA(K=4)   │   PLS(K=4)  │  CDAE(64)   │  CVAE(64, μ) │
└──────────────┴─────────────┴─────────────┴──────────────┘
        │
        ▼
For temporal models:
  MinMaxScaler → concatenate with 23 process features → LSTM / NODE
```

Implementation notes:

- The first ~10 rows of each batch contain zero Raman spectra (instrument
  warm-up). These are detected by an all-zero mask **after** encoding and
  forward-filled with the first valid latent in the batch.
- Dense linear interpolation is used to supervise the model at every time
  step, but metrics are aggregated **only** at the sparse offline
  measurement points where the ground truth was actually assayed.
- The 23 process features are standard-scaled separately from the Raman
  latents; both go through a final per-model `MinMaxScaler` so the LSTM
  sees inputs in `[0, 1]`.

---

## 6. How to Use

### 6.1 Train the static baselines (also fits the PCA / PLS Raman encoders)

```bash
python pca_baseline.py        # → checkpoints/pca_baseline.joblib
                              #   checkpoints/pca_best.joblib   (encoder)
                              #   checkpoints/pca_scaler.joblib (encoder scaler)

python pls_baseline.py        # → checkpoints/pls_baseline.joblib
                              #   checkpoints/pls_best.joblib   (encoder)
                              #   checkpoints/pls_scaler.joblib (encoder scaler)

python svr_baseline.py        # → checkpoints/svr_baseline.joblib
```

The PCA / PLS baselines must be run **before** their corresponding
`*_pi_lstm` variants, since the static-fit encoder artefacts are needed at
train time.

### 6.2 Train a temporal model

```bash
python train.py --model pi_lstm
python train.py --model neural_ode
python train.py --model cdae_pi_lstm          # requires checkpoints/cdae_best.pt
python train.py --model cvae_pi_lstm          # requires checkpoints/cvae_best.pt
python train.py --model pca_pi_lstm           # requires `pca_baseline.py` first
python train.py --model pls_pi_lstm           # requires `pls_baseline.py` first
python train.py --model cdae_process_pi_lstm  # requires checkpoints/cdae_best.pt
```

Available flags:

| Flag                | Effect                                                       |
| ------------------- | ------------------------------------------------------------ |
| `--epochs N`        | Override the YAML `n_epochs` (e.g. for smoke-tests).         |
| `--config PATH`     | Use a custom YAML instead of `configs/<model>.yaml`.         |
| `--ckpt PATH`       | Override the output checkpoint path.                         |
| `--device cpu|cuda` | Force device (default: auto-detect).                         |
| `--no-plots`        | Skip saving prediction/history PNGs.                         |

### 6.3 Evaluate a single trained model

```bash
python evaluate.py --model cdae_pi_lstm
```

### 6.4 Evaluate every model and print a comparison table

```bash
python evaluate.py --model all
```

Useful flags:

| Flag                  | Effect                                                            |
| --------------------- | ----------------------------------------------------------------- |
| `--ckpt PATH`         | Override checkpoint path (single-model mode only).                |
| `--device cpu|cuda`   | Force device.                                                     |
| `--plots`             | Save per-batch `(val\|test) × (bio\|pen)` PNG grids.              |
| `--scatter PATH`      | Save a y-true vs y-pred scatter overlay PNG across all models.    |

### 6.5 Reproduce the full benchmark from scratch

```bash
# (optional) ensure GPU 0 is used if you have several
export CUDA_VISIBLE_DEVICES=0

# Static baselines + PCA/PLS encoder artefacts
python pca_baseline.py
python pls_baseline.py
python svr_baseline.py

# Temporal models
python train.py --model pi_lstm
python train.py --model neural_ode
python train.py --model cdae_pi_lstm
python train.py --model cvae_pi_lstm
python train.py --model pca_pi_lstm
python train.py --model pls_pi_lstm
python train.py --model cdae_process_pi_lstm

# Side-by-side comparison
python evaluate.py --model all --scatter ./outputs/scatter_all.png
```

---

## 7. Baseline Results

All metrics are Penicillin concentration (g/L) at the sparse offline
measurement points. **Bold** = best per column.

### 7.1 Static baselines

Evaluated on all 18 FF-val batches and the 10 fault batches.

| Model       | K    | RMSECV (g/L) | FF-val RMSE | FF-val R² | Fault mean RMSE |
| ----------- | ---- | ------------ | ----------- | --------- | --------------- |
| PCA-PCR     | 4    | 0.4679       | 0.4799      | 0.9979    | 0.4559          |
| **PLS**     | 4    | **0.3603**   | **0.3439**  | 0.9989    | **0.3354**      |
| SVR (RBF)   | —    | 0.4144       | 0.3807      | 0.9987    | 0.3889          |

- PCA explained variance with K = 4: **88.4 %** cumulative.
- SVR best hyperparameters (RandomizedSearchCV, 12 iter × 10 GroupKFold):
  `C = 34.89`, `ε = 0.0633`, `γ = 0.00138`.

### 7.2 Temporal models (Penicillin head)

| Model                        | Val RMSE (g/L) | Val R²     | Test RMSE (g/L) | Test R²    |
| ---------------------------- | -------------- | ---------- | --------------- | ---------- |
| pi_lstm                      | 1.5578         | 0.9765     | 1.6288          | 0.9641     |
| neural_ode                   | 2.2180         | 0.9574     | 3.2110          | 0.8642     |
| **cdae_pi_lstm**             | **0.0876**     | **0.9999** | 0.3358          | 0.9985     |
| cvae_pi_lstm                 | 0.1220         | 0.9999     | 0.5913          | 0.9953     |
| pca_pi_lstm                  | 0.1279         | 0.9998     | 0.5242          | 0.9963     |
| pls_pi_lstm                  | 0.1107         | 0.9999     | 0.5202          | 0.9963     |
| cdae_process_pi_lstm         | 0.1260         | 0.9998     | **0.2270**      | **0.9993** |
| joint_cdae_pi_lstm (e2e)     | 0.1356         | 0.9998     | 0.1641          | ≥0.990     |

> **cdae_pi_lstm** (Raman only) achieves the best val RMSE: the Raman
> spectrum alone is a near-perfect proxy for P on normal batches.
> **cdae_process_pi_lstm** (Raman + 23 process vars) trades a slight
> val degradation (+0.038 g/L) for a large gain on fault batches
> (test RMSE 0.227 vs 0.336 g/L, R² 0.9993 vs 0.9985) — the process
> state variables carry fault-discriminating information that the Raman
> latent alone does not capture.
> The **joint** end-to-end model shows a similar fault-batch advantage;
> its test RMSE benefit is partly driven by the fact that the encoder
> is co-optimised with the penicillin objective rather than the
> reconstruction objective.

### 7.3 Biomass head (process-only models only)

| Model      | Val Bio RMSE | Val Bio R² | Test Bio RMSE | Test Bio R² |
| ---------- | ------------ | ---------- | ------------- | ----------- |
| pi_lstm    | 0.317        | 0.9971     | 1.233         | 0.9507      |
| neural_ode | 0.223        | 0.9990     | 1.551         | 0.9443      |

The four `*_pi_lstm` Raman variants predict Penicillin only (no biomass head),
so this comparison is restricted to the process-only models.

---

## 8. Reproducibility Notes

- All splits are derived from `np.random.default_rng(seed=42)`. The split is
  applied to the **fault-free** batches; faulty batches form the fixed test
  set. Identical perm. across all scripts → byte-identical batch
  assignments.
- All models use **Savitzky-Golay first-derivative** preprocessing
  (`window=15, polyorder=2`) on the first **2,001** Raman channels.
- PyTorch determinism is best-effort: `torch.manual_seed(42)`,
  `numpy.random.seed(42)`, `random.seed(42)`. cuDNN non-determinism may still
  cause ±1 % run-to-run variation on GPU.
- A `_patch_portable_paths()` helper in `evaluate.py` rewrites any
  Windows-absolute `csv_path` baked into a checkpoint with the local
  `./100_Batches_IndPenSim_V3.csv` if the former does not exist — so
  checkpoints trained on another OS load without manual editing.
- If you have multiple GPUs, force the working one with:
  ```bash
  export CUDA_VISIBLE_DEVICES=0
  ```

---

## License

Internal project — Renault / Genopole / ReaKt. Not for redistribution.
