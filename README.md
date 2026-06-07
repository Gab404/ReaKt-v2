# PI-LSTM: Physics-Informed LSTM for Penicillin Fermentation

Modular implementation of Physics-Informed LSTM and Neural ODE models for
biomass and penicillin concentration prediction in the IndPenSim V3 bioreactor
simulation dataset. Three frozen Raman encoders (CDAE, FusionModel V4, CoAtNet V5)
can be plugged in as additional inputs.

---

## Models

| Model | Raman Encoder | Latent Dim | Description |
|---|---|---|---|
| `pi_lstm` | — | — | Physics-Informed LSTM, process features only |
| `pi_lstm_raman` | CDAE | 64 | PI-LSTM + frozen convolutional denoising autoencoder |
| `pi_lstm_v4` | FusionModel V4 | 512 | PI-LSTM + frozen CNN fusion encoder |
| `pi_lstm_v5` | CoAtNet V5 | 32 | PI-LSTM + frozen CoAtNet encoder |
| `neural_ode` | — | — | Neural ODE, process features only |
| `neural_ode_raman` | CDAE | 64 | Neural ODE + frozen CDAE |
| `neural_ode_v4` | FusionModel V4 | 512 | Neural ODE + frozen CNN fusion encoder |
| `neural_ode_v5` | CoAtNet V5 | 32 | Neural ODE + frozen CoAtNet encoder |

All Raman encoders are **frozen** during training — only the downstream model trains.

---

## Results

Penicillin concentration prediction from Raman spectra only (IndPenSim V3).
Evaluated on 18 fault-free validation batches and 10 faulty hold-out batches.
Metrics computed exclusively at sparse offline laboratory measurement points (~20 per batch).

| Model | Type | FF-val RMSE (g/L) | Fault RMSE mean (g/L) |
|---|---|---|---|
| PLS (10-fold RMSECV baseline) | Linear, Static | 0.2440 | 0.2561 |
| SVR | Non-linear, Static | 0.2399 | 0.2708 |
| CDAE-LSTM (vanilla) | Non-linear, Temporal | 0.0772 | 0.1306 |
| CDAE-PI-LSTM (LP-ODE) | Physics + Temporal | 0.0773 | 0.1268 |
| CDAE-GreyBox-PI-LSTM (Mass Bal.) | Physics + Temporal | **0.0812** | **0.1244** |

---

## Setup

**Requirements:** Python 3.10+, PyTorch 2.1+

```bash
git clone <repo>
cd PI_LSTM

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**External assets** (not versioned, place at the paths below):

| File | Path | Purpose |
|---|---|---|
| IndPenSim V3 dataset | `./100_Batches_IndPenSim_V3.csv` | Training/evaluation data |
| CDAE checkpoint | `./checkpoints/cdae_best_model.pth` | 64-d Raman encoder |
| FusionModel V4 checkpoint | `./checkpoints/reakt_fusion_v4_best.pth` | 512-d Raman encoder |
| CoAtNet V5 checkpoint | `./checkpoints/reakt_coatnet_v5_best.pth` | 32-d Raman encoder |
| Raman scaler | `./checkpoints/scaler_raman.pkl` | StandardScaler for V4 / V5 |

Only the assets required by the model variants you intend to use are needed.

---

## Usage

### Train

```bash
# Process-only baselines
python train.py --model pi_lstm
python train.py --model neural_ode

# With Raman encoder
python train.py --model pi_lstm_raman      # CDAE 64-d
python train.py --model pi_lstm_v4         # FusionModel V4 512-d
python train.py --model pi_lstm_v5         # CoAtNet V5 32-d
python train.py --model neural_ode_raman
python train.py --model neural_ode_v4
python train.py --model neural_ode_v5
```

**Optional flags:**

| Flag | Description |
|---|---|
| `--epochs N` | Override number of epochs from config |
| `--device cpu\|cuda` | Force device (default: auto-detect) |
| `--config PATH` | Use a custom YAML config instead of `configs/<model>.yaml` |
| `--ckpt PATH` | Override output checkpoint path |
| `--no-plots` | Skip saving prediction/history plots |

### Evaluate

```bash
# Single model
python evaluate.py --model pi_lstm_v4

# All models + comparison table
python evaluate.py --model all

# Save per-batch prediction grid plots
python evaluate.py --model pi_lstm_v4 --plots

# Save scatter plot comparing all models
python evaluate.py --model all --scatter ./outputs/scatter.png

# Compare Raman encoders via linear probe (Ridge regression)
python evaluate.py --compare-encoders
python evaluate.py --compare-encoders --raman-encoder v4   # single encoder
```

**Optional flags:**

| Flag | Description |
|---|---|
| `--model MODEL` | Model to evaluate; `all` runs every available checkpoint |
| `--ckpt PATH` | Override checkpoint path (single model only) |
| `--device cpu\|cuda` | Force device |
| `--plots` | Save per-batch prediction grid PNGs |
| `--scatter PATH` | Save scatter comparison PNG |
| `--compare-encoders` | Run linear probe comparison across CDAE / V4 / V5 |
| `--raman-encoder cdae\|v4\|v5\|all` | Filter encoders for `--compare-encoders` |

---

## Data

The dataset is **IndPenSim V3** (`100_Batches_IndPenSim_V3.csv`):

- 100 batches, ~1135 time steps each (Δt = 0.2 h), 2239 columns
- **90 fault-free** batches → 80 % train / 20 % val (random split, seed 42)
- **10 faulty** batches → fixed test set
- Biomass (X) and penicillin (P) are measured sparsely (~every 12 h); all other process variables are continuous
- Raman spectra (2200 wavenumbers per row) are available for all time steps; the first ~10 rows of each batch have zero spectra (instrument warm-up) and are handled by forward-filling the first valid encoding within each batch

---

## Architecture

### PI-LSTM

```
Input (seq_len=24, n_feat) ──► LSTM(hidden=64) ──► FC(64→32) ──► head_bio
                                                               └──► head_pen
                                                  └──► physics params: r, y_max, α, β
```

Physics loss penalises deviation from the logistic growth / Luedeking-Piret ODEs:

```
dX/dt = r · X · (1 − X / y_max)      (logistic biomass growth)
dP/dt = α · X − β · P                (Luedeking-Piret penicillin)
```

### Neural ODE

```
d[X, P]/dt = MLP([X, P, U(t)])
```

`U(t)` is piecewise-linear control interpolation. Solved with RK4 during training and Dopri5 during evaluation.

---

## Repository Layout

```
PI_LSTM/
├── train.py                      # Training entry point
├── evaluate.py                   # Evaluation entry point
├── simulator.py                  # Standalone IndPenSim simulator CLI
├── finetune_closed_loop.py       # Closed-loop fine-tuning script
│
├── configs/
│   ├── pi_lstm.yaml
│   ├── pi_lstm_raman.yaml
│   ├── pi_lstm_v4.yaml
│   ├── pi_lstm_v5.yaml
│   ├── neural_ode.yaml
│   ├── neural_ode_raman.yaml
│   ├── neural_ode_v4.yaml
│   └── neural_ode_v5.yaml
│
├── src/
│   ├── config.py                 # Config dataclass (loads YAML)
│   ├── autoencoder/
│   │   └── model.py              # CDAE_Raman architecture definition
│   ├── algorithms/
│   │   ├── base.py               # BaseAlgorithm ABC + ScalerBundle
│   │   ├── pi_lstm.py            # PILSTMAlgorithm
│   │   └── neural_ode.py         # NeuralODEAlgorithm
│   ├── data/
│   │   ├── dataset.py            # PenicillinDataModule
│   │   ├── raman_encoder.py      # Frozen CDAE wrapper (64-d)
│   │   └── reakt_encoders.py     # FusionModelV4 (512-d) / CoAtNetV5 (32-d) wrappers
│   ├── models/
│   │   ├── pi_lstm.py            # BioreactorLogiPINN (LSTM + physics params)
│   │   └── neural_ode.py         # NeuralODEModel (MLP right-hand side)
│   ├── physics/
│   │   └── losses.py             # Physics-informed loss terms
│   ├── evaluation/
│   │   └── metrics.py            # RMSE / MAE / R² helpers
│   └── visualization/
│       └── plots.py              # Training history + prediction plots
│
├── checkpoints/                  # All model weights (.pt / .pth)
│   ├── pi_lstm.pt
│   ├── pi_lstm_raman.pt
│   ├── pi_lstm_v4.pt
│   ├── pi_lstm_v5.pt
│   ├── neural_ode.pt
│   ├── neural_ode_raman.pt
│   ├── neural_ode_v4.pt
│   ├── neural_ode_v5.pt
│   ├── cdae_best_model.pth       # Frozen CDAE encoder (2200-D → 64-D)
│   ├── reakt_fusion_v4_best.pth  # Frozen FusionModel V4 encoder (512-D)
│   ├── reakt_coatnet_v5_best.pth # Frozen CoAtNet V5 encoder (32-D)
│   └── scaler_raman.pkl          # StandardScaler for V4 / V5 preprocessing
│
├── simulator/                    # IndPenSim V3 simulator package
│
├── outputs/                      # Training plots, logs, MPC sweep results
│
└── 100_Batches_IndPenSim_V3.csv  # Dataset (2.6 GB, not versioned)
```
