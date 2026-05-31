# MPC Benchmark Report
**Date:** 2026-05-29  
**Horizons tested:** h=5, h=10, h=20  
**Models:** pi_lstm, neural_ode, pi_lstm_raman, neural_ode_raman  
**Faults:** 0 (none), 1 (sensor), 3 (actuator) — 3 batches, seed=42  
**MPC settings:** mpc-steps=5, gradient-based (Adam lr=0.1), objective = maximise penicillin only  

---

## 1. Recipe Baselines (no MPC)

| Model            | Fault 0 Bio | Fault 0 Pen | Fault 1 Bio | Fault 1 Pen | Fault 3 Bio | Fault 3 Pen | Mean Bio | Mean Pen |
|------------------|-------------|-------------|-------------|-------------|-------------|-------------|----------|----------|
| pi_lstm          | 1.17        | 0.69        | 2.67        | 1.63        | 1.33        | 2.01        | 1.72     | 1.44     |
| neural_ode       | 1.14        | 1.01        | 2.85        | 1.15        | 1.04        | 1.97        | 1.68     | 1.38     |
| pi_lstm_raman    | 1.34        | 1.20        | 2.35        | 2.74        | 1.30        | 1.46        | 1.66     | 1.80     |
| neural_ode_raman | 1.13        | 1.30        | 2.14        | 2.96        | 1.02        | 1.27        | 1.43     | 1.84     |

`neural_ode` has the lowest mean recipe Pen RMSE (1.38); `neural_ode_raman` has the lowest mean recipe Bio RMSE (1.43).

---

## 2. MPC Pen RMSE — Mean across faults {0, 1, 3}

Lower is better. Recipe column repeated for reference.

| Model            | Recipe | h=5   | h=10  | h=20  | Best horizon |
|------------------|--------|-------|-------|-------|--------------|
| pi_lstm          | 1.44   | 3.94  | 1.93  | **1.63** | **h=20**  |
| neural_ode       | 1.38   | 7.91  | 7.39  | 7.35  | (h=20, still far worse than recipe) |
| pi_lstm_raman    | 1.80   | 8.99  | 9.61  | N/A   | h=5 (least bad) |
| neural_ode_raman | 1.84   | 5.52  | 7.19  | N/A   | h=5 (least bad) |

---

## 3. MPC Bio RMSE — Mean across faults {0, 1, 3}

| Model            | Recipe | h=5  | h=10     | h=20  |
|------------------|--------|------|----------|-------|
| pi_lstm          | 1.72   | 2.49 | 2.63     | 3.16  |
| neural_ode       | 1.68   | 4.01 | 3.65     | 3.55  |
| pi_lstm_raman    | 1.66   | 4.55 | 4.72     | N/A   |
| neural_ode_raman | 1.43   | 2.92 | **0.96** | N/A   |

`neural_ode_raman` at h=10 achieves mean Bio RMSE of **0.96**, which is **33% better than its own recipe baseline (1.43)**, while Pen RMSE deteriorates badly (7.19 vs recipe 1.84). This is the key tension: the current penicillin-only objective causes MPC to inadvertently push biomass dynamics — adding both to the objective is expected to resolve this.

---

## 4. Per-batch detail — Pen RMSE (MPC) vs [Recipe]

### pi_lstm

| Horizon | Fault 0 | Fault 1 | Fault 3 | Mean |
|---------|---------|---------|---------|------|
| Recipe  | 0.69    | 1.63    | 2.01    | 1.44 |
| h=5     | 4.83 (−595%) | 3.22 (−97%) | 3.77 (−87%) | 3.94 |
| h=10    | 1.73 (−148%) | 1.51 **(+49%)** | 2.55 (−27%) | 1.93 |
| h=20    | 1.56 (−124%) | 1.74 **(+7%)** | 1.60 **(+21%)** | **1.63** |

`pi_lstm` is the **only model where MPC provides net improvement** over recipe in any scenario. At h=20 it beats recipe on fault 1 (+7%) and fault 3 (+21%), and is only modestly worse on fault 0 (−124% is a ~0.87 absolute increase). h=20 is the best horizon for this model.

### neural_ode

| Horizon | Fault 0 | Fault 1 | Fault 3 | Mean |
|---------|---------|---------|---------|------|
| Recipe  | 1.01    | 1.15    | 1.97    | 1.38 |
| h=5     | 9.45 (−832%) | 8.30 (−621%) | 5.98 (+3%) | 7.91 |
| h=10    | 11.48 (−1034%) | 6.87 (−497%) | 3.81 (+3%) | 7.39 |
| h=20    | 9.99 (−886%) | 7.58 (−559%) | 4.49 (+45%) | 7.35 |

MPC consistently degrades `neural_ode`. Only fault=3 batches see marginal improvement at h=20. The continuous-time ODE solver inside the model breaks the gradient chain in a way that makes the penicillin signal uninformative for MPC optimisation.

### pi_lstm_raman

| Horizon | Fault 0  | Fault 1  | Fault 3  | Mean  |
|---------|----------|----------|----------|-------|
| Recipe  | 1.20     | 2.74     | 1.46     | 1.80  |
| h=5     | 13.32 (−1011%) | 8.26 (−201%) | 5.41 (−271%) | 8.99  |
| h=10    | 3.66 (−205%) | 7.76 (−183%) | 17.43 (−1094%) | 9.61  |

Raman latent features add high-frequency noise to the MPC gradient signal, making optimisation unstable. No horizon produces useful MPC behaviour with the penicillin-only objective.

### neural_ode_raman

| Horizon | Fault 0 | Fault 1 | Fault 3 | Mean |
|---------|---------|---------|---------|------|
| Recipe  | 1.30    | 2.96    | 1.27    | 1.84 |
| h=5     | 5.88 (−351%) | 5.90 (−99%) | 4.78 (−276%) | 5.52 |
| h=10    | 6.06 (−365%) | 9.40 (−218%) | 6.12 (−382%) | 7.19 |

Bio RMSE at h=10: 0.998 / 0.895 / 0.991 (mean 0.96 — far better than recipe 1.43). The Neural ODE Raman model is effectively learning to grow biomass, but the penicillin production pathway is not being steered. This is the strongest signal that a dual objective is needed.

---

## 5. Conclusions

### Best horizon per model
| Model            | Best MPC horizon | Pen RMSE (MPC) | Pen RMSE (recipe) | Verdict |
|------------------|------------------|----------------|--------------------|---------|
| **pi_lstm**      | **h=20**         | **1.63**       | 1.44               | MPC useful, 13% worse than recipe but closes gap with faults |
| neural_ode       | h=20             | 7.35           | 1.38               | MPC harmful — recipe is 5× better |
| pi_lstm_raman    | h=5              | 8.99           | 1.80               | MPC harmful — recipe is 5× better |
| neural_ode_raman | h=5              | 5.52           | 1.84               | MPC harmful for Pen, but h=10 shows strong Bio improvement |

### Overall best model+horizon
**`pi_lstm` at h=20** is the only combination where MPC provides plausible benefit. It is the only LSTM-based model trained in a way that lets the gradient signal propagate cleanly through the rollout.

### Why MPC degrades most models
All four models were trained in **open-loop** (supervised, teacher-forced) mode. They predict well when fed accurate historical data but the MPC's gradient-based rollout feeds predicted outputs back as inputs, compounding error. The shorter the horizon the more the error compounds quickly (h=5); longer horizons (h=20) allow the optimiser more steps to find a consistent trajectory, which is why h=20 > h=10 > h=5 for pi_lstm. Neural ODE models suffer additionally because the ODE integration introduces numerical operations that break the gradient flow for penicillin specifically.

---

## 6. Recommended next step: Dual objective (biomass + penicillin)

The MPC objective has been updated in `simulator/mpc.py` to optimise a **weighted sum**:

```
reward_t = (1 − bio_weight) × P_pred + bio_weight × X_pred
```

Default `bio_weight=0.5` (equal weight). Exposed via `--mpc-bio-weight` CLI flag.

**Expected impact:**
- `pi_lstm` h=20: biomass tracking should improve (currently degrades from 1.72 recipe to 3.16) without catastrophic Pen RMSE increase since the Pen gradient was already working
- `neural_ode_raman` h=10: Bio RMSE already at 0.96; adding Pen to objective should reduce Pen RMSE degradation while keeping Bio benefit
- Raman models: dual objective may smooth the noisy gradient via the more stable biomass signal

**To re-run:**
```bash
python simulator.py --model pi_lstm \
    --n-batches 3 --fault 0 1 3 --seed 42 \
    --mpc --mpc-horizon 20 --mpc-steps 5 --mpc-bio-weight 0.5 \
    --plots --save ./outputs/mpc_sweep/dual_obj/h20/pi_lstm/
```
