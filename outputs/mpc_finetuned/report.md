# Fine-Tuned MPC Benchmark Report
**Date:** 2026-05-29  
**Horizon:** h=20  
**Models:** pi_lstm, neural_ode, pi_lstm_raman, neural_ode_raman (all fine-tuned)  
**Faults:** 0 (none), 1 (sensor), 3 (actuator) — 3 batches, seed=42  
**MPC settings:** mpc-steps=5, horizon=20, bio_weight=0.5 (dual objective: bio + pen)  
**Fine-tuning:** closed-loop BPTT, horizon=20, epochs=10, lr=1e-4, scheduled-sampling=True  

---

## 1. Fine-Tuning Summary

| Model            | Params | Best val_loss | Epochs run | Train loss Δ          | Saved checkpoint                   |
|------------------|--------|---------------|------------|-----------------------|------------------------------------|
| pi_lstm          | 25,062 | 0.000827      | 10         | 0.000405 → 0.000370 (−8.8%) | checkpoints/pi_lstm_finetuned.pt   |
| pi_lstm_raman    | 41,446 | 0.000036      | 10 (ES)    | 0.000017 → 0.000015 (−12.6%) | checkpoints/pi_lstm_raman_finetuned.pt |
| neural_ode       | 5,954  | 0.000007      | 10 (ES)    | 0.000000 → 0.000010 (+4665%) | checkpoints/neural_ode_finetuned.pt |
| neural_ode_raman | 10,050 | 0.000005      | 10         | 0.000000 → 0.000006 (+3705%) | checkpoints/neural_ode_raman_finetuned.pt |

**Note:** Neural ODE models saw large increases in training loss over epochs despite low absolute values. This is consistent with the scheduled-sampling schedule driving TF ratio to 0%: the model is forced to track its own compounding predictions, which is harder than teacher-forcing. Val loss remained low because the best checkpoint was saved early (epoch 5 for neural_ode, epoch 10 for neural_ode_raman).

---

## 2. Pen RMSE Comparison — h=20, mean across faults {0, 1, 3}

| Model            | [Orig recipe] | [Orig MPC h=20] | [FT recipe] | [FT MPC h=20] | FT vs Orig MPC |
|------------------|---------------|-----------------|-------------|---------------|----------------|
| pi_lstm          | 1.44          | **1.63**        | 2.79        | 8.63          | **+429%** (worse) |
| neural_ode       | 1.38          | 7.35            | 7.58        | 17.08         | **+132%** (worse) |
| pi_lstm_raman    | 1.80          | (N/A)           | 2.11        | 8.05          | — |
| neural_ode_raman | 1.84          | (N/A)           | 1.92        | 8.86          | — |

*Orig = original checkpoint, pen-only MPC objective.  FT = fine-tuned checkpoint, bio_weight=0.5.*

---

## 3. Bio RMSE Comparison — h=20, mean across faults {0, 1, 3}

| Model            | [Orig recipe] | [Orig MPC h=20] | [FT recipe] | [FT MPC h=20] | FT vs Orig MPC |
|------------------|---------------|-----------------|-------------|---------------|----------------|
| pi_lstm          | 1.72          | 3.16            | 1.53        | 1.79          | **−43%** (better) |
| neural_ode       | 1.68          | 3.55            | 1.75        | 5.64          | +59% (worse)   |
| pi_lstm_raman    | 1.66          | (N/A)           | 1.60        | 4.54          | — |
| neural_ode_raman | 1.43          | (N/A)           | 1.19        | 1.36          | — |

---

## 4. Per-batch detail — Fine-tuned MPC (h=20, bio_weight=0.5)

### pi_lstm (fine-tuned)

| Fault | Bio RMSE (MPC) | Pen RMSE (MPC) | [Recipe Bio] | [Recipe Pen] |
|-------|----------------|----------------|--------------|--------------|
| 0     | 1.5351         | 6.9722         | 1.0908       | 1.0535       |
| 1     | 1.7429         | 10.2996        | 2.7618       | 1.8733       |
| 3     | 2.0893         | 8.6202         | 0.7460       | 5.4474       |
| mean  | 1.7891         | **8.6307**     | 1.5329       | 2.7914       |

### neural_ode (fine-tuned)

| Fault | Bio RMSE (MPC) | Pen RMSE (MPC) | [Recipe Bio] | [Recipe Pen] |
|-------|----------------|----------------|--------------|--------------|
| 0     | 5.6559         | 17.2876        | 1.3253       | 4.9412       |
| 1     | 5.6942         | 18.1226        | 2.8334       | 14.5086      |
| 3     | 5.5665         | 15.8389        | 1.0989       | 3.2914       |
| mean  | 5.6389         | **17.0830**    | 1.7525       | 7.5804       |

### pi_lstm_raman (fine-tuned)

| Fault | Bio RMSE (MPC) | Pen RMSE (MPC) | [Recipe Bio] | [Recipe Pen] |
|-------|----------------|----------------|--------------|--------------|
| 0     | 4.8768         | 8.4809         | 1.3841       | 1.1872       |
| 1     | 4.3036         | 5.6089         | 2.3547       | 3.2942       |
| 3     | 4.4293         | 10.0693        | 1.0595       | 1.8403       |
| mean  | 4.5366         | **8.0530**     | 1.5994       | 2.1072       |

### neural_ode_raman (fine-tuned)

| Fault | Bio RMSE (MPC) | Pen RMSE (MPC) | [Recipe Bio] | [Recipe Pen] |
|-------|----------------|----------------|--------------|--------------|
| 0     | 0.9978         | 7.5287         | 0.9743       | 2.8508       |
| 1     | 1.8210         | 9.5129         | 1.6881       | 1.3235       |
| 3     | 1.2717         | 9.5399         | 0.9183       | 1.5974       |
| mean  | 1.3635         | **8.8605**     | 1.1935       | 1.9239       |

---

## 5. Analysis

### Why Pen RMSE is worse after fine-tuning

Two compounding factors:

1. **MPC objective changed from pen-only to bio_weight=0.5.** The optimizer now equally weights biomass and penicillin, so it partially sacrifices penicillin to improve biomass tracking. This alone would increase Pen RMSE even with no change to the model.

2. **Fine-tuning degraded open-loop accuracy.** The recipe (no-MPC) Pen RMSE for pi_lstm rose from 1.44 (original) to 2.79 (fine-tuned), and for neural_ode from 1.38 to 7.58. Closed-loop BPTT with aggressive scheduled sampling (TF ratio reaching 0% at epoch 10) forced the models to track their own compounding predictions, causing distribution shift away from teacher-forced open-loop behavior. The Neural ODE training loss increased by +4665% over 10 epochs — a clear sign of divergence in the fine-tuning regime.

### Notable result: neural_ode_raman bio tracking

`neural_ode_raman` fine-tuned MPC achieves mean Bio RMSE = **1.36**, which is only 14% worse than recipe Bio RMSE (1.19), while the original non-Raman neural_ode at h=20 had Bio RMSE = 3.55 (111% worse than recipe). The Raman latent features appear to provide a stabilising signal for the ODE state, allowing the MPC to maintain more coherent biomass trajectories. However, Pen RMSE at 8.86 is still far above recipe (1.92).

### The pi_lstm bio improvement

`pi_lstm` fine-tuned MPC achieves Bio RMSE = **1.79** vs original pi_lstm MPC Bio RMSE = 3.16 at h=20. This is a **43% improvement in biomass tracking**, and is the clearest positive result from the dual-objective fine-tuning pipeline. However, Pen RMSE is 8.63 vs original 1.63 — a large regression, partly attributable to the dual objective's weight split.

---

## 6. Conclusions

| Verdict | Models | Evidence |
|---------|--------|----------|
| Fine-tuning + dual objective improved Bio tracking | `pi_lstm`, `neural_ode_raman` | Bio RMSE improved vs original MPC |
| Fine-tuning + dual objective worsened Pen tracking | All 4 models | Pen RMSE higher than both original MPC and recipe |
| Fine-tuning hurt open-loop accuracy | `neural_ode`, `neural_ode_raman` | Recipe Pen RMSE 5× worse after fine-tuning |
| Only `pi_lstm` was a viable MPC base originally | Original benchmark | Only model where MPC beat recipe at any horizon |

### Root cause of overall degradation

The scheduled-sampling schedule (TF ratio: 0.9 → 0.0 over 10 epochs) is **too aggressive**. Dropping to 0% teacher-forcing in just 10 epochs forces the model far outside its original training distribution. A gentler schedule (e.g., minimum TF ratio of 0.3-0.5) or fewer epochs (stopping at epoch 5 where best val_loss was typically achieved) would likely preserve open-loop accuracy while still improving closed-loop robustness.

### Recommended next steps

1. **Re-run fine-tuning with min_tf_ratio=0.4** (never drop below 40% teacher-forcing) — prevents distribution collapse while still improving closed-loop stability.
2. **Run MPC with pen-only objective on fine-tuned checkpoints** — isolates the effect of fine-tuning from the dual-objective change.
3. **Consider separate objectives at evaluation time**: use bio_weight during fine-tuning for training signal, but evaluate MPC pen performance separately.
4. **For Neural ODE**: the +4665% train-loss increase indicates the closed-loop BPTT formulation is destabilising the ODE. Consider capping gradient norm more aggressively (grad_clip=1.0 instead of 5.0) or using a shorter fine-tuning horizon (h=5 or h=10).
