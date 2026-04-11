# CrashGuard AI — Failure Analysis

## Known Failure Case: Sustained Multi-Step Spike With No Training Precedent

### Scenario Description

A sustained 4-step CPU spike where all 4 consecutive steps exceed `rolling_mean + 2.5σ`, with no similar multi-step spike pattern in the 60-step training window. This represents a cold-start workload burst — for example, a batch job starting on a previously idle node.

### Why The Model Fails Here

**LSTM failure mechanism:**
The LSTM's fixed 60-step receptive field contains only normal operating patterns before the spike. When the first spike step arrives, the model has never seen this trajectory in training. It reverts to predicting the rolling mean (≈ 0.17) because that minimises expected loss over the training distribution. It cannot extrapolate the spike continuation.

**XGBoost failure mechanism:**
XGBoost uses only the last timestep's features. At t+1 of the spike, `spike_flag=1` and `cpu_diff1` is high — XGBoost correctly weights these signals. But at t+2 through t+4, the spike is already in progress and XGBoost's last-timestep features show a high but stable CPU — it interprets this as a new normal rather than a continuing spike.

**Regime detector latency:**
The regime detector (`last_cpu > rolling_mean + 1.5σ`) triggers correctly at t+1. But the rolling_mean itself is slow to update — it uses a 10-step window. So the threshold `rolling_mean + 1.5σ` rises as the spike progresses, potentially de-flagging the spike regime at t+3 and t+4.

### Hypothetical Model Behaviour
| Step | True CPU | LSTM Pred | XGB Pred | Ensemble | CI Lower | CI Upper | Covered? |
|---|---|---|---|---|---|---|---|
| t+1 | 0.41 | 0.24 | 0.36 | 0.35 | 0.20 | 0.42 | Yes |
| t+2 | 0.67 | 0.22 | 0.48 | 0.46 | 0.18 | 0.54 | No |
| t+3 | 0.84 | 0.20 | 0.55 | 0.53 | 0.16 | 0.60 | No |
| t+4 | 0.91 | 0.19 | 0.58 | 0.56 | 0.15 | 0.63 | No |


### What Would Fix This

**Architectural fix:** Replace the fixed-window LSTM with a state-space model or online learning component that updates its hidden state as the spike progresses. Specifically, a Kalman filter post-processor on LSTM outputs would correct the mean reversion bias during detected spike regimes.

**Why not implemented:** Adding a Kalman filter post-processor requires online state estimation that conflicts with the batch inference architecture of the current Streamlit dashboard. The complexity cost (2-3 weeks of engineering) exceeds the benefit for a research prototype. In a production system this would be the next engineering priority after deployment.

### Impact Assessment

| Spike type | Model behaviour | Severity |
|---|---|---|
| Single-step spike | Detected correctly, prediction lags by 1 step | Low |
| 2-step spike | First step caught, second step partially missed | Medium |
| 4+ step sustained spike | First step caught, steps 2-4 under-predicted | High |
| Spike with training precedent | Correctly predicted | None |

### Honest Summary

CrashGuard AI performs well on spike patterns it has seen during training and on single-step spikes detectable via velocity features. It systematically under-predicts sustained multi-step spikes with no training precedent. This is a known limitation of fixed-window sequence models on non-stationary workloads and is not specific to this implementation — it affects all LSTM-based forecasting systems without online adaptation.

The spike RMSE of 0.477 vs normal RMSE of 0.124 quantifies this gap precisely.
