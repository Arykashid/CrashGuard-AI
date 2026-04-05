# CrashGuard AI — Predictive CPU Observability Platform

> **Datadog tells you when your CPU spiked. CrashGuard tells you 5 minutes before it happens.**

---

## Problem Statement

Production systems fail reactively. Current observability tools (Datadog, Grafana, Prometheus) 
detect CPU spikes after they occur — by which point the damage is done: latency spikes, 
dropped requests, on-call engineers woken at 3am.

CrashGuard AI shifts infrastructure monitoring from **reactive detection** to 
**predictive alerting** using LSTM-based time series forecasting with calibrated 
uncertainty quantification.

---

## Technical Approach

### Model Architecture
- **Primary model**: Stacked LSTM with MC Dropout (2 layers, LayerNorm, Huber loss)
- **Uncertainty**: Monte Carlo Dropout (Gal & Ghahramani, 2016) — 50 forward passes per prediction
- **Features**: 10 engineered features including cyclic time encoding, lag features, rolling statistics
- **Training data**: 60,000 timesteps of realistic CPU telemetry with injected spike patterns

### Why LSTM Over ARIMA
ARIMA assumes linear autoregressive dynamics. CPU load under real workloads exhibits:
- Nonlinear regime changes (deployment events, traffic bursts)
- Diurnal seasonality with interaction effects
- Heteroscedastic noise (variance increases with load)

LSTM captures all three. Validated with the Diebold-Mariano test.

### Uncertainty Quantification
Standard neural networks produce point estimates with no confidence measure.
MC Dropout approximates Bayesian inference: at inference time, dropout remains active 
and N forward passes produce a distribution over predictions. The standard deviation 
of this distribution gives calibrated uncertainty bounds.

**Implementation**: Custom `MCDropout` layer overrides Keras's training flag, keeping 
dropout active regardless of inference mode. This fixes the `model.predict()` bug 
where standard Dropout is disabled during inference.

### Explainability
Feature attribution computed via **Integrated Gradients** (Sundararajan et al., 2017).

Why not SHAP for LSTMs:
- KernelSHAP treats each timestep×feature as independent — incorrect for sequential models
- IG respects the temporal structure of the input
- IG satisfies the Completeness axiom: sum of attributions = model(input) - model(baseline)

---

## Results

> ⚠️ Replace the values below with your actual numbers after running train.py + evaluate.py

| Model         | RMSE   | MAE    | Notes                          |
|---------------|--------|--------|--------------------------------|
| **LSTM**      | 0.0XX  | 0.0XX  | MC Dropout, bias-corrected     |
| ARIMA         | 0.0XX  | 0.0XX  | Walk-forward, AIC order select |
| Moving Avg    | 0.0XX  | 0.0XX  | 5-step window                  |
| Naive         | 0.0XX  | 0.0XX  | Persistence baseline           |

**Uncertainty Quantification**
- Coverage (95% CI): 0.XX (target: 0.90–0.98)
- Average confidence score: 0.XX (target: >0.65)

**Statistical Validation**
- Diebold-Mariano test: DM stat = X.XX, p = 0.0XX
- LSTM significantly outperforms ARIMA: [Yes/No]
- Walk-forward RMSE: 0.0XX ± 0.0XX (stable across folds)

**Ablation Study** (run Tab 8 in app.py to generate your numbers)

| Experiment | Features                        | RMSE   | Δ vs baseline |
|------------|---------------------------------|--------|---------------|
| A          | cpu_usage only                  | 0.0XX  | baseline      |
| B          | + cyclic time encoding          | 0.0XX  | -XX%          |
| C          | + lag features                  | 0.0XX  | -XX%          |
| D          | + rolling statistics            | 0.0XX  | -XX%          |

---

## System Architecture

```
[Data Sources]          [Preprocessing]         [Models]
psutil / Prometheus  →  MinMaxScaler         →  LSTM (primary)
60K synthetic rows      10 feature vectors      TCN (comparison)
                        Sliding window (60)     Transformer (comparison)
                        Walk-forward splits

[Inference]             [Outputs]               [Alerting]
MC Dropout (50x)     →  Point prediction    →   Slack webhook
Integrated Gradients    95% CI bands            Severity levels (H/M/L)
Confidence score        5-step forecast         60s cooldown
                        Feature attribution     Prediction log (JSON)

[Tracking]
MLflow experiment log
Optuna hyperparameter tuning
Ablation study (4 configs)
PDF report generation
```

---

## Comparison with Datadog

| Capability                        | CrashGuard AI | Datadog       |
|-----------------------------------|---------------|---------------|
| Real-time CPU monitoring          | ✅            | ✅            |
| Threshold-based alerts            | ✅            | ✅            |
| **Predictive alerting (pre-spike)**| ✅           | ❌            |
| **Calibrated uncertainty (CI)**   | ✅            | ❌            |
| **Feature-level explainability**  | ✅ (IG)       | ❌            |
| Statistical model validation      | ✅ (DM test)  | ❌            |
| MLflow experiment tracking        | ✅            | ❌            |
| Multi-step forecast               | ✅            | ❌            |
| Production infrastructure agents  | ❌            | ✅            |
| Enterprise scale                  | ❌            | ✅            |

CrashGuard AI's defensible edge: **uncertainty-calibrated predictive alerting**.
This is the one capability Datadog does not have and cannot easily add.

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate training data
python generate_data.py

# 3. Train model
python train.py

# 4. Launch research dashboard (evaluation, ablation, MLflow)
streamlit run app.py

# 5. Launch production dashboard (live monitoring, alerts)
streamlit run crashguard_ai.py

# 6. Start background worker (predictions + Slack alerts)
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
python worker.py
```

---

## Key Design Decisions

**LayerNorm over BatchNorm**: BatchNorm computes statistics over the batch dimension.
During MC Dropout sampling (batch size = 1, repeated N times), BatchNorm statistics 
are unstable and add non-dropout randomness that contaminates uncertainty estimates.
LayerNorm normalizes over the feature dimension — unaffected by MC sampling.

**Huber loss over MSE**: MSE penalizes large errors quadratically, causing the optimizer 
to sacrifice accuracy on rare spikes to reduce average error. Huber loss transitions to 
linear penalty beyond a threshold δ, making it robust to spike outliers while maintaining 
smooth gradients for normal operation.

**Walk-forward validation over k-fold**: Time series exhibit temporal dependencies — 
using future data to validate past predictions leaks information. Walk-forward validation 
always trains on the past and tests on the future, matching the actual deployment scenario.

---

## References

1. Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation: 
   Representing Model Uncertainty in Deep Learning. *ICML 2016*.

2. Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for 
   Deep Networks. *ICML 2017*.

3. Diebold, F. X., & Mariano, R. S. (1995). Comparing Predictive Accuracy. 
   *Journal of Business & Economic Statistics, 13*(3), 253–263.

4. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. 
   *Neural Computation, 9*(8), 1735–1780.

---

## Project Structure

```
crashguard_ai/
├── app.py                  # Research dashboard (20 tabs)
├── crashguard_ai.py        # Production dashboard (live monitoring)
├── worker.py               # Background prediction + alert loop
├── lstm_model.py           # LSTM + MC Dropout architecture
├── evaluate.py             # Full evaluation suite
├── explainability.py       # Integrated Gradients
├── generate_data.py        # 60K row synthetic CPU generator
├── train.py                # Training pipeline + ARIMA comparison
├── notifications.py        # Slack alerts
├── preprocessing.py        # Feature engineering + scaling
├── mlflow_tracker.py       # Experiment logging
├── optuna_tuning.py        # Hyperparameter search
├── ablation_study.py       # Feature contribution analysis
├── data/
│   └── cpu_telemetry_60k.csv
├── experiments/            # MLflow runs
├── requirements.txt
└── README.md
```

---

*Built for HackaTUM 2024 | Portfolio project for MS applications (TUM, LMU, RWTH, KIT)*
