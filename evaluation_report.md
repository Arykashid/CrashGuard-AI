# CrashGuard AI — Evaluation Report

Generated automatically. Do not edit manually.

## Verdict

**⚠️ System requires retraining or recalibration**

## Metrics

| Metric | Value | Target |
|---|---|---|
| Ensemble RMSE | 0.170376 | < 0.14 |
| LSTM RMSE | 0.170376 | < 0.14 |
| Coverage 90% | 0.0951 | 0.70–0.90 |
| Confidence | 0.1560 | 0.50–0.70 |
| ECE | 0.1578 | < 0.05 |
| Calibration quality | Poor calibration | Well calibrated |

## Model Comparison

| Model | RMSE | MAE |
|---|---|---|
| LSTM (CrashGuard) | 0.170376 | 0.125077 |
| ARIMA | 0.174106 | 0.121074 |
| Moving Average | 0.154176 | 0.113072 |
| Naive | 0.241699 | 0.150814 |

## Spike vs Normal RMSE

| Regime | RMSE |
|---|---|
| Spike regime | N/A |
| Normal regime | N/A |

## Statistical Tests

**Diebold-Mariano test** (LSTM vs ARIMA): p = 0.008572
✅ LSTM is statistically significantly better than ARIMA (p < 0.05)

**LSTM beats ARIMA**: ✅ YES