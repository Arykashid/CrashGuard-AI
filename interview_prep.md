# CrashGuard AI — Interview & Presentation Preparation

---

## 90-Second System Explanation

*Memorise this. Say it exactly. Time yourself — it must be under 90 seconds.*

"CrashGuard AI is a hybrid ensemble system for real-time CPU spike forecasting on cloud workloads. The core pipeline ingests raw CPU telemetry, engineers 15 features including lag structure, rolling statistics, and rate-of-change signals, then trains two complementary models: an LSTM with Monte Carlo Dropout for temporal pattern learning and calibrated uncertainty quantification, and an XGBoost component that captures non-linear feature interactions at the current timestep.

A regime-switching ensemble dynamically weights the two models based on whether the system is in a spike or normal operating state — detected via rolling statistics and velocity features. The uncertainty intervals are post-hoc calibrated using temperature scaling on the validation set to achieve 80% empirical coverage without retraining. The full pipeline uses walk-forward validation to prevent temporal leakage and is evaluated against ARIMA and naive baselines with Diebold-Mariano significance testing — confirming the improvement is statistically significant at p = 0.0086."

---

## 8 Tough Questions — Bulletproof Answers

---

**Q1: Why LSTM and not Transformer for this task?**

Transformers require significantly more data and compute to outperform LSTMs on short univariate sequences. For 60K rows sampled at 5-minute intervals, the attention mechanism provides minimal benefit over LSTM's recurrent structure — the temporal dependencies are local, not long-range. LSTM with MC Dropout gives better uncertainty quantification at a fraction of the inference cost. We evaluated TCN and Transformer architectures and LSTM outperformed both on this dataset.

---

**Q2: Why XGBoost on last timestep only — why not full sequence?**

XGBoost has no concept of sequence order. It treats every column as an independent feature. Feeding 60 timesteps × 15 features = 900 columns gives XGBoost 900 independent signals with no temporal structure — it cannot learn that column 1 is "one step ago" and column 15 is "fifteen steps ago." This causes severe overfitting. The last timestep captures the current system state — which is exactly what XGBoost needs for non-linear feature interaction learning.

---

**Q3: What does MC Dropout actually measure and what does it NOT measure?**

MC Dropout measures epistemic uncertainty — model uncertainty arising from limited training data. Each of the 100 forward passes uses a different random dropout mask, producing slightly different predictions. The spread across those predictions quantifies how uncertain the model is about its weights.

What it does NOT measure is aleatoric uncertainty — the irreducible noise in CPU data itself. A CPU spike that appears randomly cannot be predicted regardless of model quality. This is why our raw MC std is near zero — the model is confident about its mean prediction, not because the future is certain, but because aleatoric uncertainty is not captured by MC Dropout alone.

---

**Q4: Why walk-forward validation and not k-fold?**

k-fold cross-validation randomly shuffles data into folds. For time series, this means a model trained on data from 2024 Q3 could be validated on 2024 Q1 data it has "seen the future of." This is temporal leakage — the validation metric is optimistic and does not reflect real deployment performance. Walk-forward validation respects temporal ordering: the model always trains only on data it would have seen in production. This gives an honest estimate of real-world performance.

---

**Q5: Why log1p on the target and not standardisation?**

CPU utilisation has a heavy right tail — normal values cluster around 0.17 but spikes reach 1.0. Standardisation preserves this skew. log1p compresses the tail: a value of 1.0 becomes 0.693, reducing the spike's dominance in the loss function. This flattens the loss surface and forces the model to learn spike shape rather than predicting the conditional mean. At inference time we apply expm1 to recover the original scale.

---

**Q6: What is temperature scaling and why is it post-hoc?**

Temperature scaling finds a scalar T on the validation set such that the empirical coverage of the uncertainty interval matches the target (80%). If MC Dropout produces intervals that only cover 7% of true values, T scales those intervals wider until 80% coverage is achieved.

It is post-hoc because it requires no retraining — it is fitted after the model is fully trained, using only validation data. This prevents leakage from the test set. The alternative — retraining with a different loss function — would change the model's predictions, not just its uncertainty estimates.

---

**Q7: What does the Diebold-Mariano test prove and why does it matter?**

The Diebold-Mariano test asks: is the difference in RMSE between two forecasting models statistically significant, or could it be explained by random chance? A p-value of 0.0086 means there is less than a 0.9% probability that LSTM's improvement over ARIMA occurred by chance. This matters because RMSE differences on a single test set can be misleading — a model could outperform by luck on one particular test window. DM testing confirms the improvement is systematic and reproducible.

---

**Q8: What is the biggest weakness and what would you do with 3 more months?**

The biggest weakness is spike RMSE of 0.477 versus normal RMSE of 0.124. The model performs well on normal operating conditions but systematically under-predicts sustained multi-step spikes with no training precedent. The root cause is the fixed-window LSTM reverting to the conditional mean during novel spike trajectories.

With 3 more months I would implement online adaptation — specifically a Kalman filter post-processor on LSTM outputs that updates its state estimate as the spike progresses. This would reduce the mean reversion bias during detected spike regimes without requiring full model retraining. I would also deploy on real cloud infrastructure with a Prometheus data feed to replace the CSV source — validating that the architecture generalises beyond the Google Cluster Trace distribution.

---

## SOP Paragraph (For TUM/RWTH/LMU Applications)

*Use this verbatim in your Statement of Purpose. It is written in formal academic register.*

"As a third-year undergraduate in Artificial Intelligence and Data Science, I developed CrashGuard AI — a production-grade system for real-time CPU spike forecasting on cloud infrastructure. The project combines a stacked LSTM with Monte Carlo Dropout for temporal sequence modelling and calibrated epistemic uncertainty quantification, with an XGBoost ensemble component trained on engineered last-timestep features for non-linear interaction learning. A regime-switching ensemble dynamically allocates model weights based on spike detection via rolling statistics and velocity signals, and post-hoc temperature scaling calibrates the uncertainty intervals to empirical 80% coverage without retraining. Evaluated on 60,000 rows of real Google production telemetry using walk-forward validation, the system achieves statistically significant improvement over ARIMA baselines confirmed by Diebold-Mariano testing at p = 0.0086. This project deepened my understanding of the gap between research-grade ML and production deployment — particularly the challenges of uncertainty quantification, temporal leakage prevention, and the engineering tradeoffs between model complexity and inference reliability. I pursue graduate study to develop the theoretical foundations necessary to address these challenges at greater depth and rigour."

---

## Demo Script (3 Minutes Exact)

**Minute 1 — Open crashguard_ai.py**
"This is predicting my system's CPU right now. Real data. Real time. The blue line is actual CPU. The orange dotted line is what our model predicts will happen next. The shaded region is the 90% confidence interval — when we say 90%, 90% of true values fall inside it. That is calibrated AI, not a black box."

**Minute 2 — Trigger alert**
Lower spike threshold to 0.10 in sidebar.
"I am lowering the alert threshold. Watch the alert panel. There — a spike alert just fired. In production this goes to the on-call engineer's Slack in under one second. Before the crash. Not after. That is the difference between reactive monitoring and predictive observability."

**Minute 3 — Show Integrated Gradients**
Point to IG chart on right side.
"This tells you WHY the model made this prediction. cpu_diff1 — the rate of change — is the most important feature. The model is not a black box. We can explain every prediction at the feature level. That is what makes this production-ready, not just research-ready."

---

## Three Things To Say If You Freeze

1. "The key insight is that we are predicting, not detecting. Every other monitoring tool tells you when a crash happened. We tell you before."

2. "The Diebold-Mariano test confirms this is statistically significant — p equals 0.0086. This is not luck."

3. "The model was trained on real Google production data — 60,000 rows of actual server telemetry. This is not a toy dataset."

---

## Numbers To Memorise

| Number | What it means |
|---|---|
| 60,000 | Rows of real Google production data |
| 15 | Engineered features |
| 0.133 | XGBoost RMSE — below 0.14 target |
| 0.124 | Normal regime RMSE |
| 0.80 | Calibrated coverage |
| 0.0086 | DM test p-value — statistically significant |
| 100 | MC Dropout samples per prediction |
| 5 minutes | How far ahead we predict |
| 6.5% | Spike rate in dataset |


“The key idea is not perfect prediction — it’s predicting early enough to act. That’s what makes this system useful in production.”