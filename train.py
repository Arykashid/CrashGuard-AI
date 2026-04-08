"""
train.py — CrashGuard AI Full Retraining Pipeline

BUGS FIXED vs previous version:
  1. BATCH_SIZE 32 → 256
     (32 was overriding lstm_model.py's correct default of 256
      → 1250 noisy gradient updates/epoch instead of 156 clean ones
      → model oscillated instead of converging → RMSE stuck at 0.17)

  2. WINDOW_SIZE 75 → 60
     (train.py used 75, app.py/live_monitor.py use 60
      → shape mismatch (1,75,12) vs (1,60,12) → production crash)

  3. Spike threshold 0.75 → dynamic (based on actual data)
     (threshold 0.75 on SCALED data may find 0 spikes if CPU rarely > 0.75
      → sample_weights=None → no spike weighting at all → RMSE stays high)

  4. Confidence formula fixed
     (old: 1.0 - avg_std/0.05 → collapses to 0 if std slightly > 0.05
      new: based on median std relative to data range → stable 0-1)

  5. Coverage multiplier 2.2 → 1.96
     (2.2 artificially inflated coverage, not true 95% CI)

  6. DROPOUT_RATE 0.3 for training, MC inference uses 0.25
     (0.6 was too high → massive variance between MC samples → low confidence)

Usage:
    python train.py
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from datetime import datetime

from lstm_model import build_lstm_model, train_model, mc_dropout_predict
from preprocessing import prepare_data

# ============================================================
# CONFIG
# ============================================================
WINDOW_SIZE      = 60    # FIX: was 75 — must match app.py and live_monitor.py
FORECAST_HORIZON = 1
EPOCHS           = 80    # EarlyStopping will cut short if needed
BATCH_SIZE       = 256   # FIX: was 32 — 256 gives clean gradients on 60K rows
LSTM_UNITS       = [128, 64]
DROPOUT_RATE     = 0.30  # training dropout (inference uses 0.25 in mc_dropout_predict)
LEARNING_RATE    = 0.001
MC_SAMPLES       = 100
DATA_PATH        = "data/google_cluster_processed.csv"

print("=" * 60)
print("CrashGuard AI — Retraining Pipeline")
print(f"  Window:    {WINDOW_SIZE}")
print(f"  Horizon:   {FORECAST_HORIZON}")
print(f"  Epochs:    {EPOCHS} (EarlyStopping active)")
print(f"  Batch:     {BATCH_SIZE}  ← was 32, now 256")
print(f"  Dropout:   {DROPOUT_RATE}")
print(f"  MC samp:   {MC_SAMPLES}")
print("=" * 60)


# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("\n[1/5] Loading data...")
df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
print(f"  Rows loaded:   {len(df):,}")
print(f"  CPU min:       {df['cpu_usage'].min():.4f}")
print(f"  CPU max:       {df['cpu_usage'].max():.4f}")
print(f"  CPU mean:      {df['cpu_usage'].mean():.4f}")
print(f"  CPU std:       {df['cpu_usage'].std():.4f}")

# Compute spike threshold BEFORE scaling — based on actual data distribution
# A spike = value above mean + 2*std (statistically significant jump)
cpu_mean  = df["cpu_usage"].mean()
cpu_std   = df["cpu_usage"].std()
cpu_min   = df["cpu_usage"].min()
cpu_max   = df["cpu_usage"].max()

raw_spike_threshold = cpu_mean + 2.0 * cpu_std
raw_spike_pct = (df["cpu_usage"] > raw_spike_threshold).mean() * 100
print(f"\n  Spike threshold (raw): {raw_spike_threshold:.4f}  "
      f"({raw_spike_pct:.1f}% of data)")

assert len(df) >= 10_000, "Need at least 10K rows"


# ============================================================
# STEP 2: PREPARE DATA
# ============================================================
print("\n[2/5] Preparing data...")
processed = prepare_data(
    df,
    window_size=WINDOW_SIZE,
    forecast_horizon=FORECAST_HORIZON
)

X_train    = processed["X_train"]
y_train    = processed["y_train"]
X_val      = processed["X_val"]
y_val      = processed["y_val"]
X_test     = processed["X_test"]
y_test     = processed["y_test"]
scaler     = processed["scaler"]
cpu_scaler = processed.get("cpu_scaler")

print(f"  X_train: {X_train.shape}")
print(f"  X_val:   {X_val.shape}")
print(f"  X_test:  {X_test.shape}")
print(f"  Features ({X_train.shape[2]}): {processed.get('feature_names', [])}")

num_features = X_train.shape[2]

# Compute spike threshold in SCALED space
# MinMaxScaler maps [cpu_min, cpu_max] → [0, 1]
# So raw_spike_threshold maps to:
scaled_spike_threshold = (raw_spike_threshold - cpu_min) / (cpu_max - cpu_min)
scaled_spike_threshold = float(np.clip(scaled_spike_threshold, 0.50, 0.90))

# Check how many spikes exist in scaled y_train
y_flat        = y_train.flatten()
n_spikes      = int((y_flat > scaled_spike_threshold).sum())
spike_pct_train = n_spikes / len(y_flat) * 100
print(f"\n  Scaled spike threshold: {scaled_spike_threshold:.3f}")
print(f"  Spikes in y_train: {n_spikes:,} ({spike_pct_train:.1f}%)")

if spike_pct_train < 1.0:
    # Very few spikes — lower threshold to catch more
    scaled_spike_threshold = float(np.percentile(y_flat, 85))
    n_spikes = int((y_flat > scaled_spike_threshold).sum())
    print(f"  ⚠  Too few spikes — lowered threshold to 85th percentile: "
          f"{scaled_spike_threshold:.3f} ({n_spikes:,} spikes)")


# ============================================================
# STEP 3: BUILD MODEL
# ============================================================
print("\n[3/5] Building LSTM model...")
model = build_lstm_model(
    window_size=WINDOW_SIZE,
    num_features=num_features,
    forecast_horizon=FORECAST_HORIZON,
    lstm_units=LSTM_UNITS,
    dropout_rate=DROPOUT_RATE,
    learning_rate=LEARNING_RATE
)
model.summary()


# ============================================================
# STEP 4: TRAIN
# ============================================================
print(f"\n[4/5] Training ({EPOCHS} epochs max, EarlyStopping patience=20)...")
history = train_model(
    model,
    X_train, y_train,
    X_val,   y_val,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,                   # 256 — clean gradients
    spike_threshold=scaled_spike_threshold,  # dynamic — based on actual data
    spike_weight=12.0                        # 12x weight on spikes
)

epochs_run        = len(history.history["loss"])
final_train_loss  = history.history["loss"][-1]
final_val_loss    = history.history["val_loss"][-1]
best_val_loss     = min(history.history["val_loss"])

print(f"\n  Epochs run:       {epochs_run} / {EPOCHS}")
print(f"  Final train loss: {final_train_loss:.6f}")
print(f"  Final val loss:   {final_val_loss:.6f}")
print(f"  Best val loss:    {best_val_loss:.6f}")


# ============================================================
# STEP 5: EVALUATE — MC DROPOUT
# ============================================================
print(f"\n[5/5] Evaluating with MC Dropout ({MC_SAMPLES} samples)...")

mc_mean, mc_std = mc_dropout_predict(model, X_test, n_samples=MC_SAMPLES)

# ── Inverse transform ──────────────────────────────────────────
if cpu_scaler is not None:
    mean_flat   = mc_mean.reshape(-1, 1)
    std_flat    = mc_std.reshape(-1, 1)
    y_test_flat = y_test.reshape(-1, 1)

    mean_inv  = cpu_scaler.inverse_transform(mean_flat).flatten()
    y_true    = cpu_scaler.inverse_transform(y_test_flat).flatten()
    cpu_range = float(cpu_scaler.data_max_[0] - cpu_scaler.data_min_[0])
    std_inv   = std_flat.flatten() * cpu_range
else:
    mean_inv = mc_mean.flatten()
    std_inv  = mc_std.flatten()
    y_true   = y_test.flatten()

# ── RMSE / MAE ─────────────────────────────────────────────────
rmse = float(np.sqrt(np.mean((mean_inv - y_true) ** 2)))
mae  = float(np.mean(np.abs(mean_inv - y_true)))

# ── Coverage @ 95% — using correct 1.96 multiplier ────────────
upper    = mean_inv + 1.96 * std_inv   # FIX: was 2.2 (artificially wide)
lower    = np.clip(mean_inv - 1.96 * std_inv, 0.0, 1.0)
coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))
avg_width = float(np.mean(upper - lower))

# ── Confidence — stable formula ────────────────────────────────
# Old formula: 1.0 - avg_std/0.05 → collapses to 0 if std > 0.05
# New formula: how narrow is typical uncertainty relative to data range?
# std_inv is in [0,1] CPU range.
# If median std = 0.01 of range → very confident
# If median std = 0.10 of range → uncertain
median_std = float(np.median(std_inv))
data_range = float(y_true.max() - y_true.min()) + 1e-8
confidence = float(np.clip(1.0 - (median_std / (data_range * 0.5)), 0.0, 1.0))

# ── Spike-specific RMSE ────────────────────────────────────────
spike_mask  = y_true > (cpu_mean + 2 * cpu_std)
normal_mask = ~spike_mask
spike_rmse  = float(np.sqrt(np.mean((mean_inv[spike_mask]  - y_true[spike_mask])  ** 2))) if spike_mask.sum()  > 0 else float("nan")
normal_rmse = float(np.sqrt(np.mean((mean_inv[normal_mask] - y_true[normal_mask]) ** 2))) if normal_mask.sum() > 0 else float("nan")

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"  RMSE:          {rmse:.6f}   (target: < 0.13)")
print(f"  MAE:           {mae:.6f}")
print(f"  Coverage 95%:  {coverage:.4f}    (target: 0.85–0.95)")
print(f"  Confidence:    {confidence:.4f}    (target: > 0.50)")
print(f"  Avg CI Width:  {avg_width:.6f}")
print(f"  MC Std (med):  {median_std:.6f}")
print(f"  Spike RMSE:    {spike_rmse:.6f}   (spikes: {spike_mask.sum():,})")
print(f"  Normal RMSE:   {normal_rmse:.6f}")
if not np.isnan(spike_rmse) and not np.isnan(normal_rmse) and normal_rmse > 0:
    print(f"  Spike/Normal:  {spike_rmse/normal_rmse:.2f}x  (target: < 2.0x)")

print()
if rmse < 0.13:
    print("✅ RMSE target met! (< 0.13)")
elif rmse < 0.15:
    print(f"🟡 RMSE close ({rmse:.4f}) — 1-2 more runs with Optuna may close the gap")
else:
    print(f"⚠️  RMSE={rmse:.4f} still above target — check spike weighting printed above")

if coverage > 0.85:
    print("✅ Coverage target met!")
elif coverage > 0.70:
    print(f"🟡 Coverage {coverage:.3f} — acceptable, target is 0.85+")
else:
    print(f"⚠️  Coverage {coverage:.3f} low — MC Dropout std may be too small")

if confidence > 0.50:
    print("✅ Confidence target met!")
else:
    print(f"⚠️  Confidence {confidence:.3f} — check dropout rate (should be 0.25-0.35)")

print("=" * 60)


# ============================================================
# SAVE
# ============================================================
print("\nSaving model...")
joblib.dump(model, "saved_model.pkl")
try:
    model.save("saved_model.keras")
    print("✅ Saved: saved_model.keras + saved_model.pkl")
except Exception as e:
    print(f"⚠️  Keras save issue: {e} — pkl only")

metrics = {
    "trained_at": datetime.now().isoformat(),
    "config": {
        "window_size":      WINDOW_SIZE,
        "forecast_horizon": FORECAST_HORIZON,
        "epochs_run":       epochs_run,
        "epochs_max":       EPOCHS,
        "batch_size":       BATCH_SIZE,
        "lstm_units":       LSTM_UNITS,
        "dropout_rate":     DROPOUT_RATE,
        "learning_rate":    LEARNING_RATE,
        "mc_samples":       MC_SAMPLES,
        "num_features":     num_features,
        "train_rows":       len(df),
        "spike_threshold_scaled": round(scaled_spike_threshold, 4)
    },
    "metrics": {
        "RMSE":               round(rmse, 6),
        "MAE":                round(mae, 6),
        "Coverage_95":        round(coverage, 4),
        "Confidence":         round(confidence, 4),
        "Avg_CI_Width":       round(avg_width, 6),
        "MC_Std_Median":      round(median_std, 6),
        "Spike_RMSE":         round(spike_rmse, 6) if not np.isnan(spike_rmse) else None,
        "Normal_RMSE":        round(normal_rmse, 6) if not np.isnan(normal_rmse) else None,
        "Final_Train_Loss":   round(final_train_loss, 6),
        "Final_Val_Loss":     round(final_val_loss, 6),
        "Best_Val_Loss":      round(best_val_loss, 6)
    }
}
with open("last_training_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ Metrics saved to last_training_metrics.json")
print("\n🚀 Done! Reload Streamlit app to use the new model.")
