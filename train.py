"""
train.py — CrashGuard AI Full Retraining Pipeline

FIXES in this version:
  1. log1p applied to target before training
  2. train_model() — spike_threshold/spike_weight removed
  3. mc_dropout_predict() unpacks 4 values
  4. Coverage uses quantile-based CI
  5. XGBoost ensemble with regime-switching
  6. weight_decay passed to build_lstm_model
  7. Temperature scaling calibration added after evaluation (NEW)
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from datetime import datetime

from lstm_model import build_lstm_model, train_model, mc_dropout_predict, inverse_log1p
from preprocessing import prepare_data
from xgboost_model import train_xgb, predict_xgb, ensemble_predict, save_xgb
from calibration import find_temperature, save_temperature

# ============================================================
# CONFIG
# ============================================================
WINDOW_SIZE      = 60
FORECAST_HORIZON = 1
EPOCHS           = 30
BATCH_SIZE       = 256
LSTM_UNITS       = [256, 128]
DROPOUT_RATE     = 0.20
LEARNING_RATE    = 0.001
WEIGHT_DECAY     = 1e-4
MC_SAMPLES       = 100
DATA_PATH        = "data/google_cluster_processed.csv"

print("=" * 60)
print("CrashGuard AI — Retraining Pipeline")
print(f"  Window:      {WINDOW_SIZE}")
print(f"  Horizon:     {FORECAST_HORIZON}")
print(f"  Epochs:      {EPOCHS} (EarlyStopping active)")
print(f"  Batch:       {BATCH_SIZE}")
print(f"  Dropout:     {DROPOUT_RATE}")
print(f"  WeightDecay: {WEIGHT_DECAY}")
print(f"  MC samples:  {MC_SAMPLES}")
print("=" * 60)


# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("\n[1/7] Loading data...")
df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
print(f"  Rows loaded: {len(df):,}")
print(f"  CPU mean:    {df['cpu_usage'].mean():.4f}")
print(f"  CPU std:     {df['cpu_usage'].std():.4f}")
print(f"  CPU max:     {df['cpu_usage'].max():.4f}")

cpu_mean = df["cpu_usage"].mean()
cpu_std  = df["cpu_usage"].std()

assert len(df) >= 10_000, "Need at least 10K rows"


# ============================================================
# STEP 2: PREPARE DATA
# ============================================================
print("\n[2/7] Preparing data...")
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
num_features = X_train.shape[2]

print("\n  Applying log1p to targets...")
y_train_log = np.log1p(y_train)
y_val_log   = np.log1p(y_val)
y_test_log  = np.log1p(y_test)
print(f"  y_train range before log1p: [{y_train.min():.4f}, {y_train.max():.4f}]")
print(f"  y_train range after  log1p: [{y_train_log.min():.4f}, {y_train_log.max():.4f}]")


# ============================================================
# STEP 3: BUILD LSTM MODEL
# ============================================================
print("\n[3/7] Building LSTM model...")
model = build_lstm_model(
    window_size=WINDOW_SIZE,
    num_features=num_features,
    forecast_horizon=FORECAST_HORIZON,
    lstm_units=LSTM_UNITS,
    dropout_rate=DROPOUT_RATE,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)
model.summary()


# ============================================================
# STEP 4: TRAIN LSTM
# ============================================================
print(f"\n[4/7] Training LSTM ({EPOCHS} epochs max)...")
history = train_model(
    model,
    X_train, y_train_log,
    X_val,   y_val_log,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

epochs_run       = len(history.history["loss"])
final_train_loss = history.history["loss"][-1]
best_val_loss    = min(history.history["val_loss"])
print(f"\n  Epochs run:    {epochs_run} / {EPOCHS}")
print(f"  Train loss:    {final_train_loss:.6f}")
print(f"  Best val loss: {best_val_loss:.6f}")


# ============================================================
# STEP 4b: TRAIN XGBOOST
# ============================================================
print("\n[4b/7] Training XGBoost ensemble component...")
xgb_model = train_xgb(
    X_train, y_train_log,
    X_val,   y_val_log
)
save_xgb(xgb_model, "saved_xgb_model.pkl")


# ============================================================
# STEP 5: EVALUATE — LSTM + ENSEMBLE
# ============================================================
print(f"\n[5/7] Evaluating with MC Dropout ({MC_SAMPLES} samples)...")

mc_mean_log, mc_std_log, lower_log, upper_log = mc_dropout_predict(
    model, X_test, n_samples=MC_SAMPLES
)

mc_mean_scaled = inverse_log1p(mc_mean_log)
lower_scaled   = inverse_log1p(lower_log)
upper_scaled   = inverse_log1p(upper_log)

if cpu_scaler is not None:
    mean_inv  = cpu_scaler.inverse_transform(mc_mean_scaled.reshape(-1, 1)).flatten()
    lower_inv = cpu_scaler.inverse_transform(lower_scaled.reshape(-1, 1)).flatten()
    upper_inv = cpu_scaler.inverse_transform(upper_scaled.reshape(-1, 1)).flatten()
    y_true    = cpu_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
else:
    mean_inv  = mc_mean_scaled.flatten()
    lower_inv = lower_scaled.flatten()
    upper_inv = upper_scaled.flatten()
    y_true    = y_test.flatten()

xgb_pred_log    = predict_xgb(xgb_model, X_test)
xgb_pred_scaled = inverse_log1p(xgb_pred_log)
if cpu_scaler is not None:
    xgb_pred_inv = cpu_scaler.inverse_transform(
        xgb_pred_scaled.reshape(-1, 1)
    ).flatten()
else:
    xgb_pred_inv = xgb_pred_scaled.flatten()

final_pred, spike_mask_test = ensemble_predict(mean_inv, xgb_pred_inv, X_test)

lstm_rmse = float(np.sqrt(np.mean((mean_inv - y_true) ** 2)))
lstm_mae  = float(np.mean(np.abs(mean_inv - y_true)))
ens_rmse  = float(np.sqrt(np.mean((final_pred - y_true) ** 2)))
ens_mae   = float(np.mean(np.abs(final_pred - y_true)))

lower_inv = np.clip(lower_inv, 0.0, 1.0)
upper_inv = np.clip(upper_inv, 0.0, 1.0)
coverage  = float(np.mean((y_true >= lower_inv) & (y_true <= upper_inv)))
avg_width = float(np.mean(upper_inv - lower_inv))

mc_std_inv = mc_std_log.flatten()
if cpu_scaler is not None:
    cpu_range  = float(cpu_scaler.data_max_[0] - cpu_scaler.data_min_[0])
    mc_std_inv = mc_std_inv * cpu_range
median_std = float(np.median(mc_std_inv))
data_range = float(y_true.max() - y_true.min()) + 1e-8
confidence = float(np.clip(1.0 - (median_std / (data_range * 0.5)), 0.0, 1.0))

spike_mask  = y_true > (cpu_mean + 2 * cpu_std)
normal_mask = ~spike_mask
spike_rmse  = float(np.sqrt(np.mean((final_pred[spike_mask]  - y_true[spike_mask])  ** 2))) if spike_mask.sum()  > 0 else float("nan")
normal_rmse = float(np.sqrt(np.mean((final_pred[normal_mask] - y_true[normal_mask]) ** 2))) if normal_mask.sum() > 0 else float("nan")

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"  LSTM RMSE:      {lstm_rmse:.6f}")
print(f"  Ensemble RMSE:  {ens_rmse:.6f}   (target: < 0.14)")
print(f"  Coverage 90%:   {coverage:.4f}    (target: 0.70–0.90)")
print(f"  Confidence:     {confidence:.4f}    (target: 0.50–0.70)")
print(f"  Avg CI Width:   {avg_width:.6f}")
print(f"  Spike RMSE:     {spike_rmse:.6f}   (spikes: {spike_mask.sum():,})")
print(f"  Normal RMSE:    {normal_rmse:.6f}")
print()
if ens_rmse < 0.14:
    print("✅ RMSE target met!")
elif ens_rmse < 0.18:
    print(f"🟡 RMSE {ens_rmse:.4f} — close")
else:
    print(f"⚠️  RMSE {ens_rmse:.4f} — check scaler leakage and window size")
if coverage > 0.70:
    print("✅ Coverage target met!")
else:
    print(f"⚠️  Coverage {coverage:.3f} — temperature scaling will fix this next")
if 0.50 <= confidence <= 0.70:
    print("✅ Confidence target met!")
elif confidence > 0.70:
    print(f"🟡 Confidence {confidence:.3f} slightly high")
else:
    print(f"⚠️  Confidence {confidence:.3f} — check dropout rate")
print("=" * 60)


# ============================================================
# STEP 6: TEMPERATURE SCALING CALIBRATION
# ============================================================
# Run on VALIDATION set — not test (prevents leakage)
print("\n[6/7] Temperature scaling calibration (validation set)...")

mc_mean_val_log, mc_std_val_log, _, _ = mc_dropout_predict(
    model, X_val, n_samples=MC_SAMPLES
)

mc_mean_val_scaled = inverse_log1p(mc_mean_val_log)
mc_std_val_scaled  = mc_std_val_log.flatten()

if cpu_scaler is not None:
    mc_mean_val_inv = cpu_scaler.inverse_transform(
        mc_mean_val_scaled.reshape(-1, 1)
    ).flatten()
    mc_std_val_inv = mc_std_val_scaled * float(
        cpu_scaler.data_max_[0] - cpu_scaler.data_min_[0]
    )
    y_val_inv = cpu_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
else:
    mc_mean_val_inv = mc_mean_val_scaled.flatten()
    mc_std_val_inv  = mc_std_val_scaled
    y_val_inv       = y_val.flatten()

T = find_temperature(
    mc_std          = mc_std_val_inv,
    y_true          = y_val_inv,
    mc_mean         = mc_mean_val_inv,
    target_coverage = 0.80
)
save_temperature(T, "calibration_temperature.pkl")


# ============================================================
# STEP 7: SAVE
# ============================================================
print("\n[7/7] Saving...")
joblib.dump(model, "saved_model.pkl")
try:
    model.save("saved_model.keras")
    print("✅ Saved: saved_model.keras + saved_model.pkl")
except Exception as e:
    print(f"⚠️  Keras save: {e} — pkl only")

metrics = {
    "trained_at": datetime.now().isoformat(),
    "config": {
        "window_size":      WINDOW_SIZE,
        "forecast_horizon": FORECAST_HORIZON,
        "epochs_run":       epochs_run,
        "batch_size":       BATCH_SIZE,
        "lstm_units":       LSTM_UNITS,
        "dropout_rate":     DROPOUT_RATE,
        "weight_decay":     WEIGHT_DECAY,
        "learning_rate":    LEARNING_RATE,
        "mc_samples":       MC_SAMPLES,
        "num_features":     num_features,
        "log1p_target":     True,
        "calibration_T":    round(T, 4)
    },
    "metrics": {
        "LSTM_RMSE":     round(lstm_rmse,  6),
        "Ensemble_RMSE": round(ens_rmse,   6),
        "LSTM_MAE":      round(lstm_mae,   6),
        "Ensemble_MAE":  round(ens_mae,    6),
        "Coverage_90":   round(coverage,   4),
        "Confidence":    round(confidence, 4),
        "Avg_CI_Width":  round(avg_width,  6),
        "Spike_RMSE":    round(spike_rmse,  6) if not np.isnan(spike_rmse)  else None,
        "Normal_RMSE":   round(normal_rmse, 6) if not np.isnan(normal_rmse) else None,
        "Best_Val_Loss": round(best_val_loss, 6),
        "Calibration_T": round(T, 4)
    }
}
with open("last_training_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ Metrics saved to last_training_metrics.json")
print("\n🚀 Done! Reload Streamlit app to use the new model.")
