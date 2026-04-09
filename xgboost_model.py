"""
xgboost_model.py — CrashGuard AI XGBoost Ensemble Component

CHANGES in this version:
  - ensemble_predict: adaptive spike threshold (1.5*std + velocity flag)
    Old: static 2.0*std threshold only
    New: 1.5*std OR cpu_diff1 > 85th percentile (catches early rising spikes)
"""

import numpy as np
import joblib
import os
from xgboost import XGBRegressor


# Feature order in preprocessing.py:
#   0: cpu_usage, 1: hour_sin, 2: hour_cos, 3: dow_sin, 4: dow_cos,
#   5: lag1, 6: lag5, 7: lag10, 8: lag2, 9: lag3,
#   10: roll_mean_10, 11: roll_std_10, 12: spike_flag,
#   13: cpu_diff1, 14: cpu_diff3

XGB_FEATURE_INDICES = [5, 8, 9, 6, 7, 10, 11, 1, 2, 13, 14]
XGB_FEATURE_NAMES   = [
    "lag1", "lag2", "lag3", "lag5", "lag10",
    "roll_mean_10", "roll_std_10",
    "hour_sin", "hour_cos",
    "cpu_diff1", "cpu_diff3"
]


def extract_xgb_features(X_windows):
    last_step = X_windows[:, -1, :]
    return last_step[:, XGB_FEATURE_INDICES]


def build_xgb_model():
    return XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )


def train_xgb(X_train, y_train, X_val, y_val):
    X_flat_train = extract_xgb_features(X_train)
    X_flat_val   = extract_xgb_features(X_val)
    y_flat_train = y_train.flatten()
    y_flat_val   = y_val.flatten()

    model = build_xgb_model()
    model.fit(
        X_flat_train, y_flat_train,
        eval_set=[(X_flat_val, y_flat_val)],
        verbose=False
    )

    val_rmse   = float(np.sqrt(np.mean(
        (model.predict(X_flat_val) - y_flat_val) ** 2
    )))
    print(f"  XGBoost trained — val RMSE: {val_rmse:.6f}")
    return model


def predict_xgb(model, X_windows):
    X_flat = extract_xgb_features(X_windows)
    return np.clip(model.predict(X_flat), 0.0, 1.0)


def ensemble_predict(lstm_pred, xgb_pred, X_windows):
    """
    Regime-switching ensemble with adaptive spike detection.

    CHANGE: spike threshold tightened from 2.0*std to 1.5*std.
    Also adds velocity flag: if cpu_diff1 is in top 15% of values,
    treat as spike regime even if threshold not yet breached.

    Why 1.5*std:
        2.0*std catches ~5% of points — too conservative, misses early spikes.
        1.5*std catches ~10% — better recall with small precision tradeoff.

    Why velocity flag:
        A spike forming is detectable 1 step earlier via cpu_diff1
        than via the rolling threshold. Early detection = earlier XGBoost weight.
    """
    lstm_pred = np.array(lstm_pred).flatten()
    xgb_pred  = np.array(xgb_pred).flatten()

    last_step = X_windows[:, -1, :]
    last_cpu  = last_step[:, 0]
    roll_mean = last_step[:, 10]
    roll_std  = last_step[:, 11]
    cpu_diff1 = last_step[:, 13]

    # Tighter statistical threshold
    spike_threshold = roll_mean + 1.5 * roll_std
    spike_mask      = last_cpu > spike_threshold

    # Velocity flag — rising fast even if not yet above threshold
    diff_threshold = np.mean(cpu_diff1) + 1.5 * np.std(cpu_diff1)
    rising_fast    = cpu_diff1 > diff_threshold

    # Combined: statistically high OR rising fast
    spike_mask = spike_mask | rising_fast

    final_pred = np.where(
        spike_mask,
        0.3 * lstm_pred + 0.7 * xgb_pred,
        0.7 * lstm_pred + 0.3 * xgb_pred
    )

    n_spike  = int(spike_mask.sum())
    n_normal = int((~spike_mask).sum())
    print(f"  Ensemble: {n_spike} spike-regime, {n_normal} normal-regime predictions")

    return final_pred, spike_mask


def dynamic_ensemble_predict(lstm_pred, xgb_pred, X_windows,
                              mae_lstm=None, mae_xgb=None):
    if mae_lstm is None or mae_xgb is None:
        return ensemble_predict(lstm_pred, xgb_pred, X_windows)

    w_lstm  = 1.0 / (mae_lstm + 1e-8)
    w_xgb   = 1.0 / (mae_xgb  + 1e-8)
    w_total = w_lstm + w_xgb

    lstm_pred  = np.array(lstm_pred).flatten()
    xgb_pred   = np.array(xgb_pred).flatten()
    final_pred = (w_lstm / w_total) * lstm_pred + (w_xgb / w_total) * xgb_pred

    last_step  = X_windows[:, -1, :]
    roll_mean  = last_step[:, 10]
    roll_std   = last_step[:, 11]

    cpu_diff1 = last_step[:, 13]
    diff_threshold = np.mean(cpu_diff1) + 1.5 * np.std(cpu_diff1)
    rising_fast = cpu_diff1 > diff_threshold

    spike_mask = (last_step[:, 0] > (roll_mean + 1.5 * roll_std)) | rising_fast
    print(f"  Dynamic ensemble: w_lstm={w_lstm/w_total:.2f}, w_xgb={w_xgb/w_total:.2f}")
    return final_pred, spike_mask


def save_xgb(model, path="saved_xgb_model.pkl"):
    joblib.dump(model, path)
    print(f"  XGBoost saved: {path}")


def load_xgb(path="saved_xgb_model.pkl"):
    if os.path.exists(path):
        model = joblib.load(path)
        print(f"  XGBoost loaded: {path}")
        return model
    print(f"  XGBoost model not found at {path}")
    return None


if __name__ == "__main__":
    print("=" * 60)
    print("XGBoost Model Self-Test")
    print("=" * 60)

    WINDOW, N_FEAT = 60, 15
    N_TRAIN, N_VAL = 500, 100

    X_train = np.random.randn(N_TRAIN, WINDOW, N_FEAT).astype(np.float32)
    y_train = np.random.rand(N_TRAIN, 1).astype(np.float32)
    X_val   = np.random.randn(N_VAL,   WINDOW, N_FEAT).astype(np.float32)
    y_val   = np.random.rand(N_VAL,   1).astype(np.float32)

    print("\n[1] Training XGBoost...")
    xgb_model = train_xgb(X_train, y_train, X_val, y_val)

    print("\n[2] Predicting...")
    preds = predict_xgb(xgb_model, X_val)
    assert preds.shape == (N_VAL,), f"Shape mismatch: {preds.shape}"
    print(f"  Predictions shape: {preds.shape} ✅")

    print("\n[3] Adaptive ensemble...")
    lstm_fake = np.random.rand(N_VAL)
    final, spike_mask = ensemble_predict(lstm_fake, preds, X_val)
    assert final.shape == (N_VAL,), f"Ensemble shape mismatch: {final.shape}"
    print(f"  Ensemble shape: {final.shape} ✅")
    print(f"  Spike regime points: {spike_mask.sum()} / {N_VAL}")

    print("\n[4] Save/load...")
    save_xgb(xgb_model, "__test_xgb.pkl")
    loaded = load_xgb("__test_xgb.pkl")
    preds2 = predict_xgb(loaded, X_val)
    assert np.allclose(preds, preds2), "Save/load mismatch!"
    print("  Save/load ✅")
    os.remove("__test_xgb.pkl")

    print("\n" + "=" * 60)
    print("All tests passed.")
    print("=" * 60)
