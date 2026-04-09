"""
xgboost_model.py — CrashGuard AI XGBoost Ensemble Component

PURPOSE:
    XGBoost trained on flattened last-timestep features from each walk-forward fold.
    Combined with LSTM in a regime-switching ensemble:
      - Spike regime  (cpu > rolling_mean + 2*std): 0.3 LSTM + 0.7 XGBoost
      - Normal regime (everything else):             0.7 LSTM + 0.3 XGBoost

WHY XGBoost HELPS:
    LSTM sees sequences and learns temporal patterns.
    XGBoost sees the last timestep as flat features and learns
    non-linear feature interactions (e.g. lag1 × spike_flag).
    Together they cover both temporal and feature-interaction dimensions.

LEAKAGE PREVENTION:
    Rolling stats used as XGBoost features are computed from
    training-fold history only. No future data leaks in.

FEATURES USED (11 — last timestep only):
    lag1, lag2, lag3, lag5, lag10,
    roll_mean_10, roll_std_10,
    hour_sin, hour_cos,
    cpu_diff1, cpu_diff3

NOTE: Do NOT feed full sequence (60 × 15 = 900 cols) — XGBoost
    has no concept of sequence order and will overfit badly.
"""

import numpy as np
import joblib
import os
from xgboost import XGBRegressor


# =============================================
# FEATURE COLUMN INDICES
# (must match preprocessing.py feature_cols order)
# =============================================
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


# =============================================
# EXTRACT LAST-TIMESTEP FEATURES
# =============================================
def extract_xgb_features(X_windows):
    """
    Extract last-timestep flat features from LSTM windows.

    Args:
        X_windows: shape (N, window_size, n_features)

    Returns:
        X_flat: shape (N, 11) — last timestep only, selected features
    """
    last_step = X_windows[:, -1, :]              # (N, n_features)
    return last_step[:, XGB_FEATURE_INDICES]      # (N, 11)


# =============================================
# BUILD XGBOOST MODEL
# =============================================
def build_xgb_model():
    """
    XGBoost regressor with settings tuned for CPU time-series.

    n_estimators=400:   enough capacity, early stopping cuts as needed
    max_depth=5:        prevents overfitting on 11 flat features
    learning_rate=0.05: slower learning → better generalisation
    subsample=0.8:      row sampling → reduces variance
    colsample_bytree=0.8: feature sampling → reduces correlation between trees
    """
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


# =============================================
# TRAIN XGBOOST (WALK-FORWARD SAFE)
# =============================================
def train_xgb(X_train, y_train, X_val, y_val):
    """
    Train XGBoost on last-timestep features with early stopping.

    Leakage-safe because:
    - X_train and X_val come from walk-forward split in train.py
    - Rolling stats inside X_train windows were computed from
      training history only (enforced in preprocessing.py)
    - We only use the last timestep — no sequence leakage possible

    Args:
        X_train: (N_train, window, n_features) — LSTM-format windows
        y_train: (N_train, 1) — targets (log1p scaled)
        X_val:   (N_val, window, n_features)
        y_val:   (N_val, 1)

    Returns:
        Trained XGBRegressor
    """
    X_flat_train = extract_xgb_features(X_train)
    X_flat_val   = extract_xgb_features(X_val)
    y_flat_train = y_train.flatten()
    y_flat_val   = y_val.flatten()

    model = build_xgb_model()
    model.fit(
        X_flat_train, y_flat_train,
        eval_set=[(X_flat_val, y_flat_val)],
        early_stopping_rounds=30,
        verbose=False
    )

    best_round = model.best_iteration
    val_rmse   = float(np.sqrt(np.mean(
        (model.predict(X_flat_val) - y_flat_val) ** 2
    )))
    print(f"  XGBoost trained — best round: {best_round}, val RMSE: {val_rmse:.6f}")
    return model


# =============================================
# XGBOOST PREDICT
# =============================================
def predict_xgb(model, X_windows):
    """
    Predict using XGBoost on last-timestep features.

    Args:
        model:     trained XGBRegressor
        X_windows: (N, window, n_features)

    Returns:
        preds: (N,) flat array of predictions
    """
    X_flat = extract_xgb_features(X_windows)
    return model.predict(X_flat)


# =============================================
# REGIME-SWITCHING ENSEMBLE
# =============================================
def ensemble_predict(lstm_pred, xgb_pred, X_windows):
    """
    Regime-switching ensemble.

    Spike detection uses the last-timestep rolling_mean and rolling_std
    from inside the window — no external data needed.

    Spike regime  (last cpu > rolling_mean + 2*std): 0.3 LSTM / 0.7 XGBoost
    Normal regime (everything else):                  0.7 LSTM / 0.3 XGBoost

    WHY REGIME-SWITCHING BEATS FIXED WEIGHTS:
        LSTM is better at smooth temporal patterns (normal regime).
        XGBoost is better at sudden feature-driven jumps (spike regime).
        Fixed 0.6/0.4 blend is a compromise that's suboptimal for both.
        Switching gives each model its strongest regime.

    Args:
        lstm_pred:  (N,) LSTM predictions (already inverse-transformed)
        xgb_pred:   (N,) XGBoost predictions (already inverse-transformed)
        X_windows:  (N, window, n_features) — for spike detection

    Returns:
        final_pred: (N,) blended predictions
        spike_mask: (N,) bool — True where spike regime was detected
    """
    lstm_pred = np.array(lstm_pred).flatten()
    xgb_pred  = np.array(xgb_pred).flatten()

    # Extract spike detection signals from last timestep
    # col 10 = roll_mean_10, col 11 = roll_std_10, col 0 = cpu_usage
    last_step      = X_windows[:, -1, :]
    last_cpu       = last_step[:, 0]
    roll_mean      = last_step[:, 10]
    roll_std       = last_step[:, 11]

    spike_threshold = roll_mean + 2.0 * roll_std
    spike_mask      = last_cpu > spike_threshold

    # Spike regime: trust XGBoost more (better at sudden jumps)
    # Normal regime: trust LSTM more (better at smooth patterns)
    final_pred = np.where(
        spike_mask,
        0.3 * lstm_pred + 0.7 * xgb_pred,   # spike
        0.7 * lstm_pred + 0.3 * xgb_pred    # normal
    )

    n_spike  = int(spike_mask.sum())
    n_normal = int((~spike_mask).sum())
    print(f"  Ensemble: {n_spike} spike-regime, {n_normal} normal-regime predictions")

    return final_pred, spike_mask


# =============================================
# DYNAMIC WEIGHT ENSEMBLE (alternative)
# =============================================
def dynamic_ensemble_predict(lstm_pred, xgb_pred, X_windows,
                              mae_lstm=None, mae_xgb=None):
    """
    Dynamic ensemble — weights inversely proportional to recent MAE.

    Use this instead of regime_switching if you have per-fold MAE
    tracked during walk-forward validation.

    If MAE values not provided, falls back to regime-switching.

    Args:
        lstm_pred:  (N,) predictions
        xgb_pred:   (N,) predictions
        X_windows:  (N, window, n_features)
        mae_lstm:   float — recent LSTM MAE on validation fold
        mae_xgb:    float — recent XGBoost MAE on validation fold

    Returns:
        final_pred: (N,)
        spike_mask: (N,) bool
    """
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
    spike_mask = last_step[:, 0] > (roll_mean + 2.0 * roll_std)

    print(f"  Dynamic ensemble: w_lstm={w_lstm/w_total:.2f}, w_xgb={w_xgb/w_total:.2f}")
    return final_pred, spike_mask


# =============================================
# SAVE / LOAD
# =============================================
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


# =============================================
# SELF-TEST
# =============================================
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

    print("\n[3] Regime-switching ensemble...")
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
    print("All XGBoost tests passed.")
    print("=" * 60)
