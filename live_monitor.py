"""
Live CPU Monitor Module
Uses psutil to read real system CPU usage in real-time

FIX APPLIED:
    - build_live_feature_window now produces 12 features (was 10)
    - Added lag2, lag3 to match model training feature set
    - Added shape validation before returning
    - Added NaN safety check
    - Added safe_predict() fallback wrapper
    - Added mc_dropout_live() for proper uncertainty

Feature order (MUST match preprocessing.py exactly):
    0:  cpu_usage
    1:  hour_sin
    2:  hour_cos
    3:  dow_sin
    4:  dow_cos
    5:  lag1
    6:  lag2      <- ADDED (was missing, caused crash)
    7:  lag3      <- ADDED (was missing, caused crash)
    8:  lag5
    9:  lag10
    10: roll_mean
    11: roll_std
"""

import psutil
import time
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque


# ── In-memory buffer for live data ──────────────────────────────
_live_buffer = deque(maxlen=500)

# ── Number of features — MUST match model training ──────────────
# If your model was trained on 10 features, change this to 10
N_FEATURES = 12


def get_current_cpu():
    """Returns current CPU usage as a float between 0 and 1."""
    return psutil.cpu_percent(interval=0.1) / 100.0


def get_cpu_snapshot(n_samples=60, interval=0.5):
    """
    Collects n_samples of CPU readings spaced by interval seconds.
    Returns a DataFrame with timestamp and cpu_usage columns.
    """
    records = []
    for _ in range(n_samples):
        usage = get_current_cpu()
        ts = datetime.now()
        records.append({"timestamp": ts, "cpu_usage": usage})
        time.sleep(interval)
    return pd.DataFrame(records)


def get_system_stats():
    """Returns a dict with current system resource stats."""
    mem = psutil.virtual_memory()
    cpu_per_core = psutil.cpu_percent(percpu=True)
    return {
        "cpu_total":           round(psutil.cpu_percent(interval=0.1), 2),
        "cpu_per_core":        cpu_per_core,
        "memory_used_pct":     round(mem.percent, 2),
        "memory_available_gb": round(mem.available / (1024 ** 3), 2),
        "memory_total_gb":     round(mem.total / (1024 ** 3), 2),
        "cpu_core_count":      psutil.cpu_count(logical=True),
        "timestamp":           datetime.now().strftime("%H:%M:%S")
    }


def build_live_feature_window(cpu_history, window_size=60):
    """
    Given a list of recent CPU readings, builds the feature window
    needed for LSTM inference.

    FIX: Now produces 12 features to match model training.
    Added lag2, lag3 which were missing — caused shape mismatch crash:
        Matrix size-incompatible: In[0]: [1,10], In[1]: [12,256]

    Returns numpy array of shape (1, window_size, 12)

    Feature order (must match preprocessing.py):
        cpu_usage, hour_sin, hour_cos, dow_sin, dow_cos,
        lag1, lag2, lag3, lag5, lag10, roll_mean, roll_std
    """
    # ── Pad if not enough history ────────────────────────────────
    if len(cpu_history) < window_size:
        pad = [cpu_history[0]] * (window_size - len(cpu_history))
        cpu_history = pad + list(cpu_history)

    cpu_arr = np.array(cpu_history[-window_size:], dtype=np.float32)

    # ── NaN safety check ─────────────────────────────────────────
    if np.any(np.isnan(cpu_arr)):
        cpu_arr = np.nan_to_num(cpu_arr, nan=0.3)

    features = []

    for i in range(len(cpu_arr)):
        val = cpu_arr[i]

        # Time features (synthesized from position in window)
        hour = (i // 60) % 24
        dow  = (i // (60 * 24)) % 7

        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        dow_sin  = np.sin(2 * np.pi * dow / 7)
        dow_cos  = np.cos(2 * np.pi * dow / 7)

        # Lag features — FIX: added lag2 and lag3
        lag1  = float(cpu_arr[i - 1])  if i >= 1  else val
        lag2  = float(cpu_arr[i - 2])  if i >= 2  else val   # ADDED
        lag3  = float(cpu_arr[i - 3])  if i >= 3  else val   # ADDED
        lag5  = float(cpu_arr[i - 5])  if i >= 5  else val
        lag10 = float(cpu_arr[i - 10]) if i >= 10 else val

        # Rolling statistics
        window_slice = cpu_arr[max(0, i - 10):i + 1]
        roll_mean = float(np.mean(window_slice))
        roll_std  = float(np.std(window_slice))

        # 12 features in correct order
        features.append([
            val,                    # 0:  cpu_usage
            hour_sin, hour_cos,     # 1,2: time encoding
            dow_sin,  dow_cos,      # 3,4: day encoding
            lag1,                   # 5:  lag1
            lag2,                   # 6:  lag2  <- ADDED
            lag3,                   # 7:  lag3  <- ADDED
            lag5,                   # 8:  lag5
            lag10,                  # 9:  lag10
            roll_mean,              # 10: rolling mean
            roll_std                # 11: rolling std
        ])

    window_array = np.array(features, dtype=np.float32)

    # ── Shape validation ─────────────────────────────────────────
    expected_shape = (window_size, N_FEATURES)
    if window_array.shape != expected_shape:
        raise ValueError(
            f"live_monitor: Feature window shape mismatch. "
            f"Expected {expected_shape}, got {window_array.shape}. "
            f"Check N_FEATURES={N_FEATURES} matches your model training."
        )

    return window_array.reshape(1, window_size, N_FEATURES)


def safe_predict(model, live_window, fallback_value=0.3):
    """
    Safe wrapper around model.predict with shape validation and fallback.
    If prediction fails for any reason, returns fallback_value instead of crashing.

    Args:
        model:          trained Keras model
        live_window:    output of build_live_feature_window -> (1, 60, 12)
        fallback_value: value to return if prediction fails

    Returns:
        float: predicted CPU value clipped to [0, 1]
    """
    try:
        # Validate dimensions
        if live_window.ndim != 3:
            raise ValueError(f"Expected 3D array, got {live_window.ndim}D")

        batch, timesteps, features = live_window.shape

        if features != N_FEATURES:
            raise ValueError(
                f"Model expects {N_FEATURES} features, "
                f"live window has {features}. "
                f"Retrain model or fix build_live_feature_window."
            )

        # NaN check
        if np.any(np.isnan(live_window)):
            live_window = np.nan_to_num(live_window, nan=0.3)

        pred = model.predict(live_window, verbose=0)[0][0]
        return float(np.clip(pred, 0.0, 1.0))

    except Exception as e:
        print(f"  [live_monitor] Prediction failed: {e}")
        print(f"  [live_monitor] Using fallback value: {fallback_value}")
        return float(fallback_value)


def mc_dropout_live(model, live_window, n_samples=30):
    """
    MC Dropout uncertainty estimation for live prediction.

    Runs model n_samples times with dropout ACTIVE (training=True).
    Returns mean prediction and confidence score.

    IMPORTANT: Uses model(x, training=True) NOT model.predict()
    This keeps dropout ON during inference — that is the whole point of MC Dropout.

    Args:
        model:       trained Keras model with dropout layers
        live_window: output of build_live_feature_window -> (1, 60, 12)
        n_samples:   number of MC forward passes (default 30)

    Returns:
        mean_pred:  float, mean prediction across samples
        confidence: float in [0.1, 0.95], higher = more confident
        std_pred:   float, standard deviation across samples
        lower_ci:   float, lower bound of 95% CI
        upper_ci:   float, upper bound of 95% CI
    """
    try:
        # Run n_samples forward passes with dropout ON
        preds = np.array([
            float(model(live_window, training=True).numpy()[0][0])
            for _ in range(n_samples)
        ])

        mean_pred = float(np.mean(preds))
        std_pred  = float(np.std(preds))

        # 95% confidence interval
        lower_ci = float(np.clip(mean_pred - 1.96 * std_pred, 0.0, 1.0))
        upper_ci = float(np.clip(mean_pred + 1.96 * std_pred, 0.0, 1.0))

        # Confidence score
        # Narrower CI = higher confidence
        # CI width 0.0 -> confidence 0.95
        # CI width 1.0 -> confidence 0.10
        ci_width   = upper_ci - lower_ci
        confidence = float(np.clip(1.0 - ci_width, 0.10, 0.95))

        return mean_pred, confidence, std_pred, lower_ci, upper_ci

    except Exception as e:
        print(f"  [live_monitor] MC Dropout failed: {e}")
        fallback = 0.3
        return fallback, 0.5, 0.0, fallback - 0.05, fallback + 0.05