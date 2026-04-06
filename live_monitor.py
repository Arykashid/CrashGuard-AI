"""
Live CPU Monitor Module
Uses psutil to read real system CPU usage in real-time

FIX IN THIS VERSION:
    - build_live_feature_window now returns (1, window_size, 12) features
    - Added lag2 and lag3 to match model training features exactly
    - Added shape validation before returning
    - Added NaN safety check
    - Added fallback if prediction fails

Feature order (must match preprocessing.py exactly):
    0:  cpu_usage
    1:  hour_sin
    2:  hour_cos
    3:  dow_sin
    4:  dow_cos
    5:  lag1
    6:  lag2      ← ADDED (was missing)
    7:  lag3      ← ADDED (was missing)
    8:  lag5
    9:  lag10
    10: roll_mean
    11: roll_std
    Total: 12 features
"""

import psutil
import time
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque


# In-memory buffer for live data
_live_buffer = deque(maxlen=500)

# How many features the model expects
NUM_FEATURES = 12


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
        "cpu_total":          round(psutil.cpu_percent(interval=0.1), 2),
        "cpu_per_core":       cpu_per_core,
        "memory_used_pct":    round(mem.percent, 2),
        "memory_available_gb": round(mem.available / (1024 ** 3), 2),
        "memory_total_gb":    round(mem.total / (1024 ** 3), 2),
        "cpu_core_count":     psutil.cpu_count(logical=True),
        "timestamp":          datetime.now().strftime("%H:%M:%S")
    }


def build_live_feature_window(cpu_history, window_size=60):
    """
    Given a list of recent CPU readings, builds the feature window
    needed for LSTM inference.

    FIX: Now returns shape (1, window_size, 12) — not (1, window_size, 10)
    Added lag2 and lag3 to match model training features exactly.

    Feature order (matches preprocessing.py):
        cpu_usage, hour_sin, hour_cos, dow_sin, dow_cos,
        lag1, lag2, lag3, lag5, lag10, roll_mean, roll_std

    Args:
        cpu_history:  list of recent CPU values (floats 0-1)
        window_size:  number of timesteps (default 60)

    Returns:
        numpy array of shape (1, window_size, 12)
    """
    # ── Pad if not enough history ──────────────────────────────
    if len(cpu_history) < window_size:
        pad = [cpu_history[0]] * (window_size - len(cpu_history))
        cpu_history = pad + list(cpu_history)

    cpu_arr = np.array(cpu_history[-window_size:], dtype=np.float32)

    # ── NaN safety check ──────────────────────────────────────
    if np.any(np.isnan(cpu_arr)):
        cpu_arr = np.nan_to_num(cpu_arr, nan=0.3)

    # ── Build features for each timestep ──────────────────────
    features = []

    for i in range(len(cpu_arr)):
        val = cpu_arr[i]

        # Time features synthesized from buffer position
        hour     = (i // 60) % 24
        dow      = (i // (60 * 24)) % 7
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        dow_sin  = np.sin(2 * np.pi * dow / 7)
        dow_cos  = np.cos(2 * np.pi * dow / 7)

        # Lag features — FIX: added lag2 and lag3
        lag1  = float(cpu_arr[i - 1])  if i >= 1  else val
        lag2  = float(cpu_arr[i - 2])  if i >= 2  else val   # ← NEW
        lag3  = float(cpu_arr[i - 3])  if i >= 3  else val   # ← NEW
        lag5  = float(cpu_arr[i - 5])  if i >= 5  else val
        lag10 = float(cpu_arr[i - 10]) if i >= 10 else val

        # Rolling statistics
        window_slice = cpu_arr[max(0, i - 10):i + 1]
        roll_mean = float(np.mean(window_slice))
        roll_std  = float(np.std(window_slice))

        # Feature vector — 12 features — must match preprocessing.py order
        features.append([
            val,                        # 0: cpu_usage
            hour_sin, hour_cos,         # 1,2: cyclical hour
            dow_sin,  dow_cos,          # 3,4: cyclical day of week
            lag1, lag2, lag3,           # 5,6,7: short-term lags ← lag2,lag3 added
            lag5, lag10,                # 8,9: medium-term lags
            roll_mean, roll_std         # 10,11: rolling stats
        ])

    window = np.array(features, dtype=np.float32)

    # ── Shape validation ──────────────────────────────────────
    expected_shape = (window_size, NUM_FEATURES)
    if window.shape != expected_shape:
        raise ValueError(
            f"live_monitor: Feature window shape mismatch. "
            f"Expected {expected_shape}, got {window.shape}. "
            f"Check that NUM_FEATURES={NUM_FEATURES} matches your trained model."
        )

    # Final NaN check after feature engineering
    if np.any(np.isnan(window)):
        window = np.nan_to_num(window, nan=0.0)

    return window.reshape(1, window_size, NUM_FEATURES)


def safe_live_predict(model, cpu_history, window_size=60):
    """
    Safe wrapper for live prediction.
    Returns (prediction, error_message) tuple.

    If prediction fails for any reason, returns last known CPU value
    as fallback — demo never crashes.

    Args:
        model:        trained Keras model
        cpu_history:  list of recent CPU values
        window_size:  must match training window size

    Returns:
        pred (float):  predicted CPU value 0-1
        error (str):   None if success, error message if fallback used
    """
    last_known = float(cpu_history[-1]) if cpu_history else 0.3

    try:
        # Build feature window
        live_window = build_live_feature_window(cpu_history, window_size)

        # Validate shape before prediction
        expected = (1, window_size, NUM_FEATURES)
        if live_window.shape != expected:
            return last_known, f"Shape mismatch: expected {expected}, got {live_window.shape}"

        # Run prediction
        pred = float(np.clip(
            model.predict(live_window, verbose=0)[0][0],
            0.0, 1.0
        ))
        return pred, None

    except Exception as e:
        # Fallback — return last known value so demo never crashes
        return last_known, str(e)


def mc_dropout_live_predict(model, cpu_history, window_size=60, n_samples=30):
    """
    MC Dropout prediction for live data.
    Runs model n_samples times with dropout ACTIVE.
    Returns mean, std, lower CI, upper CI, confidence.

    This is the CORRECT MC Dropout implementation:
    - Uses model(x, training=True) not model.predict()
    - training=True keeps dropout layers active
    - Multiple passes give uncertainty estimate

    Args:
        model:        trained Keras model with dropout layers
        cpu_history:  list of recent CPU values
        window_size:  must match training window size
        n_samples:    number of MC dropout passes (default 30)

    Returns:
        dict with keys: mean, std, lower, upper, confidence
        Returns safe fallback dict if prediction fails
    """
    last_known = float(cpu_history[-1]) if cpu_history else 0.3

    fallback = {
        "mean":       last_known,
        "std":        0.05,
        "lower":      max(0.0, last_known - 0.10),
        "upper":      min(1.0, last_known + 0.10),
        "confidence": 0.5,
        "error":      None
    }

    try:
        live_window = build_live_feature_window(cpu_history, window_size)

        # MC Dropout — run n_samples passes with dropout ON
        mc_preds = []
        for _ in range(n_samples):
            # training=True keeps dropout layers active
            pred = float(model(live_window, training=True).numpy()[0][0])
            mc_preds.append(pred)

        mc_preds = np.array(mc_preds)
        mean_pred = float(np.mean(mc_preds))
        std_pred  = float(np.std(mc_preds))

        # 95% confidence interval
        lower = float(np.clip(mean_pred - 1.96 * std_pred, 0.0, 1.0))
        upper = float(np.clip(mean_pred + 1.96 * std_pred, 0.0, 1.0))

        # Confidence based on CI width
        # Narrow CI = high confidence, Wide CI = low confidence
        ci_width = upper - lower
        # Map ci_width [0, 1] → confidence [0.95, 0.05]
        confidence = float(np.clip(1.0 - ci_width, 0.05, 0.95))

        return {
            "mean":       float(np.clip(mean_pred, 0.0, 1.0)),
            "std":        std_pred,
            "lower":      lower,
            "upper":      upper,
            "confidence": confidence,
            "error":      None
        }

    except Exception as e:
        fallback["error"] = str(e)
        return fallback