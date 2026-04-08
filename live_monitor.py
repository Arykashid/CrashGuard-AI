"""
Live CPU Monitor Module — CrashGuard AI
Uses psutil to read real system CPU usage in real-time

UPDATED: N_FEATURES 12 → 15
  Added spike_flag, cpu_diff1, cpu_diff3 to match preprocessing.py

Feature order (MUST match preprocessing.py exactly):
    0:  cpu_usage
    1:  hour_sin
    2:  hour_cos
    3:  dow_sin
    4:  dow_cos
    5:  lag1
    6:  lag5
    7:  lag10
    8:  lag2
    9:  lag3
    10: roll_mean_10
    11: roll_std_10
    12: spike_flag      NEW
    13: cpu_diff1       NEW
    14: cpu_diff3       NEW

Shape contract:
    build_live_feature_window() returns (window_size, N_FEATURES) — NO batch dim.
    app.py adds batch dim: raw.reshape(1, window_size, N_FEATURES)
"""

import psutil
import time
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque

_live_buffer = deque(maxlen=500)
N_FEATURES = 15   # MUST match preprocessing.py


def get_current_cpu():
    return psutil.cpu_percent(interval=0.1) / 100.0


def get_cpu_snapshot(n_samples=60, interval=0.5):
    records = []
    for _ in range(n_samples):
        records.append({"timestamp": datetime.now(), "cpu_usage": get_current_cpu()})
        time.sleep(interval)
    return pd.DataFrame(records)


def get_system_stats():
    mem = psutil.virtual_memory()
    return {
        "cpu_total":           round(psutil.cpu_percent(interval=0.1), 2),
        "cpu_per_core":        psutil.cpu_percent(percpu=True),
        "memory_used_pct":     round(mem.percent, 2),
        "memory_available_gb": round(mem.available / (1024 ** 3), 2),
        "memory_total_gb":     round(mem.total / (1024 ** 3), 2),
        "cpu_core_count":      psutil.cpu_count(logical=True),
        "timestamp":           datetime.now().strftime("%H:%M:%S")
    }


def build_live_feature_window(cpu_history, window_size=60):
    """
    Build feature window for live LSTM inference.
    Returns (window_size, 15) — NO batch dim.
    app.py does: X = raw.reshape(1, window_size, N_FEATURES)
    """
    if len(cpu_history) < window_size:
        pad = [cpu_history[0]] * (window_size - len(cpu_history))
        cpu_history = pad + list(cpu_history)

    cpu_arr = np.array(cpu_history[-window_size:], dtype=np.float32)
    if np.any(np.isnan(cpu_arr)):
        cpu_arr = np.nan_to_num(cpu_arr, nan=0.3)

    features = []
    for i in range(len(cpu_arr)):
        val = cpu_arr[i]

        hour     = (i // 60) % 24
        dow      = (i // (60 * 24)) % 7
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        dow_sin  = np.sin(2 * np.pi * dow  / 7)
        dow_cos  = np.cos(2 * np.pi * dow  / 7)

        lag1  = float(cpu_arr[i - 1])  if i >= 1  else val
        lag2  = float(cpu_arr[i - 2])  if i >= 2  else val
        lag3  = float(cpu_arr[i - 3])  if i >= 3  else val
        lag5  = float(cpu_arr[i - 5])  if i >= 5  else val
        lag10 = float(cpu_arr[i - 10]) if i >= 10 else val

        roll10       = cpu_arr[max(0, i - 9): i + 1]
        roll_mean_10 = float(np.mean(roll10))
        roll_std_10  = float(np.std(roll10)) if len(roll10) > 1 else 0.0

        roll20     = cpu_arr[max(0, i - 19): i + 1]
        rm20       = float(np.mean(roll20))
        rs20       = float(np.std(roll20)) if len(roll20) > 1 else 0.0
        spike_flag = 1.0 if (val > rm20 + 2.0 * rs20) else 0.0

        cpu_diff1 = float(cpu_arr[i] - cpu_arr[i - 1]) if i >= 1 else 0.0
        cpu_diff3 = float(cpu_arr[i] - cpu_arr[i - 3]) if i >= 3 else 0.0

        features.append([
            val,            # 0  cpu_usage
            hour_sin,       # 1  hour_sin
            hour_cos,       # 2  hour_cos
            dow_sin,        # 3  dow_sin
            dow_cos,        # 4  dow_cos
            lag1,           # 5  lag1
            lag5,           # 6  lag5
            lag10,          # 7  lag10
            lag2,           # 8  lag2
            lag3,           # 9  lag3
            roll_mean_10,   # 10 roll_mean_10
            roll_std_10,    # 11 roll_std_10
            spike_flag,     # 12 spike_flag  NEW
            cpu_diff1,      # 13 cpu_diff1   NEW
            cpu_diff3,      # 14 cpu_diff3   NEW
        ])

    window_array = np.array(features, dtype=np.float32)
    expected     = (window_size, N_FEATURES)
    if window_array.shape != expected:
        raise ValueError(
            f"Shape mismatch: expected {expected}, got {window_array.shape}"
        )
    return window_array   # (window_size, 15) — app.py adds batch dim


def safe_predict(model, live_window, fallback_value=0.3):
    try:
        if live_window.ndim != 3:
            raise ValueError(f"Expected 3D, got {live_window.ndim}D")
        if live_window.shape[2] != N_FEATURES:
            raise ValueError(f"Expected {N_FEATURES} features, got {live_window.shape[2]}")
        if np.any(np.isnan(live_window)):
            live_window = np.nan_to_num(live_window, nan=0.3)
        return float(np.clip(model.predict(live_window, verbose=0)[0][0], 0.0, 1.0))
    except Exception as e:
        print(f"  [live_monitor] Prediction failed: {e}")
        return float(fallback_value)


def mc_dropout_live(model, live_window, n_samples=30):
    """
    MC Dropout for live prediction.
    live_window: (1, window_size, N_FEATURES) — caller must reshape first.
    """
    try:
        preds      = np.array([
            float(model(live_window, training=True).numpy()[0][0])
            for _ in range(n_samples)
        ])
        mean_pred  = float(np.mean(preds))
        std_pred   = float(np.std(preds))
        lower_ci   = float(np.clip(mean_pred - 1.96 * std_pred, 0.0, 1.0))
        upper_ci   = float(np.clip(mean_pred + 1.96 * std_pred, 0.0, 1.0))
        confidence = float(np.clip(1.0 - (upper_ci - lower_ci), 0.10, 0.95))
        return mean_pred, confidence, std_pred, lower_ci, upper_ci
    except Exception as e:
        print(f"  [live_monitor] MC Dropout failed: {e}")
        return 0.3, 0.5, 0.0, 0.25, 0.35
