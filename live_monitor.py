"""
Live CPU Monitor Module
Uses psutil to read real system CPU usage in real-time
"""

import psutil
import time
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque


# In-memory buffer for live data
_live_buffer = deque(maxlen=500)


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
        "cpu_total": round(psutil.cpu_percent(interval=0.1), 2),
        "cpu_per_core": cpu_per_core,
        "memory_used_pct": round(mem.percent, 2),
        "memory_available_gb": round(mem.available / (1024 ** 3), 2),
        "memory_total_gb": round(mem.total / (1024 ** 3), 2),
        "cpu_core_count": psutil.cpu_count(logical=True),
        "timestamp": datetime.now().strftime("%H:%M:%S")
    }


def build_live_feature_window(cpu_history, window_size=60):
    """
    Given a list of recent CPU readings, builds the feature window
    needed for LSTM inference.

    Returns numpy array of shape (1, window_size, 10)
    """
    if len(cpu_history) < window_size:
        # Pad with first value if not enough history
        pad = [cpu_history[0]] * (window_size - len(cpu_history))
        cpu_history = pad + list(cpu_history)

    cpu_arr = np.array(cpu_history[-window_size:])

    features = []

    for i, val in enumerate(cpu_arr):
        # Synthesize time features from position
        hour = (i // 60) % 24
        dow = (i // (60 * 24)) % 7

        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        dow_sin = np.sin(2 * np.pi * dow / 7)
        dow_cos = np.cos(2 * np.pi * dow / 7)

        lag1 = cpu_arr[i - 1] if i > 0 else val
        lag5 = cpu_arr[i - 5] if i > 4 else val
        lag10 = cpu_arr[i - 10] if i > 9 else val

        roll_mean = np.mean(cpu_arr[max(0, i - 10):i + 1])
        roll_std = np.std(cpu_arr[max(0, i - 10):i + 1])

        features.append([
            val,
            hour_sin, hour_cos,
            dow_sin, dow_cos,
            lag1, lag5, lag10,
            roll_mean, roll_std
        ])

    return np.array(features).reshape(1, window_size, 10)
