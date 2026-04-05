"""
Data Generator Module (Research-Grade Infrastructure Simulator)

Generates realistic multivariate server metrics:
CPU, Memory, Disk I/O, Network I/O

Includes:
- Daily & weekly seasonality
- Trend
- Random noise
- Spikes (short bursts)
- Overload regimes
- Correlated system metrics
- Anomaly labels for evaluation
"""

import pandas as pd
import numpy as np


def generate_system_metrics(
    num_samples=10000,
    start_date="2024-01-01",
    freq="1min",
    noise_level=0.1,
    trend=True,
    spike_probability=0.002,
    overload_probability=0.001
):
    timestamps = pd.date_range(start=start_date, periods=num_samples, freq=freq)
    t = np.arange(num_samples)

    # ---------------- Base CPU behaviour ----------------
    daily = 50 + 20 * np.sin(2 * np.pi * t / (24 * 60))
    weekly = 10 * np.sin(2 * np.pi * t / (7 * 24 * 60))

    trend_component = 0.005 * t if trend else 0
    noise = np.random.normal(0, noise_level * 15, num_samples)

    cpu = daily + weekly + trend_component + noise

    anomaly_flag = np.zeros(num_samples)

    # ---------------- Spike events ----------------
    for i in range(num_samples):
        if np.random.rand() < spike_probability:
            duration = np.random.randint(5, 30)
            magnitude = np.random.uniform(20, 50)

            cpu[i:i + duration] += magnitude
            anomaly_flag[i:i + duration] = 1

    # ---------------- Overload regimes ----------------
    i = 0
    while i < num_samples:
        if np.random.rand() < overload_probability:
            duration = np.random.randint(100, 500)
            magnitude = np.random.uniform(30, 60)

            cpu[i:i + duration] += magnitude
            anomaly_flag[i:i + duration] = 1
            i += duration
        else:
            i += 1

    cpu = np.clip(cpu, 0, 100)

    # ---------------- Correlated system metrics ----------------

    memory = cpu * 0.6 + np.random.normal(0, 5, num_samples)
    memory = np.clip(memory, 10, 100)

    disk_io = 50 + cpu * 0.5 + np.random.normal(0, 10, num_samples)
    disk_io = np.clip(disk_io, 0, None)

    network_io = 30 + cpu * 0.8 + np.random.normal(0, 15, num_samples)
    network_io = np.clip(network_io, 0, None)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "cpu_usage": cpu,
        "memory_usage": memory,
        "disk_io": disk_io,
        "network_io": network_io,
        "anomaly_flag": anomaly_flag
    })

    return df


def add_missing_values(df, missing_ratio=0.03):
    df = df.copy()

    n_missing = int(len(df) * missing_ratio)
    idx = np.random.choice(df.index, n_missing, replace=False)

    for col in ["cpu_usage", "memory_usage", "disk_io", "network_io"]:
        df.loc[idx, col] = np.nan

    return df