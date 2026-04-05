"""
generate_data.py
Replaces data/cpu_timeseries.csv with 10000 realistic rows.
Runs in 2 seconds. Saves to same path your app.py already reads from.
"""
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os

np.random.seed(42)
n = 15000

t = np.arange(n)

# Base CPU with diurnal pattern (low at night, high during day)
cpu = 0.35 + 0.20 * np.sin(2 * np.pi * t / 1440 - np.pi / 2)

# Weekly pattern
cpu += 0.05 * np.sin(2 * np.pi * t / 10080)

# Autoregressive component (CPU has memory)
ar = np.zeros(n)
for i in range(2, n):
    ar[i] = 0.7 * ar[i-1] + 0.1 * ar[i-2] + np.random.normal(0, 0.01)
cpu += ar

# Add realistic spikes (this is what LSTM learns to predict)
spike_count = 0
for _ in range(240):
    idx    = np.random.randint(100, n - 100)
    height = np.random.uniform(0.50, 0.75)
    width  = np.random.randint(5, 20)
    # Small ramp before spike (makes it predictable from context)
    for j in range(min(8, idx)):
        cpu[idx - j] += height * 0.3 * (1 - j/8)
    # Spike body with exponential decay
    for j in range(min(width, n - idx)):
        cpu[idx + j] += height * np.exp(-j / (width * 0.4))
    spike_count += 1

# Noise
cpu += np.random.normal(0, 0.015, n)
cpu  = np.clip(cpu, 0.01, 0.99).astype("float32")

# Timestamps — 1 second apart (matches your original format)
timestamps = pd.date_range("2024-01-01 00:00:00", periods=n, freq="10s")

df = pd.DataFrame({
    "timestamp": timestamps.strftime("%d-%m-%Y %H:%M:%S"),
    "cpu_usage": cpu
})

os.makedirs("data", exist_ok=True)
df.to_csv("data/cpu_timeseries.csv", index=False)

print(f"Done!")
print(f"Rows:         {len(df)}")
print(f"CPU min:      {cpu.min():.3f}")
print(f"CPU max:      {cpu.max():.3f}")
print(f"CPU mean:     {cpu.mean():.3f}")
print(f"Spikes added: {spike_count}")
print(f"Saved to:     data/cpu_timeseries.csv")
print()
print(df.head())