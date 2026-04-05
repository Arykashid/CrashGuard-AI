"""
Anomaly Detection Module
Uses Isolation Forest + Z-score for CPU spike detection
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies_isolation_forest(cpu_series, contamination=0.05):
    values = np.array(cpu_series).reshape(-1, 1)
    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    )
    labels = model.fit_predict(values)
    return labels == -1


def detect_anomalies_zscore(cpu_series, threshold=3.0):
    values = np.array(cpu_series)
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return np.zeros(len(values), dtype=bool)
    z_scores = np.abs((values - mean) / std)
    return z_scores > threshold


def get_anomaly_summary(cpu_series, timestamps=None):
    iso_anomalies = detect_anomalies_isolation_forest(cpu_series)
    z_anomalies = detect_anomalies_zscore(cpu_series)
    combined = iso_anomalies | z_anomalies
    anomaly_indices = np.where(combined)[0]
    anomaly_values = np.array(cpu_series)[anomaly_indices]
    return {
        "total_points": len(cpu_series),
        "anomaly_count": int(combined.sum()),
        "anomaly_pct": round(float(combined.mean()) * 100, 2),
        "anomaly_indices": anomaly_indices.tolist(),
        "anomaly_values": anomaly_values.tolist(),
        "isolation_forest_flags": int(iso_anomalies.sum()),
        "zscore_flags": int(z_anomalies.sum()),
        "combined_flags": combined
    }