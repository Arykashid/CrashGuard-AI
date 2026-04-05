"""
Error Analysis Module
Analyzes where and why the model fails.
Research-grade diagnostic tool.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ================= SPIKE DETECTION =================
def detect_spikes(y_true, threshold_multiplier=2.0):
    """
    Detects CPU spikes — points where value exceeds
    mean + threshold_multiplier * std.
    Returns boolean mask.
    """
    mean = np.mean(y_true)
    std = np.std(y_true)
    threshold = mean + threshold_multiplier * std
    return y_true > threshold


def detect_drops(y_true, threshold_multiplier=2.0):
    """Detects sudden CPU drops."""
    mean = np.mean(y_true)
    std = np.std(y_true)
    threshold = mean - threshold_multiplier * std
    return y_true < threshold


def detect_bursts(y_true, window=5, burst_factor=1.5):
    """
    Detects workload bursts — rapid consecutive increases.
    Returns boolean mask.
    """
    bursts = np.zeros(len(y_true), dtype=bool)
    for i in range(window, len(y_true)):
        recent_mean = np.mean(y_true[i - window:i])
        if y_true[i] > recent_mean * burst_factor:
            bursts[i] = True
    return bursts


# ================= ERROR SEGMENTATION =================
def segment_errors(y_true, y_pred):
    """
    Segments prediction errors by region type:
    - Normal periods
    - Spike periods
    - Burst periods
    - Drop periods

    Returns dict with per-segment RMSE and MAE.
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    errors = np.abs(y_true - y_pred)
    sq_errors = (y_true - y_pred) ** 2

    spikes = detect_spikes(y_true)
    bursts = detect_bursts(y_true)
    drops = detect_drops(y_true)
    normal = ~(spikes | bursts | drops)

    segments = {
        "Normal": normal,
        "Spike": spikes,
        "Burst": bursts,
        "Drop": drops
    }

    results = {}

    for name, mask in segments.items():
        if mask.sum() > 0:
            rmse = float(np.sqrt(np.mean(sq_errors[mask])))
            mae = float(np.mean(errors[mask]))
            count = int(mask.sum())
            pct = round(float(mask.mean()) * 100, 1)
        else:
            rmse = 0.0
            mae = 0.0
            count = 0
            pct = 0.0

        results[name] = {
            "RMSE": rmse,
            "MAE": mae,
            "Count": count,
            "Pct": pct,
            "mask": mask
        }

    return results


# ================= WORST PREDICTIONS =================
def get_worst_predictions(y_true, y_pred, n=10):
    """Returns indices and values of the n worst predictions."""
    errors = np.abs(y_true - y_pred)
    worst_idx = np.argsort(errors)[-n:][::-1]

    return pd.DataFrame({
        "Index": worst_idx,
        "True Value": y_true[worst_idx].round(6),
        "Predicted": y_pred[worst_idx].round(6),
        "Abs Error": errors[worst_idx].round(6),
        "Region": ["Spike" if detect_spikes(y_true[[i]])[0]
                   else "Burst" if detect_bursts(y_true, window=3)[[i]][0]
                   else "Normal"
                   for i in worst_idx]
    })


# ================= ROLLING ERROR =================
def rolling_error(y_true, y_pred, window=10):
    """Computes rolling RMSE to show error over time."""
    sq_errors = (y_true - y_pred) ** 2
    rolling = pd.Series(sq_errors).rolling(window).mean().apply(np.sqrt)
    return rolling.values


# ================= PLOTLY CHARTS =================
def plot_error_by_region(segment_results):
    """Bar chart: RMSE by region type."""

    regions = [k for k in segment_results if k != "Normal" or True]
    rmse_vals = [segment_results[k]["RMSE"] for k in regions]
    counts = [segment_results[k]["Count"] for k in regions]
    colors = {
        "Normal": "#22c55e",
        "Spike": "#ef4444",
        "Burst": "#f59e0b",
        "Drop": "#3b82f6"
    }

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=regions,
        y=rmse_vals,
        marker_color=[colors.get(r, "#6b7280") for r in regions],
        text=[f"{v:.6f}<br>n={c}" for v, c in zip(rmse_vals, counts)],
        textposition="auto"
    ))
    fig.update_layout(
        title="Prediction Error by Region Type — Where does the model fail?",
        xaxis_title="Region Type",
        yaxis_title="RMSE",
        template="plotly_dark",
        height=380
    )
    return fig


def plot_prediction_vs_actual_with_errors(y_true, y_pred, segment_results, n_points=100):
    """Shows actual vs predicted with error regions highlighted."""

    y_true = y_true[:n_points]
    y_pred = y_pred[:n_points]
    x = list(range(n_points))

    spikes = detect_spikes(y_true)
    bursts = detect_bursts(y_true)

    fig = go.Figure()

    # Actual
    fig.add_trace(go.Scatter(
        x=x, y=y_true,
        name="Actual",
        line=dict(color="white", width=2)
    ))

    # Predicted
    fig.add_trace(go.Scatter(
        x=x, y=y_pred,
        name="Predicted",
        line=dict(color="#ef4444", width=2, dash="dot")
    ))

    # Error fill
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=list(y_true) + list(y_pred[::-1]),
        fill="toself",
        fillcolor="rgba(239,68,68,0.1)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Error Region",
        showlegend=True
    ))

    # Spike markers
    spike_idx = np.where(spikes)[0]
    if len(spike_idx) > 0:
        fig.add_trace(go.Scatter(
            x=spike_idx.tolist(),
            y=y_true[spike_idx].tolist(),
            mode="markers",
            name="CPU Spike",
            marker=dict(color="#fbbf24", size=10, symbol="triangle-up")
        ))

    # Burst markers
    burst_idx = np.where(bursts)[0]
    if len(burst_idx) > 0:
        fig.add_trace(go.Scatter(
            x=burst_idx.tolist(),
            y=y_true[burst_idx].tolist(),
            mode="markers",
            name="Burst",
            marker=dict(color="#f97316", size=8, symbol="diamond")
        ))

    fig.update_layout(
        title="Prediction vs Actual — Error Analysis with Spike/Burst Marking",
        xaxis_title="Time Step",
        yaxis_title="CPU Usage",
        template="plotly_dark",
        height=450
    )
    return fig


def plot_rolling_error(y_true, y_pred):
    """Shows how error evolves over time."""

    rolling = rolling_error(y_true, y_pred, window=10)
    spikes = detect_spikes(y_true)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("Rolling RMSE (window=10)", "CPU Usage with Spike Regions"),
        vertical_spacing=0.12
    )

    fig.add_trace(go.Scatter(
        y=rolling,
        name="Rolling RMSE",
        line=dict(color="#ef4444", width=2)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        y=y_true,
        name="Actual CPU",
        line=dict(color="white", width=1)
    ), row=2, col=1)

    # Shade spike regions
    spike_idx = np.where(spikes)[0]
    for idx in spike_idx:
        fig.add_vrect(
            x0=max(0, idx - 1), x1=min(len(y_true), idx + 1),
            fillcolor="rgba(239,68,68,0.2)",
            line_width=0,
            row=2, col=1
        )

    fig.update_layout(
        template="plotly_dark",
        height=500,
        title="Error Analysis — Rolling RMSE Correlated with CPU Spikes"
    )
    return fig


def plot_error_distribution_by_region(y_true, y_pred, segment_results):
    """Box plots showing error distribution per region."""

    errors = np.abs(y_true - y_pred)
    fig = go.Figure()

    colors = {
        "Normal": "#22c55e",
        "Spike": "#ef4444",
        "Burst": "#f59e0b",
        "Drop": "#3b82f6"
    }

    for region, info in segment_results.items():
        mask = info["mask"]
        if mask.sum() > 1:
            fig.add_trace(go.Box(
                y=errors[mask].tolist(),
                name=f"{region} (n={info['Count']})",
                marker_color=colors.get(region, "#6b7280"),
                boxpoints="outliers"
            ))

    fig.update_layout(
        title="Error Distribution by Region — Spikes cause largest errors",
        yaxis_title="Absolute Error",
        template="plotly_dark",
        height=400
    )
    return fig
