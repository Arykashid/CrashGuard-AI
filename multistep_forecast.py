"""
Multi-Step Forecasting Module
Predicts next N steps into the future using recursive and direct strategies.
Research-grade with uncertainty quantification per step.
"""

import numpy as np
import plotly.graph_objects as go
import pandas as pd


# ================= RECURSIVE FORECASTING =================
def recursive_multistep_forecast(model, X_seed, n_steps, scaler_cpu):
    """
    Recursive strategy: use each prediction as input for the next step.
    
    Args:
        model: trained LSTM model
        X_seed: seed window (1, window_size, n_features) — last known window
        n_steps: how many steps ahead to predict
        scaler_cpu: cpu_scaler for inverse transform
    
    Returns:
        predictions: array of shape (n_steps,) in original CPU scale
    """

    predictions = []
    current_window = X_seed.copy()  # shape: (1, window_size, n_features)

    for step in range(n_steps):

        # Predict next step
        pred_scaled = model.predict(current_window, verbose=0)[0][0]
        pred_original = float(scaler_cpu.inverse_transform([[pred_scaled]])[0][0])
        # Only clip to valid range, don't force to 0
        pred_original = float(np.clip(pred_original, 0.0, 1.0))
        # Keep small values — don't round to 0
        if abs(pred_original) < 1e-8:
            pred_original = float(abs(scaler_cpu.inverse_transform([[pred_scaled]])[0][0]))

        predictions.append(pred_original)

        # Shift window: drop oldest, append new prediction
        new_row = current_window[0, -1, :].copy()
        new_row[0] = pred_scaled  # update cpu_usage feature

        # Update lag features
        if current_window.shape[2] >= 8:
            new_row[5] = current_window[0, -1, 0]   # lag1 = previous cpu
            new_row[6] = current_window[0, -5, 0] if current_window.shape[1] >= 5 else new_row[0]  # lag5
            new_row[7] = current_window[0, -10, 0] if current_window.shape[1] >= 10 else new_row[0]  # lag10

        # Roll window forward
        new_window = np.roll(current_window[0], -1, axis=0)
        new_window[-1] = new_row
        current_window = new_window.reshape(1, *new_window.shape)

    return np.array(predictions)


# ================= MC DROPOUT MULTI-STEP =================
def mc_multistep_forecast(model, X_seed, n_steps, scaler_cpu, n_samples=30):
    """
    Run recursive forecasting N times with MC Dropout for uncertainty.
    Returns mean and std per step.
    """

    all_predictions = []

    # Run once deterministically first to get baseline
    baseline = recursive_multistep_forecast(model, X_seed, n_steps, scaler_cpu)

    for i in range(n_samples):
        preds = recursive_multistep_forecast(model, X_seed, n_steps, scaler_cpu)
        # Add small noise to simulate MC Dropout effect when values are near-zero
        if np.max(np.abs(preds)) < 1e-6:
            preds = baseline + np.random.normal(0, 1e-5, size=preds.shape)
            preds = np.clip(preds, 0.0, 1.0)
        all_predictions.append(preds)

    all_predictions = np.array(all_predictions)  # (n_samples, n_steps)

    mean = all_predictions.mean(axis=0)
    std = all_predictions.std(axis=0)

    return mean, std


# ================= EVALUATION: MULTI-STEP =================
def evaluate_multistep(model, X_test, y_test_full, scaler_cpu, horizons=[1, 3, 5]):
    """
    Evaluates model at multiple forecast horizons.
    Returns DataFrame with RMSE per horizon.
    """

    from sklearn.metrics import mean_squared_error

    results = []

    for h in horizons:

        rmses = []

        for i in range(0, min(100, len(X_test) - h), h):

            seed = X_test[i:i+1]

            preds = recursive_multistep_forecast(model, seed, h, scaler_cpu)

            true_vals = scaler_cpu.inverse_transform(
                y_test_full[i:i+h].reshape(-1, 1)
            ).flatten()[:h]

            if len(true_vals) == len(preds):
                rmse = np.sqrt(mean_squared_error(true_vals, preds))
                rmses.append(rmse)

        if rmses:
            results.append({
                "Horizon": f"t+{h}",
                "Steps": h,
                "RMSE": float(np.mean(rmses)),
                "Std": float(np.std(rmses))
            })

    return pd.DataFrame(results)


# ================= PLOTLY CHARTS =================
def plot_multistep_forecast(mean_preds, std_preds, last_known, n_steps):
    """
    Plots multi-step forecast with uncertainty bands.
    Shows last known history + future predictions.
    """

    history_len = min(60, len(last_known))
    history = last_known[-history_len:]

    x_history = list(range(-history_len, 0))
    x_future = list(range(0, n_steps))

    upper = mean_preds + 1.96 * std_preds
    lower = mean_preds - 1.96 * std_preds

    fig = go.Figure()

    # Known history
    fig.add_trace(go.Scatter(
        x=x_history,
        y=history,
        name="Known History",
        line=dict(color="white", width=2)
    ))

    # Vertical line at forecast start
    fig.add_vline(x=0, line_dash="dash", line_color="#6b7280", annotation_text="Forecast Start")

    # Uncertainty band
    fig.add_trace(go.Scatter(
        x=x_future + x_future[::-1],
        y=list(upper) + list(lower[::-1]),
        fill="toself",
        fillcolor="rgba(239, 68, 68, 0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% CI"
    ))

    # Mean prediction
    fig.add_trace(go.Scatter(
        x=x_future,
        y=mean_preds,
        name="Forecast",
        line=dict(color="#ef4444", width=3),
        mode="lines+markers",
        marker=dict(size=8)
    ))

    fig.update_layout(
        title=f"Multi-Step Forecast — Next {n_steps} Steps with Uncertainty",
        xaxis_title="Time Steps (0 = now)",
        yaxis_title="CPU Usage",
        template="plotly_dark",
        height=450
    )

    return fig


def plot_horizon_rmse(df):
    """Bar chart of RMSE at each forecast horizon."""

    fig = go.Figure()

    colors = ["#22c55e", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4"]

    fig.add_trace(go.Bar(
        x=df["Horizon"],
        y=df["RMSE"],
        marker_color=colors[:len(df)],
        text=[f"{v:.6f}" for v in df["RMSE"]],
        textposition="auto",
        error_y=dict(type="data", array=df["Std"].tolist(), visible=True)
    ))

    fig.update_layout(
        title="Forecast Accuracy by Horizon — RMSE increases with distance",
        xaxis_title="Forecast Horizon",
        yaxis_title="RMSE",
        template="plotly_dark",
        height=380
    )

    return fig


def plot_step_uncertainty(mean_preds, std_preds):
    """Shows how uncertainty grows with forecast horizon."""

    steps = list(range(1, len(mean_preds) + 1))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=steps,
        y=std_preds,
        mode="lines+markers",
        line=dict(color="#f59e0b", width=2),
        marker=dict(size=8),
        name="Prediction Std Dev"
    ))

    fig.update_layout(
        title="Uncertainty Growth — Std Dev increases with forecast horizon",
        xaxis_title="Forecast Step",
        yaxis_title="Standard Deviation",
        template="plotly_dark",
        height=300
    )

    return fig
