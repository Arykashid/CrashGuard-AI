"""
Real-Time Simulation Module (Research Grade – German Master's Level)

Implements leakage-free rolling-window prediction with:
- True real-time inference logic
- Instantaneous error tracking
- Cumulative error drift analysis
"""

import numpy as np


def simulate_realtime_prediction(
    model,
    data,
    scaler,
    window_size=60,
    start_idx=0,
    num_predictions=100
):
    """
    Simulate real-time forecasting using a rolling prediction window.

    Parameters:
    ----------
    model : trained keras model
    data : pandas DataFrame with 'cpu_usage'
    scaler : fitted MinMaxScaler
    window_size : int
        Lookback window size
    start_idx : int
        Simulation start index
    num_predictions : int
        Number of real-time prediction steps

    Returns:
    -------
    dict containing predictions, actuals, errors,
    and cumulative error drift.
    """

    series = data["cpu_usage"].values

    # Initialize window using ONLY past observations
    current_window = series[start_idx:start_idx + window_size].copy()

    predictions = []
    actuals = []
    errors = []
    cumulative_error = []

    error_sum = 0.0

    for i in range(num_predictions):
        # Scale input window
        x_scaled = scaler.transform(current_window.reshape(-1, 1))
        x_scaled = x_scaled.reshape(1, window_size, 1)

        # Predict next time step
        y_scaled = model.predict(x_scaled, verbose=0)
        y_pred = scaler.inverse_transform(y_scaled)[0][0]

        # Ground truth (used strictly for evaluation)
        y_true = series[start_idx + window_size + i]

        # Error analysis
        err = y_pred - y_true
        error_sum += abs(err)

        predictions.append(y_pred)
        actuals.append(y_true)
        errors.append(err)
        cumulative_error.append(error_sum)

        # Update window using prediction (true real-time assumption)
        current_window = np.append(current_window[1:], y_pred)

    return {
        "predictions": np.array(predictions),
        "actuals": np.array(actuals),
        "errors": np.array(errors),
        "cumulative_error": np.array(cumulative_error)
    }