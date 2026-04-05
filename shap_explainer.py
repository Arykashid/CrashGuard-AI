"""
SHAP Explainability Module
Explains LSTM predictions using SHAP KernelExplainer
"""

import numpy as np
import shap
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


# ================= SHAP EXPLAINER =================
def get_shap_explainer(model, X_background):
    """
    Creates a SHAP KernelExplainer using a small background dataset.
    We flatten the 3D input to 2D for SHAP compatibility.
    """

    # Flatten: (samples, window, features) -> (samples, window*features)
    # Use minimum of 50 or available samples to avoid shape mismatch
    n_bg = min(50, len(X_background))
    X_bg = X_background[:n_bg]
    X_bg_flat = X_bg.reshape(n_bg, -1)

    # Store window/feature shape for reshaping
    _window = X_background.shape[1]
    _features = X_background.shape[2]

    def model_predict(X_flat):
        # Reshape back to 3D for model
        n = X_flat.shape[0]
        X_3d = X_flat.reshape(n, _window, _features)
        pred = model.predict(X_3d, verbose=0)
        # Flatten to 1D - works for any forecast horizon
        return pred.flatten()[:n]

    explainer = shap.KernelExplainer(model_predict, X_bg_flat)

    return explainer


def compute_shap_values(explainer, X_sample, n_samples=10):
    """
    Computes SHAP values for n_samples test instances.
    Returns shap_values array and flattened X.
    """

    X_flat = X_sample[:n_samples].reshape(n_samples, -1)

    shap_values = explainer.shap_values(X_flat, nsamples=100)

    return shap_values, X_flat


def get_feature_importance(shap_values, feature_names, window_size):
    """
    Aggregates SHAP values across time steps per feature.
    Returns a DataFrame with mean absolute SHAP per feature.
    """

    n_features = len(feature_names)

    # shap_values shape: (n_samples, window_size * n_features)
    shap_array = np.array(shap_values)

    # Reshape to (n_samples, window_size, n_features)
    shap_3d = shap_array.reshape(shap_array.shape[0], window_size, n_features)

    # Mean absolute SHAP per feature across time and samples
    mean_abs_shap = np.mean(np.abs(shap_3d), axis=(0, 1))

    df = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Abs_SHAP": mean_abs_shap
    }).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

    return df


def get_temporal_shap(shap_values, window_size, n_features):
    """
    Returns SHAP importance aggregated over features per time step.
    Shows which time steps matter most.
    """

    shap_array = np.array(shap_values)

    # Reshape to (n_samples, window_size, n_features)
    shap_3d = shap_array.reshape(shap_array.shape[0], window_size, n_features)

    # Mean absolute SHAP per time step
    temporal_importance = np.mean(np.abs(shap_3d), axis=(0, 2))

    return temporal_importance


# ================= PLOTLY CHARTS =================
def plot_feature_importance(df):
    """Bar chart of feature importance."""

    fig = go.Figure()

    colors = ["#ef4444" if i == 0 else "#3b82f6" if i == 1 else "#6b7280"
              for i in range(len(df))]

    fig.add_trace(go.Bar(
        x=df["Mean_Abs_SHAP"],
        y=df["Feature"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.6f}" for v in df["Mean_Abs_SHAP"]],
        textposition="auto"
    ))

    fig.update_layout(
        title="SHAP Feature Importance — What drives CPU predictions?",
        xaxis_title="Mean |SHAP Value|",
        yaxis_title="Feature",
        template="plotly_dark",
        height=450,
        yaxis=dict(autorange="reversed")
    )

    return fig


def plot_temporal_importance(temporal_importance):
    """Line chart showing which time steps matter most."""

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(len(temporal_importance))),
        y=temporal_importance,
        mode="lines+markers",
        line=dict(color="#ef4444", width=2),
        marker=dict(size=4)
    ))

    fig.update_layout(
        title="Temporal SHAP — Which past time steps influence predictions most?",
        xaxis_title="Time Step (0 = oldest, N = most recent)",
        yaxis_title="Mean |SHAP Value|",
        template="plotly_dark",
        height=350
    )

    return fig


def plot_shap_waterfall(shap_vals_single, feature_names, window_size):
    """
    Waterfall-style chart for a single prediction explanation.
    Shows top 10 most impactful features.
    """

    n_features = len(feature_names)

    # Reshape single sample shap values
    shap_3d = shap_vals_single.reshape(window_size, n_features)

    # Sum across time steps per feature
    feature_shap = shap_3d.sum(axis=0)

    df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP": feature_shap
    }).sort_values("SHAP", key=abs, ascending=False).head(10)

    colors = ["#ef4444" if v > 0 else "#3b82f6" for v in df["SHAP"]]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df["Feature"],
        y=df["SHAP"],
        marker_color=colors,
        text=[f"{v:+.6f}" for v in df["SHAP"]],
        textposition="auto"
    ))

    fig.update_layout(
        title="Single Prediction Explanation — Top 10 Feature Contributions",
        xaxis_title="Feature",
        yaxis_title="SHAP Value",
        template="plotly_dark",
        height=400
    )

    return fig