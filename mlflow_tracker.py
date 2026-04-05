"""
MLflow Experiment Tracking Module
Logs all experiments, hyperparameters, metrics, and models.
Research-grade ML experiment management.
"""

import mlflow
import mlflow.keras
import numpy as np
import os
from datetime import datetime


# ================= SETUP =================
EXPERIMENT_NAME = "CPU_Workload_Forecasting"


def setup_mlflow(tracking_uri="mlruns"):
    """Initialize MLflow with local tracking."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


# ================= LOG EXPERIMENT =================
def log_experiment(
    model,
    model_type,
    params,
    metrics,
    window_size,
    forecast_horizon,
    feature_names
):
    """
    Logs a complete experiment run to MLflow.

    Args:
        model: trained Keras model
        model_type: "LSTM", "TCN", or "Transformer"
        params: dict of hyperparameters
        metrics: dict with RMSE, MAE, Coverage etc.
        window_size: int
        forecast_horizon: int
        feature_names: list of feature names
    """

    setup_mlflow()

    run_name = f"{model_type}_{datetime.now().strftime('%H%M%S')}"

    with mlflow.start_run(run_name=run_name):

        # ---- Tags ----
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("project", "CPU_Forecasting")
        mlflow.set_tag("author", "Ary Kashid")
        mlflow.set_tag("features", str(feature_names))

        # ---- Parameters ----
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("window_size", window_size)
        mlflow.log_param("forecast_horizon", forecast_horizon)
        mlflow.log_param("num_features", len(feature_names))

        for key, val in params.items():
            mlflow.log_param(key, val)

        # ---- Metrics ----
        if "LSTM" in metrics:
            mlflow.log_metric("lstm_rmse", metrics["LSTM"]["RMSE"])
            mlflow.log_metric("lstm_mae", metrics["LSTM"]["MAE"])

        if "Naive" in metrics:
            mlflow.log_metric("naive_rmse", metrics["Naive"]["RMSE"])

        if "ARIMA" in metrics:
            mlflow.log_metric("arima_rmse", metrics["ARIMA"]["RMSE"])

        if "Diagnostics" in metrics:
            diag = metrics["Diagnostics"]
            if diag.get("LjungBox_pvalue") == diag.get("LjungBox_pvalue"):
                mlflow.log_metric("ljungbox_pvalue", diag["LjungBox_pvalue"])
            if diag.get("Coverage_95") == diag.get("Coverage_95"):
                mlflow.log_metric("coverage_95", diag["Coverage_95"])
            if diag.get("WalkForward_RMSE") == diag.get("WalkForward_RMSE"):
                mlflow.log_metric("walkforward_rmse", diag["WalkForward_RMSE"])
            if diag.get("DieboldMariano_pvalue") == diag.get("DieboldMariano_pvalue"):
                mlflow.log_metric("dm_pvalue", diag["DieboldMariano_pvalue"])

        # ---- Log Model ----
        try:
            mlflow.keras.log_model(model, artifact_path="model")
        except Exception:
            pass

        # ---- Return run info ----
        run_id = mlflow.active_run().info.run_id

    return run_id


# ================= GET ALL RUNS =================
def get_all_runs():
    """
    Returns a DataFrame of all experiment runs with metrics.
    """

    setup_mlflow()

    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

        if experiment is None:
            return None

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.lstm_rmse ASC"]
        )

        if not runs:
            return None

        import pandas as pd

        records = []

        for run in runs:
            record = {
                "Run ID": run.info.run_id[:8],
                "Run Name": run.info.run_name,
                "Model": run.data.tags.get("model_type", "N/A"),
                "LSTM RMSE": round(run.data.metrics.get("lstm_rmse", float("nan")), 6),
                "LSTM MAE": round(run.data.metrics.get("lstm_mae", float("nan")), 6),
                "Coverage 95%": round(run.data.metrics.get("coverage_95", float("nan")), 4),
                "WF RMSE": round(run.data.metrics.get("walkforward_rmse", float("nan")), 6),
                "Status": run.info.status,
                "Timestamp": datetime.fromtimestamp(
                    run.info.start_time / 1000
                ).strftime("%Y-%m-%d %H:%M")
            }
            records.append(record)

        return pd.DataFrame(records)

    except Exception as e:
        return None


# ================= GET BEST RUN =================
def get_best_run():
    """Returns the run with lowest LSTM RMSE."""

    setup_mlflow()

    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

        if experiment is None:
            return None

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.lstm_rmse ASC"],
            max_results=1
        )

        if not runs:
            return None

        best = runs[0]

        return {
            "run_id": best.info.run_id[:8],
            "model_type": best.data.tags.get("model_type", "N/A"),
            "lstm_rmse": best.data.metrics.get("lstm_rmse", float("nan")),
            "lstm_mae": best.data.metrics.get("lstm_mae", float("nan")),
            "coverage_95": best.data.metrics.get("coverage_95", float("nan")),
            "params": best.data.params
        }

    except Exception:
        return None
