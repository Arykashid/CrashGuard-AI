"""
FastAPI REST API — CPU Workload Forecasting
Production-grade API endpoint for model serving.

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /              — API info
    GET  /health        — Health check
    POST /predict       — Single prediction
    POST /predict/batch — Batch predictions
    POST /predict/multistep — Multi-step forecast
    GET  /model/info    — Model information
    GET  /metrics       — Latest evaluation metrics
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import joblib
from lstm_model import MCDropout
import json
import os
from datetime import datetime
from lstm_model import MCDropout  # Required for model loading


# ================= APP SETUP =================
app = FastAPI(
    title="CPU Workload Forecasting API",
    description="Research-grade REST API for real-time CPU workload prediction using LSTM/TCN/Transformer",
    version="1.0.0",
    contact={
        "name": "Ary Kashid",
        "email": "your@email.com"
    }
)

# Allow all origins for demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ================= LOAD MODEL =================
model = None
evaluation_results = None

def load_model():
    global model, evaluation_results

    model_path = "saved_model.pkl"
    results_path = "evaluation/lstm_results.json"

    if os.path.exists(model_path):
        import tensorflow as tf
        model = joblib.load(model_path)
        print(f"✅ Model loaded from {model_path}")
    else:
        print(f"⚠️ No model found at {model_path}. Train model first.")

    if os.path.exists(results_path):
        with open(results_path) as f:
            evaluation_results = json.load(f)

load_model()


# ================= REQUEST MODELS =================
class PredictRequest(BaseModel):
    cpu_usage: float = Field(
        ..., ge=0.0, le=1.0,
        description="Current CPU usage (0.0 to 1.0)",
        example=0.45
    )
    hour: int = Field(
        default=12, ge=0, le=23,
        description="Hour of day (0-23)",
        example=14
    )
    day_of_week: int = Field(
        default=1, ge=0, le=6,
        description="Day of week (0=Monday, 6=Sunday)",
        example=1
    )
    window_size: int = Field(
        default=60, ge=10, le=300,
        description="Lookback window size",
        example=60
    )


class BatchPredictRequest(BaseModel):
    readings: List[float] = Field(
        ..., min_items=1, max_items=1000,
        description="List of CPU readings (0.0 to 1.0)",
        example=[0.3, 0.35, 0.4, 0.38, 0.42]
    )
    hour: int = Field(default=12, ge=0, le=23)
    day_of_week: int = Field(default=1, ge=0, le=6)


class MultiStepRequest(BaseModel):
    cpu_history: List[float] = Field(
        ..., min_items=10,
        description="Recent CPU history (at least 10 values)",
        example=[0.3, 0.32, 0.35, 0.4, 0.38, 0.42, 0.45, 0.43, 0.41, 0.39]
    )
    n_steps: int = Field(
        default=5, ge=1, le=60,
        description="Number of steps to forecast ahead",
        example=5
    )


# ================= HELPER FUNCTIONS =================
def build_feature_vector(cpu, hour, dow):
    """Builds feature vector from inputs."""
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)

    return np.array([
        cpu,
        hour_sin, hour_cos,
        dow_sin, dow_cos,
        cpu,  # lag1
        cpu,  # lag5
        cpu,  # lag10
        cpu,  # roll_mean
        0.01  # roll_std
    ])


def make_window(features, window_size):
    """Creates model input window."""
    return np.tile(features, (window_size, 1)).reshape(1, window_size, 10)


# ================= ENDPOINTS =================

@app.get("/", tags=["Info"])
def root():
    """API information and available endpoints."""
    return {
        "name": "CPU Workload Forecasting API",
        "version": "1.0.0",
        "author": "Ary Kashid",
        "description": "Research-grade CPU workload prediction API",
        "model_loaded": model is not None,
        "endpoints": {
            "POST /predict": "Single CPU prediction",
            "POST /predict/batch": "Batch predictions",
            "POST /predict/multistep": "Multi-step forecast",
            "GET /model/info": "Model information",
            "GET /metrics": "Evaluation metrics",
            "GET /health": "Health check"
        },
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health", tags=["Info"])
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", tags=["Prediction"])
def predict(request: PredictRequest):
    """
    Single CPU usage prediction.

    Returns predicted CPU usage for next time step
    along with confidence level.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train model first using: python train.py"
        )

    try:
        features = build_feature_vector(
            request.cpu_usage,
            request.hour,
            request.day_of_week
        )
        window = make_window(features, request.window_size)
        raw_pred = float(model.predict(window, verbose=0)[0][0])
        prediction = float(np.clip(raw_pred, 0.0, 1.0))

        # Alert level
        if prediction > 0.8:
            alert = "HIGH"
            recommendation = "Scale up resources immediately"
        elif prediction > 0.6:
            alert = "MEDIUM"
            recommendation = "Monitor closely, consider pre-scaling"
        else:
            alert = "NORMAL"
            recommendation = "No action needed"

        return {
            "prediction": round(prediction, 6),
            "prediction_pct": f"{prediction:.1%}",
            "alert_level": alert,
            "recommendation": recommendation,
            "input": {
                "cpu_usage": request.cpu_usage,
                "hour": request.hour,
                "day_of_week": request.day_of_week
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(request: BatchPredictRequest):
    """
    Batch CPU predictions for multiple readings.

    Useful for processing historical data or
    making predictions for multiple time points.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        predictions = []

        for cpu in request.readings:
            features = build_feature_vector(
                cpu,
                request.hour,
                request.day_of_week
            )
            window = make_window(features, 60)
            raw_pred = float(model.predict(window, verbose=0)[0][0])
            pred = float(np.clip(raw_pred, 0.0, 1.0))
            predictions.append(round(pred, 6))

        return {
            "predictions": predictions,
            "count": len(predictions),
            "avg_prediction": round(float(np.mean(predictions)), 6),
            "max_prediction": round(float(np.max(predictions)), 6),
            "min_prediction": round(float(np.min(predictions)), 6),
            "high_alert_count": sum(1 for p in predictions if p > 0.8),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/multistep", tags=["Prediction"])
def predict_multistep(request: MultiStepRequest):
    """
    Multi-step CPU forecast.

    Predicts next N steps using recursive forecasting.
    Each prediction uses previous prediction as input.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        history = list(request.cpu_history)
        predictions = []
        window_size = min(60, len(history))

        for step in range(request.n_steps):
            recent = history[-window_size:]
            cpu = recent[-1]
            hour = datetime.now().hour
            dow = datetime.now().weekday()

            features = build_feature_vector(cpu, hour, dow)
            window = np.tile(features, (window_size, 1)).reshape(1, window_size, 10)
            raw_pred = float(model.predict(window, verbose=0)[0][0])
            pred = float(np.clip(raw_pred, 0.0, 1.0))

            predictions.append(round(pred, 6))
            history.append(pred)

        return {
            "predictions": predictions,
            "n_steps": request.n_steps,
            "forecast_horizon": f"{request.n_steps} seconds ahead",
            "trend": "Rising" if predictions[-1] > predictions[0] else "Falling",
            "max_predicted": round(max(predictions), 6),
            "alert": "HIGH" if max(predictions) > 0.8 else "NORMAL",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info", tags=["Model"])
def model_info():
    """Returns model architecture and configuration info."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        n_params = model.count_params()
        layers = [
            {"name": l.name, "type": type(l).__name__}
            for l in model.layers
        ]
    except Exception:
        n_params = "Unknown"
        layers = []

    return {
        "model_type": "LSTM",
        "parameters": n_params,
        "input_shape": "(batch, window_size, 10_features)",
        "output_shape": "(batch, forecast_horizon)",
        "features": [
            "cpu_usage", "hour_sin", "hour_cos",
            "dow_sin", "dow_cos", "lag1", "lag5",
            "lag10", "roll_mean_10", "roll_std_10"
        ],
        "layers": layers,
        "framework": "TensorFlow/Keras",
        "uncertainty": "Monte Carlo Dropout (50 passes)",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/metrics", tags=["Model"])
def get_metrics():
    """Returns latest evaluation metrics."""
    if evaluation_results is None:
        return {
            "message": "No evaluation results found. Run: python train.py --all",
            "timestamp": datetime.now().isoformat()
        }

    return {
        "metrics": evaluation_results,
        "timestamp": datetime.now().isoformat()
    }