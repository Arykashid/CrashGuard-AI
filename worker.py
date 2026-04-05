"""
worker.py — CrashGuard AI Background Worker
FIXES APPLIED:
  1. WINDOW_SIZE default 40 → 60 (matches training)
  2. model.predict() → model(X_tensor, training=False) (MC Dropout active)
  3. n_mc 10 → 30 (stable confidence estimates)
  4. Recursive forecast capped at 10 steps (beyond = error accumulation)
  5. Confidence: exponential decay, never clips to 0.00
"""

import time
import joblib
import numpy as np
import psutil
import os
import json
import logging
import tensorflow as tf
from datetime import datetime
from collections import deque
from notifications import send_slack_alert, SLACK_WEBHOOK_URL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("worker.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# =============================================
# CONFIG — ALL must match training exactly
# =============================================
CHECK_INTERVAL       = int(os.getenv("CHECK_INTERVAL", 5))
WINDOW_SIZE          = int(os.getenv("WINDOW_SIZE", 60))       # FIX: was 40
SPIKE_THRESHOLD      = float(os.getenv("SPIKE_THRESHOLD", 0.75))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.4))
ALERT_COOLDOWN       = int(os.getenv("ALERT_COOLDOWN", 60))
PREDICTION_LOG       = os.getenv("PREDICTION_LOG", "predictions.json")
FORECAST_STEPS       = int(os.getenv("FORECAST_STEPS", 10))    # FIX: was 60
PROMETHEUS_URL       = os.getenv("PROMETHEUS_URL", "")
SERVER_NAME          = os.getenv("SERVER_NAME", "node-1")
N_MC_SAMPLES         = 30                                       # FIX: was 10

cpu_history    = deque(maxlen=200)
last_alert_time = None
alert_count    = 0


def load_model():
    try:
        from lstm_model import MCDropout
        custom_objects = {"MCDropout": MCDropout}
    except Exception:
        custom_objects = {}
    try:
        if os.path.exists("saved_model.keras"):
            model = tf.keras.models.load_model(
                "saved_model.keras", custom_objects=custom_objects
            )
            log.info("Model loaded: saved_model.keras")
            return model
        elif os.path.exists("saved_model.pkl"):
            with tf.keras.utils.custom_object_scope(custom_objects):
                model = joblib.load("saved_model.pkl")
            log.info("Model loaded: saved_model.pkl")
            return model
        else:
            log.warning("No model found. Run train.py first.")
            return None
    except Exception as e:
        log.error(f"Model load error: {e}")
        return None


def get_cpu():
    if PROMETHEUS_URL:
        try:
            import requests
            query = "1 - avg(rate(node_cpu_seconds_total{mode='idle'}[1m]))"
            r = requests.get(
                f"{PROMETHEUS_URL}/api/v1/query",
                params={"query": query}, timeout=3
            )
            data = r.json()
            if data["status"] == "success" and data["data"]["result"]:
                return float(np.clip(
                    float(data["data"]["result"][0]["value"][1]), 0.0, 1.0
                ))
        except Exception as e:
            log.warning(f"Prometheus failed, using psutil: {e}")
    return psutil.cpu_percent(interval=0.5) / 100.0


def build_features(cpu_array):
    features = []
    for i, c in enumerate(cpu_array):
        h         = datetime.now().hour
        dow       = datetime.now().weekday()
        hour_sin  = np.sin(2 * np.pi * h / 24)
        hour_cos  = np.cos(2 * np.pi * h / 24)
        dow_sin   = np.sin(2 * np.pi * dow / 7)
        dow_cos   = np.cos(2 * np.pi * dow / 7)
        lag1      = cpu_array[i - 1]  if i > 0  else c
        lag5      = cpu_array[i - 5]  if i > 4  else c
        lag10     = cpu_array[i - 10] if i > 9  else c
        roll_mean = np.mean(cpu_array[max(0, i - 10):i + 1])
        roll_std  = np.std(cpu_array[max(0, i - 10):i + 1]) + 1e-6
        features.append([
            c, hour_sin, hour_cos, dow_sin, dow_cos,
            lag1, lag5, lag10, roll_mean, roll_std
        ])
    return np.array(features, dtype=np.float32)


def mc_predict_single(model, X):
    """
    CORRECT MC Dropout inference.

    KEY: model(X, training=False) triggers MCDropout.call()
    which internally forces training=True → dropout stays active
    → each of N_MC_SAMPLES passes is different → std > 0.

    WRONG (old code): model.predict(X) → dropout disabled → std=0
                      → confidence=0.00 every time
    """
    X_tensor = tf.constant(X, dtype=tf.float32)
    preds = [
        float(model(X_tensor, training=False).numpy()[0][0])
        for _ in range(N_MC_SAMPLES)
    ]
    mean = float(np.clip(np.mean(preds), 0.0, 1.0))
    std  = float(np.std(preds))

    # Exponential decay confidence — smooth, never hits 0.00
    # std=0.00 → conf=1.00 | std=0.05 → conf=0.37 | std=0.15 → conf=0.05
    confidence = float(np.clip(np.exp(-std / 0.05), 0.05, 1.0))
    return mean, std, confidence


def predict_next_5_minutes(model, history):
    if model is None or len(history) < WINDOW_SIZE:
        return None, None, None, None
    try:
        all_predictions = []
        hist = list(history)[-WINDOW_SIZE:]

        for step in range(FORECAST_STEPS):
            cpu_arr  = np.array(hist[-WINDOW_SIZE:])
            features = build_features(cpu_arr)
            X        = features.reshape(1, WINDOW_SIZE, 10)

            mean_pred, std_pred, confidence = mc_predict_single(model, X)

            all_predictions.append({
                "step":         step + 1,
                "predicted":    mean_pred,
                "std":          std_pred,
                "confidence":   confidence,
                "time_seconds": (step + 1) * CHECK_INTERVAL
            })
            hist.append(mean_pred)

        spike_step, time_to_spike = None, None
        for p in all_predictions:
            if p["predicted"] > SPIKE_THRESHOLD:
                spike_step    = p["step"]
                time_to_spike = p["time_seconds"]
                break

        avg_confidence = float(np.mean([p["confidence"] for p in all_predictions]))
        return all_predictions, avg_confidence, spike_step, time_to_spike

    except Exception as e:
        log.error(f"Prediction error: {e}")
        return None, None, None, None


def log_prediction(current, predicted, confidence, alert_fired):
    try:
        entry = {
            "timestamp":     datetime.now().isoformat(),
            "current_cpu":   round(current, 4),
            "predicted_cpu": round(predicted, 4) if predicted is not None else None,
            "confidence":    round(confidence, 4) if confidence is not None else None,
            "alert_fired":   alert_fired
        }
        logs = []
        if os.path.exists(PREDICTION_LOG):
            with open(PREDICTION_LOG) as f:
                try:
                    logs = json.load(f)
                except Exception:
                    logs = []
        logs.append(entry)
        with open(PREDICTION_LOG, "w") as f:
            json.dump(logs[-1000:], f, indent=2)
    except Exception as e:
        log.error(f"Log error: {e}")


def check_alert(current, predictions, confidence, time_to_spike):
    global last_alert_time, alert_count

    if predictions is None:
        return False
    if last_alert_time:
        elapsed = (datetime.now() - last_alert_time).seconds
        if elapsed < ALERT_COOLDOWN:
            return False

    max_pred = max(p["predicted"] for p in predictions)

    if max_pred > SPIKE_THRESHOLD and confidence > CONFIDENCE_THRESHOLD:
        severity = (
            "HIGH"   if max_pred > 0.85 else
            "MEDIUM" if max_pred > 0.75 else
            "LOW"
        )
        minutes = round(time_to_spike / 60, 1) if time_to_spike else "< 1"
        action  = (
            "Scale instance immediately!"
            if severity == "HIGH"
            else "Monitor closely — consider pre-scaling"
        )
        log.warning(
            f"SPIKE ALERT | {severity} | "
            f"In {minutes}min | "
            f"Predicted: {max_pred:.1%} | "
            f"Current: {current:.1%} | "
            f"Confidence: {confidence:.3f}"
        )
        if SLACK_WEBHOOK_URL:
            sent = send_slack_alert(
                level=severity, predicted_cpu=max_pred,
                current_cpu=current,
                action=f"Spike in {minutes}min. {action}"
            )
            log.info(f"Slack: {'sent' if sent else 'FAILED'}")
        last_alert_time = datetime.now()
        alert_count    += 1
        return True
    return False


def run():
    log.info("CrashGuard AI Worker starting")
    log.info(f"Window: {WINDOW_SIZE} | Forecast: {FORECAST_STEPS} | MC samples: {N_MC_SAMPLES}")
    log.info(f"Spike threshold: {SPIKE_THRESHOLD:.0%} | Slack: {'yes' if SLACK_WEBHOOK_URL else 'no'}")

    model = load_model()
    if model is None:
        log.warning("No model — collecting data only")

    while True:
        try:
            current = get_cpu()
            cpu_history.append(current)

            predictions, confidence, spike_step, time_to_spike = \
                predict_next_5_minutes(model, cpu_history)

            if predictions is not None:
                max_pred   = max(p["predicted"] for p in predictions)
                spike_info = (
                    f"Spike in {round(time_to_spike/60,1)}min"
                    if time_to_spike else "No spike"
                )
                log.info(
                    f"CPU: {current:.1%} | Max pred: {max_pred:.1%} | "
                    f"Conf: {confidence:.3f} | {spike_info} | Alerts: {alert_count}"
                )
            else:
                log.info(
                    f"CPU: {current:.1%} | "
                    f"Warming up ({max(0, WINDOW_SIZE - len(cpu_history))} more needed)"
                )

            alert_fired   = check_alert(current, predictions, confidence or 0.0, time_to_spike)
            predicted_val = max(p["predicted"] for p in predictions) if predictions else None
            log_prediction(current, predicted_val, confidence, alert_fired)
            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            log.info("Worker stopped.")
            break
        except Exception as e:
            log.error(f"Loop error: {e}")
            time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    run()
