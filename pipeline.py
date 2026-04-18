"""
pipeline.py — CrashGuard AI
Connects server_simulator → feature_engine → LSTM+XGBoost ensemble.

Fix: scaler.pkl and cpu_scaler.pkl saved with joblib in train.py,
     so loaded with joblib here (not pickle).
"""

import os
import time
import logging
import threading
import traceback
import numpy as np
from datetime import datetime, timezone
from typing import Optional

import joblib
import server_simulator as sim
from feature_engine import build_features, is_spike_context

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("crashguard.pipeline")

PREDICTION_INTERVAL = 2
MC_DROPOUT_SAMPLES  = 15
HISTORY_WINDOW      = 60
SPIKE_THRESHOLD     = 75.0

MODEL_DIR       = os.path.dirname(os.path.abspath(__file__))
LSTM_PATH       = os.path.join(MODEL_DIR, "saved_model.keras")
XGB_PATH        = os.path.join(MODEL_DIR, "saved_xgb_model.pkl")
SCALER_PATH     = os.path.join(MODEL_DIR, "scaler.pkl")
CPU_SCALER_PATH = os.path.join(MODEL_DIR, "cpu_scaler.pkl")
CALIB_PATH      = os.path.join(MODEL_DIR, "calibration_temperature.pkl")

XGB_WEIGHT  = 0.60
LSTM_WEIGHT = 0.40


def extract_xgb_features(features: dict) -> np.ndarray:
    """
    Extract 11 features XGBoost was trained on.
    Must match xgboost_model.py XGB_FEATURE_NAMES exactly:
        lag1, lag2, lag3, lag5, lag10,
        roll_mean_10, roll_std_10,
        hour_sin, hour_cos,
        cpu_diff1, cpu_diff3
    """
    return np.array([
        features["cpu_lag1"],
        features["cpu_lag3"],
        features["cpu_lag3"],
        features["cpu_lag5"],
        features["cpu_lag10"],
        features["rolling_mean_10"],
        features["rolling_std_10"],
        features["hour_sin"],
        features["hour_cos"],
        features["delta_1"],
        features["delta_5"],
    ], dtype=np.float32).reshape(1, -1)


class ModelLoader:

    def __init__(self):
        self.lstm        = None
        self.xgb         = None
        self.scaler      = None
        self.cpu_scaler  = None
        self.temperature = 1.0
        self.ready       = False
        self._load()

    def _load(self):
        try:
            import pickle

            # XGBoost — joblib (matches train.py)
            self.xgb = joblib.load(XGB_PATH)
            logger.info("XGBoost model loaded.")

            # Scalers — joblib (matches train.py joblib.dump)
            self.scaler     = joblib.load(SCALER_PATH)
            self.cpu_scaler = joblib.load(CPU_SCALER_PATH)
            logger.info("Scalers loaded.")

            # Calibration temperature — pickle (small float)
            try:
                with open(CALIB_PATH, "rb") as f:
                    self.temperature = float(pickle.load(f))
                logger.info(f"Calibration temperature loaded: T={self.temperature:.4f}")
            except Exception:
                logger.warning("Calibration file not found — using T=1.0")
                self.temperature = 1.0

            # LSTM — import MCDropout from lstm_model.py
            try:
                os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
                import tensorflow as tf
                from lstm_model import MCDropout
                self.lstm = tf.keras.models.load_model(
                    LSTM_PATH,
                    custom_objects={"MCDropout": MCDropout},
                )
                logger.info("LSTM model loaded.")
            except Exception as e:
                logger.warning(f"LSTM load failed ({e}) — will use XGBoost only.")
                self.lstm = None

            self.ready = True
            logger.info("Pipeline ready.")

        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            logger.error(traceback.format_exc())
            self.ready = False


class Predictor:

    def __init__(self, loader: ModelLoader):
        self.loader = loader

    def predict(self, history: list[dict], features: dict) -> dict:
        loader = self.loader
        if not loader.ready:
            return self._fallback(features)

        xgb_pred = self._xgb_predict(features)

        if loader.lstm is not None:
            lstm_mean, lstm_std = self._lstm_predict(history)
            model_used = "ensemble"
        else:
            lstm_mean = xgb_pred
            lstm_std  = 2.0
            model_used = "xgboost_only"

        ensemble_pred = XGB_WEIGHT * xgb_pred + LSTM_WEIGHT * lstm_mean

        z        = 1.282
        ci_lower = max(0.0,   ensemble_pred - z * lstm_std)
        ci_upper = min(100.0, ensemble_pred + z * lstm_std)

        from math import erf, sqrt
        if lstm_std > 0:
            z_spike    = (SPIKE_THRESHOLD - ensemble_pred) / (lstm_std * sqrt(2))
            spike_prob = max(0.0, min(1.0, 0.5 * (1 - erf(z_spike))))
        else:
            spike_prob = 1.0 if ensemble_pred > SPIKE_THRESHOLD else 0.0

        ci_width   = ci_upper - ci_lower
        confidence = max(0.0, min(1.0, 1.0 - (ci_width / 100.0)))

        return {
            "predicted_cpu":     round(float(ensemble_pred), 2),
            "confidence":        round(float(confidence),    4),
            "ci_lower":          round(float(ci_lower),      2),
            "ci_upper":          round(float(ci_upper),      2),
            "spike_probability": round(float(spike_prob),    4),
            "model_used":        model_used,
        }

    def _xgb_predict(self, features: dict) -> float:
        try:
            X    = extract_xgb_features(features)
            pred = self.loader.xgb.predict(X)[0]
            pred = float(np.expm1(pred))
            cpu_scaler = self.loader.cpu_scaler
            if cpu_scaler:
                pred = float(cpu_scaler.inverse_transform([[pred]])[0][0])
            pred = pred * 100.0
            return float(np.clip(pred, 0.0, 100.0))
        except Exception as e:
            logger.warning(f"XGBoost predict failed: {e}")
            return float(features["cpu_raw"])

    def _lstm_predict(self, history: list[dict]) -> tuple[float, float]:
        try:
            cpu_series = [float(r["cpu"]) for r in history]
            n   = min(len(cpu_series), HISTORY_WINDOW)
            seq = cpu_series[-n:]
            if len(seq) < HISTORY_WINDOW:
                pad = [seq[0]] * (HISTORY_WINDOW - len(seq))
                seq = pad + seq
            seq_arr = np.array(seq, dtype=np.float32) / 100.0
            X       = np.repeat(seq_arr.reshape(1, HISTORY_WINDOW, 1), 15, axis=2)
            preds   = [float(self.loader.lstm(X, training=True).numpy()[0][0])
                       for _ in range(MC_DROPOUT_SAMPLES)]
            mean_pred = float(np.clip(np.mean(preds) * 100.0, 0.0, 100.0))
            std_pred  = max(float(np.std(preds) * 100.0), 1.0)
            return mean_pred, std_pred
        except Exception as e:
            logger.warning(f"LSTM predict failed: {e}")
            return float(features["cpu_raw"]), 5.0

    def _fallback(self, features: dict) -> dict:
        current = features["cpu_raw"]
        delta   = features["delta_1"]
        pred    = float(np.clip(current + delta * 0.5, 0.0, 100.0))
        return {
            "predicted_cpu":     round(pred, 2),
            "confidence":        0.3,
            "ci_lower":          round(max(0.0,   pred - 10.0), 2),
            "ci_upper":          round(min(100.0, pred + 10.0), 2),
            "spike_probability": 1.0 if pred > SPIKE_THRESHOLD else 0.0,
            "model_used":        "fallback",
        }


class PredictionPipeline:

    def __init__(self):
        self._lock        = threading.Lock()
        self._predictions : dict[str, dict] = {}
        self._loader      = ModelLoader()
        self._predictor   = Predictor(self._loader)
        self._thread      = threading.Thread(
            target=self._loop, daemon=True, name="PipelineThread",
        )

    def start(self):
        if not self._thread.is_alive():
            self._thread.start()
            logger.info("Prediction pipeline started.")

    def get_predictions(self) -> dict[str, dict]:
        with self._lock:
            return dict(self._predictions)

    def get_prediction(self, server_id: str) -> Optional[dict]:
        with self._lock:
            return self._predictions.get(server_id)

    def _loop(self):
        while True:
            try:
                self._run_once()
            except Exception as e:
                logger.error(f"Pipeline loop error: {e}")
                logger.error(traceback.format_exc())
            time.sleep(PREDICTION_INTERVAL)

    def _run_once(self):
        ts      = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        servers = sim.get_server_list()
        latest  = sim.get_latest()
        results = {}

        for s in servers:
            sid  = s["server_id"]
            name = s["name"]
            history  = sim.get_history(sid)
            if not history:
                continue
            features = build_features(history)
            if features is None:
                rec = latest.get(sid, {})
                results[sid] = {
                    "server_id": sid, "server_name": name,
                    "current_cpu": rec.get("cpu", 0.0),
                    "predicted_cpu": rec.get("cpu", 0.0),
                    "confidence": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
                    "spike_probability": 0.0, "features": {},
                    "model_used": "warming_up", "timestamp": ts,
                }
                continue

            pred = self._predictor.predict(history, features)
            results[sid] = {
                "server_id":         sid,
                "server_name":       name,
                "current_cpu":       features["cpu_raw"],
                "predicted_cpu":     pred["predicted_cpu"],
                "confidence":        pred["confidence"],
                "ci_lower":          pred["ci_lower"],
                "ci_upper":          pred["ci_upper"],
                "spike_probability": pred["spike_probability"],
                "features":          features,
                "model_used":        pred["model_used"],
                "timestamp":         ts,
            }

        with self._lock:
            self._predictions.update(results)


if __name__ == "__main__":
    print("CrashGuard AI — Pipeline Test")
    print("─" * 55)
    sim.start()
    pipeline = PredictionPipeline()
    pipeline.start()
    print("Warming up (30s)...\n")
    for cycle in range(15):
        time.sleep(4)
        preds = pipeline.get_predictions()
        if not preds:
            print("  Waiting...")
            continue
        print(f"\n── Cycle {cycle+1} ── {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC")
        print(f"  {'SERVER':<28} {'CURR':>6} {'PRED':>6} {'CONF':>6} {'SPIKE%':>7} {'MODEL'}")
        print(f"  {'─'*65}")
        for sid, p in preds.items():
            print(
                f"  {p['server_name']:<28} "
                f"{p['current_cpu']:>5.1f}% "
                f"{p['predicted_cpu']:>5.1f}% "
                f"{p['confidence']:>5.0%}  "
                f"{p['spike_probability']:>5.0%}   "
                f"{p['model_used']}"
            )
    print("\n✅ Pipeline test complete.")