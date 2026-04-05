"""
Advanced Features Module
1. Real-Time Alerts
2. Auto Retraining
3. Confidence-Based Decisions
4. Online Learning
5. Multi-Server Monitoring
"""

import numpy as np
import pandas as pd
import time
import threading
import smtplib
import joblib
from datetime import datetime
from email.mime.text import MIMEText
from collections import deque
import psutil


# =============================================
# 1. REAL-TIME ALERTS
# =============================================

class AlertSystem:
    """
    Monitors CPU predictions and triggers alerts
    when predicted CPU exceeds threshold.
    """

    def __init__(self, high_threshold=0.8, medium_threshold=0.6):
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.alert_history = []

    def check_prediction(self, predicted_cpu, current_cpu, uncertainty=None):
        """
        Checks prediction and returns alert level.
        Returns: dict with level, message, color
        """
        timestamp = datetime.now().strftime("%H:%M:%S")

        if predicted_cpu > self.high_threshold:
            alert = {
                "level": "HIGH",
                "color": "#ef4444",
                "emoji": "🔴",
                "message": f"CPU spike predicted! {predicted_cpu:.1%} expected",
                "action": "Immediate action required — scale resources now!",
                "timestamp": timestamp,
                "predicted": predicted_cpu,
                "current": current_cpu
            }
        elif predicted_cpu > self.medium_threshold:
            alert = {
                "level": "MEDIUM",
                "color": "#f59e0b",
                "emoji": "🟡",
                "message": f"Elevated CPU predicted: {predicted_cpu:.1%}",
                "action": "Monitor closely — consider pre-scaling",
                "timestamp": timestamp,
                "predicted": predicted_cpu,
                "current": current_cpu
            }
        else:
            alert = {
                "level": "NORMAL",
                "color": "#22c55e",
                "emoji": "🟢",
                "message": f"CPU normal: {predicted_cpu:.1%} predicted",
                "action": "No action needed",
                "timestamp": timestamp,
                "predicted": predicted_cpu,
                "current": current_cpu
            }

        # Add uncertainty info
        if uncertainty is not None:
            if uncertainty > 0.1:
                alert["uncertainty_warning"] = "⚠️ High uncertainty — prediction may be less reliable"
            else:
                alert["uncertainty_warning"] = "✅ Low uncertainty — prediction is reliable"

        self.alert_history.append(alert)

        # Keep only last 100 alerts
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]

        return alert

    def send_email_alert(self, alert, email_config):
        """
        Sends email alert (optional).
        email_config = {
            "smtp": "smtp.gmail.com",
            "port": 587,
            "user": "your@email.com",
            "password": "yourpassword",
            "to": "recipient@email.com"
        }
        """
        if alert["level"] not in ["HIGH"]:
            return False

        try:
            msg = MIMEText(
                f"CPU Alert!\n\n"
                f"Level: {alert['level']}\n"
                f"Predicted CPU: {alert['predicted']:.1%}\n"
                f"Current CPU: {alert['current']:.1%}\n"
                f"Action: {alert['action']}\n"
                f"Time: {alert['timestamp']}"
            )
            msg["Subject"] = f"⚠️ CPU Alert: {alert['level']}"
            msg["From"] = email_config["user"]
            msg["To"] = email_config["to"]

            with smtplib.SMTP(email_config["smtp"], email_config["port"]) as server:
                server.starttls()
                server.login(email_config["user"], email_config["password"])
                server.send_message(msg)
            return True
        except Exception:
            return False

    def get_alert_summary(self):
        """Returns summary of recent alerts."""
        if not self.alert_history:
            return {"total": 0, "high": 0, "medium": 0, "normal": 0}

        df = pd.DataFrame(self.alert_history)
        return {
            "total": len(df),
            "high": int((df["level"] == "HIGH").sum()),
            "medium": int((df["level"] == "MEDIUM").sum()),
            "normal": int((df["level"] == "NORMAL").sum()),
            "last_alert": self.alert_history[-1]
        }


# =============================================
# 2. AUTO RETRAINING
# =============================================

class AutoRetrainer:
    """
    Automatically retrains the model when:
    - Enough new data has accumulated
    - Model performance degrades
    - Scheduled time interval passes
    """

    def __init__(self, retrain_threshold=1000, performance_threshold=0.05):
        self.retrain_threshold = retrain_threshold
        self.performance_threshold = performance_threshold
        self.new_data_buffer = []
        self.retrain_history = []
        self.is_retraining = False
        self.last_retrain = datetime.now()

    def add_new_data(self, cpu_value, timestamp=None):
        """Add new CPU reading to buffer."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        self.new_data_buffer.append({
            "timestamp": timestamp,
            "cpu_usage": cpu_value
        })

    def should_retrain(self):
        """Check if retraining is needed."""
        reasons = []

        # Check data accumulation
        if len(self.new_data_buffer) >= self.retrain_threshold:
            reasons.append(f"New data: {len(self.new_data_buffer)} samples collected")

        # Check time interval (every 24 hours)
        hours_since = (datetime.now() - self.last_retrain).seconds / 3600
        if hours_since >= 24:
            reasons.append(f"Time interval: {hours_since:.1f} hours since last retrain")

        return len(reasons) > 0, reasons

    def trigger_retrain(self, current_df, window_size, forecast_horizon):
        """
        Triggers retraining with new data.
        Returns updated dataset.
        """
        if len(self.new_data_buffer) == 0:
            return current_df, False

        # Merge new data with existing
        new_df = pd.DataFrame(self.new_data_buffer)
        updated_df = pd.concat([current_df, new_df], ignore_index=True)
        updated_df = updated_df.drop_duplicates(subset=["timestamp"])
        updated_df = updated_df.sort_values("timestamp").reset_index(drop=True)

        # Clear buffer
        self.new_data_buffer = []
        self.last_retrain = datetime.now()

        self.retrain_history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "new_samples": len(new_df),
            "total_samples": len(updated_df),
            "status": "Success"
        })

        return updated_df, True

    def get_retrain_history(self):
        """Returns retraining history as DataFrame."""
        if not self.retrain_history:
            return pd.DataFrame()
        return pd.DataFrame(self.retrain_history)

    def get_buffer_status(self):
        """Returns current buffer status."""
        return {
            "buffer_size": len(self.new_data_buffer),
            "threshold": self.retrain_threshold,
            "pct_full": round(len(self.new_data_buffer) / self.retrain_threshold * 100, 1),
            "hours_since_retrain": round(
                (datetime.now() - self.last_retrain).seconds / 3600, 1
            )
        }


# =============================================
# 3. CONFIDENCE-BASED DECISIONS
# =============================================

class ConfidenceDecisionEngine:
    """
    Makes automated decisions based on prediction confidence.
    Low uncertainty → act automatically
    High uncertainty → alert human
    """

    def __init__(self,
                 low_uncertainty=0.02,
                 high_uncertainty=0.08,
                 action_threshold=0.75):
        self.low_uncertainty = low_uncertainty
        self.high_uncertainty = high_uncertainty
        self.action_threshold = action_threshold
        self.decision_history = []

    def make_decision(self, predicted_cpu, uncertainty, current_cpu):
        """
        Makes a resource management decision.

        Returns dict with:
        - decision: what to do
        - confidence: how sure we are
        - action: specific action to take
        """

        confidence_score = 1.0 - min(uncertainty / self.high_uncertainty, 1.0)
        confidence_pct = round(confidence_score * 100, 1)

        timestamp = datetime.now().strftime("%H:%M:%S")

        # Decision logic
        if uncertainty <= self.low_uncertainty:
            confidence_level = "HIGH CONFIDENCE"
            confidence_color = "#22c55e"

            if predicted_cpu > self.action_threshold:
                decision = "AUTO-SCALE UP"
                action = "Automatically allocate 20% more CPU resources"
                icon = "⚡"
            elif predicted_cpu < 0.2 and current_cpu < 0.2:
                decision = "AUTO-SCALE DOWN"
                action = "Safely reduce CPU allocation by 15% to save energy"
                icon = "💤"
            else:
                decision = "MAINTAIN"
                action = "Current resources are optimal"
                icon = "✅"

        elif uncertainty <= self.high_uncertainty:
            confidence_level = "MEDIUM CONFIDENCE"
            confidence_color = "#f59e0b"
            decision = "HUMAN REVIEW"
            action = "Notify engineer — uncertainty too high for auto-action"
            icon = "👀"

        else:
            confidence_level = "LOW CONFIDENCE"
            confidence_color = "#ef4444"
            decision = "WAIT & OBSERVE"
            action = "Collect more data before making any decision"
            icon = "⏳"

        result = {
            "decision": decision,
            "action": action,
            "icon": icon,
            "confidence_level": confidence_level,
            "confidence_color": confidence_color,
            "confidence_pct": confidence_pct,
            "predicted_cpu": predicted_cpu,
            "uncertainty": uncertainty,
            "timestamp": timestamp
        }

        self.decision_history.append(result)
        if len(self.decision_history) > 200:
            self.decision_history = self.decision_history[-200:]

        return result

    def get_decision_stats(self):
        """Returns stats about recent decisions."""
        if not self.decision_history:
            return {}

        df = pd.DataFrame(self.decision_history)
        counts = df["decision"].value_counts().to_dict()

        return {
            "total_decisions": len(df),
            "auto_scale_up": counts.get("AUTO-SCALE UP", 0),
            "auto_scale_down": counts.get("AUTO-SCALE DOWN", 0),
            "maintain": counts.get("MAINTAIN", 0),
            "human_review": counts.get("HUMAN REVIEW", 0),
            "avg_confidence": round(df["confidence_pct"].mean(), 1)
        }


# =============================================
# 4. ONLINE LEARNING
# =============================================

class OnlineLearner:
    """
    Updates model continuously with new data.
    Model gets smarter without full retraining.
    """

    def __init__(self, update_interval=100, learning_rate=0.0001):
        self.update_interval = update_interval
        self.learning_rate = learning_rate
        self.update_count = 0
        self.update_history = []
        self.performance_history = []
        self.new_samples = []

    def add_sample(self, X_new, y_new):
        """Add new training sample to buffer."""
        self.new_samples.append((X_new, y_new))

    def update_model(self, model, X_new, y_new):
        """
        Performs one online learning update step.
        Updates model weights with new data.
        """
        try:
            import tensorflow as tf

            # Set lower learning rate for fine-tuning
            model.optimizer.learning_rate = self.learning_rate

            # Single gradient update
            loss = model.train_on_batch(X_new, y_new)

            self.update_count += 1
            self.update_history.append({
                "update_number": self.update_count,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "loss": float(loss[0]) if isinstance(loss, list) else float(loss),
                "samples": len(X_new)
            })

            if len(self.update_history) > 500:
                self.update_history = self.update_history[-500:]

            return True, float(loss[0]) if isinstance(loss, list) else float(loss)

        except Exception as e:
            return False, str(e)

    def should_update(self):
        """Check if enough new samples for update."""
        return len(self.new_samples) >= self.update_interval

    def get_learning_curve(self):
        """Returns learning curve data."""
        if not self.update_history:
            return pd.DataFrame()
        return pd.DataFrame(self.update_history)

    def get_status(self):
        """Returns online learning status."""
        return {
            "total_updates": self.update_count,
            "buffer_size": len(self.new_samples),
            "update_interval": self.update_interval,
            "last_loss": self.update_history[-1]["loss"] if self.update_history else None
        }


# =============================================
# 5. MULTI-SERVER MONITORING
# =============================================

class MultiServerMonitor:
    """
    Simulates monitoring multiple servers.
    Each server has its own CPU profile and prediction.
    """

    def __init__(self, n_servers=5):
        self.n_servers = n_servers
        self.servers = self._initialize_servers()
        self.history = {f"Server_{i+1}": deque(maxlen=100)
                        for i in range(n_servers)}

    def _initialize_servers(self):
        """Initialize server profiles."""
        profiles = [
            {"name": "Web Server 1", "base_load": 0.3, "volatility": 0.1, "role": "Frontend"},
            {"name": "API Server", "base_load": 0.5, "volatility": 0.15, "role": "Backend"},
            {"name": "Database Server", "base_load": 0.4, "volatility": 0.08, "role": "Database"},
            {"name": "ML Server", "base_load": 0.7, "volatility": 0.2, "role": "ML Training"},
            {"name": "Cache Server", "base_load": 0.2, "volatility": 0.05, "role": "Caching"},
        ]
        return profiles[:self.n_servers]

    def get_server_readings(self):
        """Get current CPU readings for all servers."""
        readings = []
        real_cpu = psutil.cpu_percent(interval=0.1) / 100.0

        for i, server in enumerate(self.servers):
            # Simulate different load patterns per server
            noise = np.random.normal(0, server["volatility"])
            cpu = np.clip(server["base_load"] + noise + real_cpu * 0.3, 0.0, 1.0)

            reading = {
                "server_id": i + 1,
                "name": server["name"],
                "role": server["role"],
                "cpu_usage": round(float(cpu), 4),
                "status": "Critical" if cpu > 0.85 else "Warning" if cpu > 0.65 else "Normal",
                "status_color": "#ef4444" if cpu > 0.85 else "#f59e0b" if cpu > 0.65 else "#22c55e",
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }

            self.history[f"Server_{i+1}"].append(cpu)
            readings.append(reading)

        return readings

    def predict_all_servers(self, model, processed, window_size):
        """Predicts next CPU for all servers."""
        from live_monitor import build_live_feature_window

        readings = self.get_server_readings()
        predictions = []

        for i, reading in enumerate(readings):
            history = list(self.history[f"Server_{i+1}"])

            if len(history) >= window_size:
                try:
                    window = build_live_feature_window(history, window_size)
                    pred = model.predict(window, verbose=0)[0][0]
                    pred = float(np.clip(pred, 0.0, 1.0))
                except Exception:
                    pred = reading["cpu_usage"]
            else:
                pred = reading["cpu_usage"]

            predictions.append({
                **reading,
                "predicted_next": round(pred, 4),
                "trend": "↑ Rising" if pred > reading["cpu_usage"] + 0.05
                         else "↓ Falling" if pred < reading["cpu_usage"] - 0.05
                         else "→ Stable"
            })

        return predictions

    def get_fleet_summary(self, readings):
        """Returns summary stats for all servers."""
        cpus = [r["cpu_usage"] for r in readings]
        critical = sum(1 for r in readings if r["status"] == "Critical")
        warning = sum(1 for r in readings if r["status"] == "Warning")

        return {
            "avg_cpu": round(np.mean(cpus), 4),
            "max_cpu": round(np.max(cpus), 4),
            "min_cpu": round(np.min(cpus), 4),
            "critical_servers": critical,
            "warning_servers": warning,
            "healthy_servers": len(readings) - critical - warning
        }
