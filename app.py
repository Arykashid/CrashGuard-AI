"""
app.py — CrashGuard AI
Flask backend. Starts simulator, pipeline, decision engine, alert system.
Serves the dashboard and exposes REST API endpoints.

Fixes applied:
  [1] Demo trigger uses public sim.inject_reading() — no private access
  [2] Alert processing moved to background worker thread
  [3] All API endpoints wrapped in try/except — graceful error JSON

Endpoints:
  GET  /                    → dashboard.html
  GET  /api/status          → all server predictions + decisions
  GET  /api/history/<sid>   → CPU history for one server
  GET  /api/servers         → server list
  GET  /api/alerts          → alert stats
  GET  /health              → health check
  POST /api/demo/trigger    → inject a CPU spike for demo
"""

import os
import logging
import threading
import time
from datetime import datetime, timezone
from flask import Flask, jsonify, request, send_from_directory

import server_simulator as sim
from pipeline        import PredictionPipeline
from decision_engine import DecisionEngine
from alert_system    import AlertSystem

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("crashguard.app")

# ─────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────
app = Flask(__name__, static_folder="assets", static_url_path="/static", template_folder=".")

# ─────────────────────────────────────────────
# GLOBAL COMPONENTS
# ─────────────────────────────────────────────
pipeline  = None
engine    = None
alerts    = None
_started  = False
_init_lock = threading.Lock()


def initialize():
    """Start all background components. Called once at first request."""
    global pipeline, engine, alerts, _started
    with _init_lock:
        if _started:
            return
        logger.info("Initializing CrashGuard AI components...")
        sim.start()
        pipeline = PredictionPipeline()
        pipeline.start()
        engine  = DecisionEngine()
        alerts  = AlertSystem()
        # FIX 2 — start background alert worker
        _start_alert_worker()
        _started = True
        logger.info("All components started.")


# ─────────────────────────────────────────────
# FIX 2 — BACKGROUND ALERT WORKER
# Decouples alert processing from request cycle.
# Runs every 2 seconds independently of API calls.
# ─────────────────────────────────────────────

def _alert_worker():
    """Background thread — processes alerts every 2 seconds."""
    while True:
        try:
            if pipeline and engine and alerts:
                predictions = pipeline.get_predictions()
                if predictions:
                    decisions = engine.evaluate(predictions)
                    sent      = alerts.process_decisions(decisions)
                    if sent:
                        logger.info(f"Alert worker fired {len(sent)} alert(s).")
        except Exception as e:
            logger.error(f"Alert worker error: {e}")
        time.sleep(2)


def _start_alert_worker():
    t = threading.Thread(target=_alert_worker, daemon=True, name="AlertWorker")
    t.start()
    logger.info("Alert worker started.")


@app.before_request
def ensure_initialized():
    initialize()


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main dashboard."""
    return send_from_directory(".", "crashguard_dashboard.html")


@app.route("/api/status")
def api_status():
    """
    FIX 3 — wrapped in try/except.
    Returns latest predictions + decisions for all 5 servers.
    Called by dashboard.js every 2 seconds.
    Alert processing is now handled by background worker — not here.
    """
    try:
        predictions = pipeline.get_predictions()
        decisions   = engine.evaluate(predictions)

        servers = []
        for sid, d in decisions.items():
            pred = predictions.get(sid, {})
            servers.append({
                "server_id":               d["server_id"],
                "server_name":             d["server_name"],
                "current_cpu":             d["current_cpu"],
                "predicted_cpu":           d["predicted_cpu"],
                "confidence":              d["confidence"],
                "adjusted_confidence":     d.get("adjusted_confidence", d["confidence"]),
                "action_confidence":       d.get("action_confidence", 0.0),
                "spike_probability":       d["spike_probability"],
                "crash_risk_5min":         d.get("crash_risk_5min", 0.0),
                "ci_lower":               pred.get("ci_lower", 0.0),
                "ci_upper":               pred.get("ci_upper", 0.0),
                "decision":               d["decision"],
                "severity":               d["severity"],
                "color":                  d["color"],
                "reason":                 d["reason"],
                "action":                 d["action"],
                "risk_score":             d.get("risk_score", 0.0),
                "trend":                  d.get("trend", "stable"),
                "spike_count":            d["spike_count"],
                "spike_rate":             d.get("spike_rate", 0.0),
                "rolling_std":            d["rolling_std"],
                "rolling_mean":           d["rolling_mean"],
                "model_used":             d["model_used"],
                "prediction_disagreement": d.get("prediction_disagreement", 0.0),
                "model_reliability":      d.get("model_reliability", 1.0),
                "alert_ready":            d.get("alert_ready", False),
                "incident_id":            d.get("incident_id"),
                "incident_transition":    d.get("incident_transition"),
                "timestamp":              d["timestamp"],
            })

        eval_metrics = engine.get_eval_metrics()

        return jsonify({
            "servers":      servers,
            "eval_metrics": eval_metrics,
            "timestamp":    datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "status":       "ok",
        })

    except Exception as e:
        logger.error(f"/api/status error: {e}")
        return jsonify({
            "error":     str(e),
            "status":    "error",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }), 500


@app.route("/api/history/<server_id>")
def api_history(server_id: str):
    """
    FIX 3 — wrapped in try/except.
    Returns last N CPU readings for a server.
    """
    try:
        n = request.args.get("n", 60, type=int)
        n = max(10, min(n, 150))
        history = sim.get_history(server_id, n=n)
        return jsonify({
            "server_id": server_id,
            "history":   history,
            "count":     len(history),
            "status":    "ok",
        })
    except Exception as e:
        logger.error(f"/api/history error: {e}")
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/api/servers")
def api_servers():
    """FIX 3 — wrapped in try/except."""
    try:
        return jsonify(sim.get_server_list())
    except Exception as e:
        logger.error(f"/api/servers error: {e}")
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/api/alerts")
def api_alerts():
    """FIX 3 — wrapped in try/except."""
    try:
        return jsonify(alerts.get_stats())
    except Exception as e:
        logger.error(f"/api/alerts error: {e}")
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/api/demo/trigger", methods=["POST"])
def api_demo_trigger():
    """
    FIX 1 — uses public sim.inject_reading() instead of sim._lock/_histories.
    FIX 3 — wrapped in try/except.
    Injects a CPU spike for demo purposes.
    Body: { "server_id": "server_c", "cpu": 92.0 }
    """
    try:
        data      = request.get_json(force=True, silent=True) or {}
        server_id = data.get("server_id", "server_c")
        cpu_value = float(data.get("cpu", 92.0))
        cpu_value = max(20.0, min(99.0, cpu_value))

        # FIX 1 — public API, no private state access
        success = sim.inject_reading(server_id, cpu_value)

        if success:
            logger.info(f"Demo trigger: {server_id} → {cpu_value}%")
            return jsonify({
                "success":   True,
                "server_id": server_id,
                "cpu":       cpu_value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        else:
            return jsonify({
                "success": False,
                "error":   f"Unknown server_id: {server_id}",
            }), 400

    except Exception as e:
        logger.error(f"/api/demo/trigger error: {e}")
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/api/metrics")
def api_metrics():
    """Returns decision evaluation metrics."""
    try:
        return jsonify(engine.get_eval_metrics())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    """FIX 3 — wrapped in try/except."""
    try:
        return jsonify({
            "status":    "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {
                "simulator": sim._thread.is_alive(),
                "pipeline":  pipeline._thread.is_alive() if pipeline else False,
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  CrashGuard AI — Starting Server")
    print("=" * 60)
    print(f"  Dashboard: http://localhost:5000")
    print(f"  API:       http://localhost:5000/api/status")
    print(f"  Health:    http://localhost:5000/health")
    print()

    if os.getenv("DEMO_MODE") == "1":
        print("  Mode: DEMO (seed=42)")
    else:
        print("  Mode: LIVE (random)")

    if os.getenv("SLACK_WEBHOOK_URL"):
        print("  Slack: CONFIGURED ✅")
    else:
        print("  Slack: DRY-RUN (set SLACK_WEBHOOK_URL to enable)")

    print("=" * 60)
    print()

    initialize()
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
    )
