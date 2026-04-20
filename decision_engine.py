"""
decision_engine.py — CrashGuard AI
Autonomous decision engine. Takes pipeline predictions and produces
one of 5 decisions per server every tick.

Decision Logic:
  SCALE     — predicted_cpu > 85% AND confidence > 0.7
  RESTART   — fluctuating pattern (rolling_std > 0.15 * MAX_CPU)
  ESCALATE  — 3+ spikes detected in last 10 minutes
  MONITOR   — predicted_cpu > 65%
  STABLE    — everything normal

Output per server (dict):
    server_id       : str
    server_name     : str
    current_cpu     : float
    predicted_cpu   : float
    confidence      : float
    spike_probability: float
    decision        : str    — SCALE | RESTART | ESCALATE | MONITOR | STABLE
    severity        : str    — CRITICAL | HIGH | MEDIUM | LOW | OK
    reason          : str    — human-readable explanation
    action          : str    — what the system is doing
    spike_count     : int    — spikes in last 10 minutes
    model_used      : str
    timestamp       : str
"""

import time
import threading
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("crashguard.decision")

# ─────────────────────────────────────────────
# THRESHOLDS — derived from training data
# ─────────────────────────────────────────────
MAX_CPU = 95.0

SCALE_CPU_THRESHOLD     = 0.85 * MAX_CPU   # 80.75% — trigger SCALE
SCALE_CONF_THRESHOLD    = 0.70             # minimum confidence for SCALE
MONITOR_CPU_THRESHOLD   = 0.68 * MAX_CPU   # 64.6%  — trigger MONITOR
FLUCTUATING_STD         = 0.15 * MAX_CPU   # 14.25% — trigger RESTART
SPIKE_WINDOW_SECONDS    = 600              # 10 minutes
SPIKE_ESCALATE_COUNT    = 3               # spikes in window → ESCALATE
SPIKE_ESCALATE_RATE     = 0.3             # minimum spike_rate for ESCALATE
SPIKE_ESCALATE_CPU      = 80.0            # minimum current_cpu for ESCALATE
SPIKE_CPU_THRESHOLD     = 0.80 * MAX_CPU   # 76%    — what counts as a spike
ALERT_COOLDOWN_SECONDS  = 300             # 5 min cooldown per server
PREDICTION_INTERVAL     = 2               # seconds between predictions (matches pipeline)

# ─────────────────────────────────────────────
# DECISION CONSTANTS
# ─────────────────────────────────────────────
DECISIONS = {
    "SCALE":    {"severity": "CRITICAL", "color": "#FF3B3B"},
    "ESCALATE": {"severity": "CRITICAL", "color": "#FF3B3B"},
    "RESTART":  {"severity": "HIGH",     "color": "#FF8C00"},
    "MONITOR":  {"severity": "MEDIUM",   "color": "#FFD700"},
    "STABLE":   {"severity": "OK",       "color": "#00C851"},
}

# ─────────────────────────────────────────────
# SPIKE TRACKER — per server
# ─────────────────────────────────────────────

class SpikeTracker:
    """
    Tracks spike events per server in a rolling time window.
    Thread-safe.
    """
    def __init__(self, window_seconds: int = SPIKE_WINDOW_SECONDS):
        self._lock          = threading.Lock()
        self._window        = window_seconds
        # server_id → deque of UTC timestamps (float)
        self._spike_times: dict[str, deque] = {}

    def record_spike(self, server_id: str):
        now = time.time()
        with self._lock:
            if server_id not in self._spike_times:
                self._spike_times[server_id] = deque()
            self._spike_times[server_id].append(now)
            self._purge(server_id, now)

    def get_spike_count(self, server_id: str) -> int:
        now = time.time()
        with self._lock:
            if server_id not in self._spike_times:
                return 0
            self._purge(server_id, now)
            return len(self._spike_times[server_id])

    def _purge(self, server_id: str, now: float):
        """Remove spikes older than the window."""
        cutoff = now - self._window
        dq = self._spike_times[server_id]
        while dq and dq[0] < cutoff:
            dq.popleft()


# ─────────────────────────────────────────────
# COOLDOWN TRACKER — per server
# ─────────────────────────────────────────────

class CooldownTracker:
    """
    Prevents alert flooding — enforces per-server cooldown.
    Only HIGH and CRITICAL decisions trigger cooldown.
    """
    def __init__(self, cooldown_seconds: int = ALERT_COOLDOWN_SECONDS):
        self._lock      = threading.Lock()
        self._cooldown  = cooldown_seconds
        self._last_alert: dict[str, float] = {}

    def can_alert(self, server_id: str) -> bool:
        now = time.time()
        with self._lock:
            last = self._last_alert.get(server_id, 0.0)
            return (now - last) >= self._cooldown

    def record_alert(self, server_id: str):
        with self._lock:
            self._last_alert[server_id] = time.time()

    def seconds_until_next(self, server_id: str) -> int:
        now  = time.time()
        with self._lock:
            last = self._last_alert.get(server_id, 0.0)
            remaining = self._cooldown - (now - last)
            return max(0, int(remaining))


# ─────────────────────────────────────────────
# DECISION ENGINE
# ─────────────────────────────────────────────

class DecisionEngine:
    """
    Evaluates predictions from PredictionPipeline and produces
    autonomous decisions for all 5 servers.

    Usage:
        engine = DecisionEngine()
        decisions = engine.evaluate(predictions)   # call every tick
    """

    def __init__(self):
        self._spike_tracker    = SpikeTracker()
        self._cooldown_tracker = CooldownTracker()
        self._lock             = threading.Lock()
        self._last_decisions: dict[str, dict] = {}
        # Action feedback memory — per-server (cpu, decision) history
        self._action_history: dict[str, deque] = {}

    def evaluate(self, predictions: dict[str, dict]) -> dict[str, dict]:
        """
        Evaluate predictions for all servers.

        Args:
            predictions: output of PredictionPipeline.get_predictions()

        Returns:
            dict[server_id → decision_record]
        """
        results = {}
        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

        for sid, pred in predictions.items():
            if pred.get("model_used") in ("warming_up",):
                results[sid] = self._warming_up_record(pred, ts)
                continue

            decision_record = self._decide(sid, pred, ts)
            results[sid]    = decision_record

        with self._lock:
            self._last_decisions.update(results)

        return results

    def get_last_decisions(self) -> dict[str, dict]:
        """Thread-safe snapshot of last decisions."""
        with self._lock:
            return dict(self._last_decisions)

    def _decide(self, server_id: str, pred: dict, ts: str) -> dict:
        current_cpu   = pred.get("current_cpu",      0.0)
        predicted_cpu = pred.get("predicted_cpu",    0.0)
        confidence    = pred.get("confidence",       0.0)
        spike_prob    = pred.get("spike_probability",0.0)
        features      = pred.get("features",         {})
        server_name   = pred.get("server_name",      server_id)
        model_used    = pred.get("model_used",       "unknown")

        rolling_std   = features.get("rolling_std_10", 0.0)
        rolling_mean  = features.get("rolling_mean_10", 0.0)

        # Strong prediction correction — prevent unrealistic drops
        predicted_cpu = self._correct_prediction(current_cpu, predicted_cpu)

        # Model self-awareness — track prediction gap
        prediction_gap = abs(current_cpu - predicted_cpu)

        # Extreme disagreement — degrade confidence when model is unreliable
        if prediction_gap > 35:
            confidence *= 0.5

        # Derive trend from delta features
        delta_1 = features.get("delta_1", 0.0)
        delta_5 = features.get("delta_5", 0.0)
        trend   = self._trend_label(delta_1, delta_5)

        # Derive dynamic load state for explanations
        load_state = self._load_state(current_cpu)

        # Track spike events
        if current_cpu > SPIKE_CPU_THRESHOLD:
            self._spike_tracker.record_spike(server_id)

        spike_count = self._spike_tracker.get_spike_count(server_id)
        spike_rate  = spike_count / max(1, SPIKE_WINDOW_SECONDS // PREDICTION_INTERVAL) if spike_count > 0 else 0.0

        # Model disagreement annotation for explainability
        disagree_note = (f" Model disagreement: {prediction_gap:.1f}%."
                         if prediction_gap > 5.0 else "")

        # Action feedback memory — detect if previous SCALE failed
        scale_failed = False
        if server_id in self._action_history:
            hist = self._action_history[server_id]
            if len(hist) >= 1:
                prev_cpu, prev_decision = hist[-1]
                if prev_decision == "SCALE" and current_cpu >= prev_cpu:
                    scale_failed = True

        # ── Decision persistence — ESCALATE is sticky ──────────────
        last = self._last_decisions.get(server_id)
        if last and last.get("decision") == "ESCALATE":
            if current_cpu >= 60 or spike_count >= 1:
                decision = "ESCALATE"
                reason   = (f"System remains in escalated state — instability not yet resolved "
                            f"(CPU={current_cpu:.1f}% at {load_state}, "
                            f"{spike_count} spike(s) active){disagree_note}")
                action   = "Escalating to on-call engineer via PagerDuty"

        # ── Hard reality override (highest priority) ─────────────
        elif current_cpu >= 85:
            decision = "SCALE"
            reason   = (f"CPU {current_cpu:.1f}% at {load_state} — "
                        f"hard override triggered (≥85%){disagree_note}")
            action   = "Triggering auto-scaling — spinning up additional instances"

        elif current_cpu >= 75 and trend in ("rising", "rapidly_rising"):
            decision = "SCALE"
            reason   = (f"CPU {current_cpu:.1f}% at {load_state} with {trend} trend — "
                        f"preemptive scale triggered{disagree_note}")
            action   = "Triggering auto-scaling — spinning up additional instances"

        # ── Decision logic (priority order) ───────────────────────
        elif (
            (spike_count >= SPIKE_ESCALATE_COUNT and (current_cpu >= 80 or scale_failed))
            or (current_cpu > 85 and spike_prob > 0.7)
            or (prediction_gap > 40 and current_cpu > 75)
        ):
            decision = "ESCALATE"
            fail_note = (" Previous scaling action failed to reduce CPU load —"
                         " escalation required." if scale_failed else "")
            reason   = (f"Multi-signal escalation triggered "
                        f"(CPU={current_cpu:.1f}% at {load_state}, "
                        f"spikes={spike_count}, risk={spike_prob:.0%}, "
                        f"disagreement={prediction_gap:.1f}%) — "
                        f"sustained instability beyond autonomous recovery capacity"
                        f"{disagree_note}{fail_note}")
            action   = "Escalating to on-call engineer via PagerDuty"

        elif predicted_cpu > SCALE_CPU_THRESHOLD and confidence > SCALE_CONF_THRESHOLD:
            decision = "SCALE"
            reason   = (f"Predicted CPU {predicted_cpu:.1f}% exceeds threshold "
                        f"{SCALE_CPU_THRESHOLD:.0f}% with {confidence:.0%} confidence — "
                        f"current {load_state}{disagree_note}")
            action   = "Triggering auto-scaling — spinning up additional instances"

        elif rolling_std > FLUCTUATING_STD:
            decision = "RESTART"
            reason   = (f"CPU volatility (std={rolling_std:.1f}%) exceeds "
                        f"threshold {FLUCTUATING_STD:.0f}% — unstable process detected "
                        f"at {load_state}")
            action   = "Scheduling graceful service restart to clear instability"

        elif prediction_gap > 40 and current_cpu > 70:
            decision = "SCALE"
            reason   = (f"CPU {current_cpu:.1f}% at {load_state} with extreme model "
                        f"disagreement ({prediction_gap:.1f}%) — "
                        f"high model disagreement detected, prediction reliability degraded{disagree_note}")
            action   = "Triggering auto-scaling — conservative action under unreliable prediction"

        elif predicted_cpu > MONITOR_CPU_THRESHOLD:
            decision = "MONITOR"
            reason   = (f"Predicted CPU {predicted_cpu:.1f}% above monitoring "
                        f"threshold {MONITOR_CPU_THRESHOLD:.0f}% — {load_state}{disagree_note}")
            action   = "Increasing monitoring frequency — watching for escalation"

        elif current_cpu >= 60:
            decision = "MONITOR"
            reason   = (f"CPU {current_cpu:.1f}% at {load_state} — "
                        f"proactive monitoring engaged{disagree_note}")
            action   = "Increasing monitoring frequency — watching for escalation"

        else:
            decision = "STABLE"
            reason   = (f"CPU {current_cpu:.1f}% at {load_state} — "
                        f"predicted {predicted_cpu:.1f}% with {confidence:.0%} confidence")
            action   = "No action required — system operating normally"

        severity  = DECISIONS[decision]["severity"]
        color     = DECISIONS[decision]["color"]

        # Track cooldown for alertable decisions
        alert_ready = False
        if severity in ("CRITICAL", "HIGH"):
            alert_ready = self._cooldown_tracker.can_alert(server_id)
            if alert_ready:
                self._cooldown_tracker.record_alert(server_id)

        cooldown_remaining = self._cooldown_tracker.seconds_until_next(server_id)

        # Record action feedback memory
        if server_id not in self._action_history:
            self._action_history[server_id] = deque(maxlen=3)
        self._action_history[server_id].append((current_cpu, decision))

        return {
            "server_id":           server_id,
            "server_name":         server_name,
            "current_cpu":         round(current_cpu,   2),
            "predicted_cpu":       round(predicted_cpu, 2),
            "confidence":          round(confidence,    4),
            "spike_probability":   round(spike_prob,    4),
            "prediction_gap":      round(prediction_gap, 2),
            "decision":            decision,
            "severity":            severity,
            "color":               color,
            "reason":              reason,
            "action":              action,
            "spike_count":         spike_count,
            "scale_failed":        scale_failed,
            "rolling_std":         round(rolling_std,   2),
            "rolling_mean":        round(rolling_mean,  2),
            "alert_ready":         alert_ready,
            "cooldown_remaining":  cooldown_remaining,
            "model_used":          model_used,
            "timestamp":           ts,
        }

    @staticmethod
    def _correct_prediction(current_cpu: float, raw_prediction: float) -> float:
        """Prevent unrealistic prediction drops when CPU is high."""
        if current_cpu >= 80 and raw_prediction < current_cpu - 15:
            return current_cpu - 10
        return raw_prediction

    @staticmethod
    def _trend_label(delta_1: float, delta_5: float) -> str:
        """Derive trend label from delta features."""
        if delta_1 > 3.0 and delta_5 > 5.0:
            return "rapidly_rising"
        elif delta_1 > 1.0:
            return "rising"
        elif delta_1 < -3.0 and delta_5 < -5.0:
            return "rapidly_falling"
        elif delta_1 < -1.0:
            return "falling"
        return "stable"

    @staticmethod
    def _load_state(current_cpu: float) -> str:
        """Dynamic load state for accurate explanations."""
        if current_cpu >= 80:
            return "critical load"
        elif current_cpu >= 65:
            return "elevated load"
        return "normal load"

    def _warming_up_record(self, pred: dict, ts: str) -> dict:
        return {
            "server_id":          pred.get("server_id",   ""),
            "server_name":        pred.get("server_name", ""),
            "current_cpu":        pred.get("current_cpu", 0.0),
            "predicted_cpu":      pred.get("current_cpu", 0.0),
            "confidence":         0.0,
            "spike_probability":  0.0,
            "decision":           "STABLE",
            "severity":           "OK",
            "color":              "#00C851",
            "reason":             "System warming up — collecting baseline data",
            "action":             "Waiting for sufficient history",
            "spike_count":        0,
            "rolling_std":        0.0,
            "rolling_mean":       0.0,
            "alert_ready":        False,
            "cooldown_remaining": 0,
            "model_used":         "warming_up",
            "timestamp":          ts,
        }


# ─────────────────────────────────────────────
# STANDALONE TEST — python decision_engine.py
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import server_simulator as sim
    from pipeline import PredictionPipeline

    print("CrashGuard AI — Decision Engine Test")
    print("─" * 60)

    sim.start()
    pipeline = PredictionPipeline()
    pipeline.start()
    engine   = DecisionEngine()

    print("Warming up (90s)...\n")
    time.sleep(90)

    SEVERITY_ICON = {
        "CRITICAL": "🔴",
        "HIGH":     "🟠",
        "MEDIUM":   "🟡",
        "LOW":      "🔵",
        "OK":       "🟢",
    }

    for cycle in range(10):
        time.sleep(4)
        predictions = pipeline.get_predictions()
        if not predictions:
            print("  Waiting for predictions...")
            continue

        decisions = engine.evaluate(predictions)

        print(f"\n── Cycle {cycle+1} ── {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC")
        print(f"  {'SERVER':<28} {'CPU':>6} {'PRED':>6} {'DECISION':<10} {'SEVERITY':<10} {'SPIKES'}")
        print(f"  {'─'*75}")

        for sid, d in decisions.items():
            icon = SEVERITY_ICON.get(d["severity"], "⚪")
            print(
                f"  {d['server_name']:<28} "
                f"{d['current_cpu']:>5.1f}% "
                f"{d['predicted_cpu']:>5.1f}% "
                f"{icon} {d['decision']:<8}  "
                f"{d['severity']:<10} "
                f"{d['spike_count']}"
            )
            print(f"    → {d['reason']}")
            print(f"    ⚙ {d['action']}")

    print("\n✅ Decision engine test complete.")
