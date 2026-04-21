"""
decision_engine.py — CrashGuard AI  v3 (Production Grade)
Autonomous decision engine with prediction consistency layer,
spike intelligence, trend awareness, and intelligent explanations.

Key improvements over v1:
  [1] Prediction consistency layer — corrects contradictions between
      current state and raw model output before making decisions
  [2] Spike probability computed from observed behavior, not model output
  [3] Trend-aware decisions — rising vs falling treated differently
  [4] Intelligent explanation engine — context-aware, not template strings
  [5] Combined signal scoring — avoids pure threshold feel

Key improvements over v2:
  [6] Decision graduation — 3 spikes→RESTART, 10 spikes→ESCALATE
  [7] Hard floor for critical servers (CPU > 70% → pred floor at -3%)
  [8] Uncertainty intelligence in explanations

Key improvements over v3:
  [9]  prediction_disagreement + model_reliability fields
  [10] Pattern-aware ESCALATE — spike_count + spike_rate + trend
  [11] Adaptive risk weights based on system state
  [12] Hysteresis — decision memory prevents flapping
  [13] Adjusted confidence — penalised when disagreement is high

Output per server:
    server_id, server_name, current_cpu, predicted_cpu (corrected),
    confidence, spike_probability (corrected), decision, severity,
    reason, action, spike_count, trend, rolling_std, rolling_mean,
    alert_ready, cooldown_remaining, model_used, timestamp
"""

import math
import time
import threading
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("crashguard.decision")

# ─────────────────────────────────────────────
# CONSTANTS — derived from Google Cluster training data
# ─────────────────────────────────────────────
MAX_CPU = 95.0

# Decision thresholds
SCALE_CPU_THRESHOLD    = 0.85 * MAX_CPU   # 80.75%
SCALE_CONF_THRESHOLD   = 0.65             # lowered: model predictions are imperfect
MONITOR_CPU_THRESHOLD  = 0.68 * MAX_CPU   # 64.6%
RESTART_STD_THRESHOLD  = 0.14 * MAX_CPU   # 13.3% — high volatility
SPIKE_CPU_THRESHOLD    = 0.80 * MAX_CPU   # 76% — counts as a spike event
SPIKE_RESTART_COUNT    = 3               # spikes in window → RESTART (graceful)
SPIKE_ESCALATE_COUNT   = 10              # spikes in window → ESCALATE (critical)
SPIKE_WINDOW_SECONDS   = 600             # 10-minute rolling window
ALERT_COOLDOWN_SECONDS = 300             # 5-minute per-server cooldown

# Prediction correction bounds
MAX_PRED_DROP_STABLE   = 15.0   # max allowed prediction drop for stable server
MAX_PRED_DROP_CRITICAL = 5.0    # max allowed prediction drop for unstable server
TREND_BIAS_FACTOR      = 0.35   # how strongly trend biases corrected prediction

DECISIONS = {
    "SCALE":    {"severity": "CRITICAL", "color": "#FF3B3B"},
    "ESCALATE": {"severity": "CRITICAL", "color": "#FF3B3B"},
    "RESTART":  {"severity": "HIGH",     "color": "#FF8C00"},
    "MONITOR":  {"severity": "MEDIUM",   "color": "#FFD700"},
    "STABLE":   {"severity": "OK",       "color": "#00C851"},
}


# ─────────────────────────────────────────────
# SPIKE TRACKER
# ─────────────────────────────────────────────

class SpikeTracker:
    """Rolling window spike counter per server. Thread-safe."""

    def __init__(self, window_seconds: int = SPIKE_WINDOW_SECONDS):
        self._lock   = threading.Lock()
        self._window = window_seconds
        self._times: dict[str, deque] = {}

    def record(self, server_id: str):
        now = time.time()
        with self._lock:
            if server_id not in self._times:
                self._times[server_id] = deque()
            self._times[server_id].append(now)
            self._purge(server_id, now)

    def count(self, server_id: str) -> int:
        now = time.time()
        with self._lock:
            if server_id not in self._times:
                return 0
            self._purge(server_id, now)
            return len(self._times[server_id])

    def spike_rate(self, server_id: str) -> float:
        """Spikes per minute in the current window."""
        n = self.count(server_id)
        return n / (self._window / 60.0)

    def _purge(self, server_id: str, now: float):
        cutoff = now - self._window
        dq = self._times[server_id]
        while dq and dq[0] < cutoff:
            dq.popleft()


# ─────────────────────────────────────────────
# COOLDOWN TRACKER
# ─────────────────────────────────────────────

class CooldownTracker:
    """Per-server alert cooldown. Thread-safe."""

    def __init__(self, cooldown_seconds: int = ALERT_COOLDOWN_SECONDS):
        self._lock     = threading.Lock()
        self._cooldown = cooldown_seconds
        self._last: dict[str, float] = {}

    def can_alert(self, server_id: str) -> bool:
        with self._lock:
            return (time.time() - self._last.get(server_id, 0.0)) >= self._cooldown

    def record(self, server_id: str):
        with self._lock:
            self._last[server_id] = time.time()

    def seconds_remaining(self, server_id: str) -> int:
        with self._lock:
            remaining = self._cooldown - (time.time() - self._last.get(server_id, 0.0))
            return max(0, int(remaining))


# ─────────────────────────────────────────────
# MODULE 1 — PREDICTION CONSISTENCY LAYER
# ─────────────────────────────────────────────

def correct_prediction(
    current_cpu: float,
    raw_prediction: float,
    rolling_mean: float,
    rolling_std: float,
    spike_count: int,
    confidence: float,
) -> float:
    """
    FIX 1 — Corrects logically contradictory model predictions.

    Problems this fixes:
    - Server at 90% CPU predicted to drop to 20% (impossible without intervention)
    - Server stable at 25% predicted to jump to 75% (inconsistent with history)

    Logic:
    1. Compute trend direction from current vs rolling mean
    2. Apply trend bias to raw prediction
    3. Enforce consistency bounds based on system stability
    4. Blend corrected prediction with raw for smooth output
    """
    if current_cpu <= 0 or rolling_mean <= 0:
        return raw_prediction

    # FIX 2 — Clamp raw prediction before correction
    # Prevents nonsense like 82% CPU → 50% predicted drop
    if current_cpu >= 80 and raw_prediction < current_cpu - 15:
        raw_prediction = current_cpu - 10

    # Trend direction: positive = rising, negative = falling
    trend_delta = current_cpu - rolling_mean
    trend_factor = math.tanh(trend_delta / max(rolling_std, 1.0))  # -1 to +1

    # Trend-biased prediction: pull raw prediction toward trend direction
    trend_bias = trend_factor * TREND_BIAS_FACTOR * current_cpu
    trend_adjusted = raw_prediction + trend_bias

    # Consistency bounds based on instability
    is_unstable = spike_count >= SPIKE_ESCALATE_COUNT or current_cpu > SPIKE_CPU_THRESHOLD
    max_drop = MAX_PRED_DROP_CRITICAL if is_unstable else MAX_PRED_DROP_STABLE

    # Prediction cannot drop more than max_drop below current in unstable state
    if is_unstable:
        floor = current_cpu - max_drop
        trend_adjusted = max(trend_adjusted, floor)

    # Blend: weight toward trend-adjusted when confidence is low
    # High confidence → trust model more; low confidence → trust trend more
    blend_weight = min(max(confidence, 0.3), 0.9)
    corrected = blend_weight * trend_adjusted + (1 - blend_weight) * current_cpu

    # FIX 2 — Hard floor for critical servers (CPU > 70%)
    # Prevents any visible contradiction on dashboard during demo.
    # A server above 70% CPU cannot realistically drop more than 3% per tick.
    if current_cpu > 70.0:
        corrected = max(corrected, current_cpu - 3.0)

    return round(float(min(max(corrected, 0.0), 100.0)), 2)


# ─────────────────────────────────────────────
# MODULE 2 — SPIKE PROBABILITY SCORER
# ─────────────────────────────────────────────

def compute_spike_probability(
    current_cpu: float,
    corrected_pred: float,
    rolling_mean: float,
    rolling_std: float,
    spike_count: int,
    confidence: float,
) -> float:
    """
    FIX 2 — Computes spike probability from observed behavior signals.

    Raw pipeline spike_prob is unreliable (often 0.0 due to model mismatch).
    This replaces it with a signal-based computation.

    Signals used:
    - spike_count: historical frequency (most reliable)
    - current deviation from mean: how far above normal right now
    - predicted trajectory: is the corrected prediction going up?
    - rolling volatility: high std = unpredictable = higher risk
    """
    # Signal 1: historical spike frequency (0-1)
    freq_signal = min(spike_count / 10.0, 1.0)

    # Signal 2: current deviation from mean (how far above baseline)
    if rolling_std > 0:
        z_score = (current_cpu - rolling_mean) / rolling_std
        dev_signal = min(max(z_score / 3.0, 0.0), 1.0)  # 3-sigma → probability 1
    else:
        dev_signal = 1.0 if current_cpu > SPIKE_CPU_THRESHOLD else 0.0

    # Signal 3: predicted trajectory (is it heading up?)
    traj_signal = max((corrected_pred - rolling_mean) / max(rolling_mean, 1.0), 0.0)
    traj_signal = min(traj_signal * 2.0, 1.0)

    # Signal 4: volatility risk
    vol_signal = min(rolling_std / (MAX_CPU * 0.15), 1.0)

    # Weighted combination — freq and deviation are most reliable
    spike_prob = (
        0.40 * freq_signal +
        0.30 * dev_signal  +
        0.15 * traj_signal +
        0.15 * vol_signal
    )

    return round(min(max(spike_prob, 0.0), 1.0), 4)


# ─────────────────────────────────────────────
# MODULE 3 — TREND CLASSIFIER
# ─────────────────────────────────────────────

def classify_trend(
    current_cpu: float,
    rolling_mean: float,
    rolling_std: float,
    delta_1: float,
) -> str:
    """Classify current CPU trend for explanations and decisions."""
    if delta_1 > 4.0:
        return "rapidly_rising"
    elif delta_1 > 1.5:
        return "rising"
    elif delta_1 < -4.0:
        return "rapidly_falling"
    elif delta_1 < -1.5:
        return "falling"
    elif rolling_std > RESTART_STD_THRESHOLD:
        return "volatile"
    elif current_cpu > rolling_mean + rolling_std:
        return "elevated"
    else:
        return "stable"


# ─────────────────────────────────────────────
# MODULE 4 — EXPLANATION ENGINE
# ─────────────────────────────────────────────

def build_explanation(
    decision: str,
    current_cpu: float,
    corrected_pred: float,
    rolling_mean: float,
    rolling_std: float,
    spike_count: int,
    spike_prob: float,
    confidence: float,
    trend: str,
    spike_rate: float,
    load_state: str = "normal load",
) -> tuple[str, str]:
    """
    FIX 4 — Generates intelligent, context-aware explanations.
    Uses load_state to prevent "CPU 81% within normal range" lies.
    Returns (reason, action) specific to current server state.
    """
    trend_labels = {
        "rapidly_rising": "rapidly rising",
        "rising":         "trending upward",
        "rapidly_falling":"rapidly declining",
        "falling":        "declining",
        "volatile":       "highly volatile",
        "elevated":       "elevated above baseline",
        "stable":         "stable",
    }
    trend_str = trend_labels.get(trend, "stable")

    if decision == "ESCALATE":
        per_min = round(spike_rate, 1)
        if spike_count >= 15:
            reason = (
                f"Extreme instability under {load_state} — {spike_count} CPU spikes in 10 min "
                f"({per_min}/min rate). Current {current_cpu:.1f}% with {trend_str} trajectory "
                f"indicates systemic failure pattern, not transient load. "
                f"Prediction uncertainty increases under extreme conditions — "
                f"decision reinforced using behavioral signals and spike history."
            )
        elif spike_count >= 7:
            reason = (
                f"Persistent spike pattern ({spike_count} events, {per_min}/min) under {load_state}. "
                f"CPU {current_cpu:.1f}% is {trend_str} — automated recovery insufficient, "
                f"human intervention required. "
                f"Prediction uncertainty increases under extreme conditions — "
                f"decision reinforced using behavioral signals and spike history."
            )
        else:
            reason = (
                f"Repeated instability ({spike_count} spikes in 10 min) at {load_state} "
                f"({current_cpu:.1f}%, {trend_str}). Spike frequency ({per_min}/min) "
                f"indicates workload cannot self-stabilize. "
                f"Prediction uncertainty increases under extreme conditions — "
                f"decision reinforced using behavioral signals and spike history."
            )
        action = "Paging on-call engineer — SLA breach risk in next 5 minutes"

    elif decision == "SCALE":
        margin = corrected_pred - SCALE_CPU_THRESHOLD
        reason = (
            f"Server under {load_state} at {current_cpu:.1f}% ({trend_str}). "
            f"Predicted CPU {corrected_pred:.1f}% ({margin:+.1f}% above scale threshold) "
            f"at {confidence:.0%} model confidence. Rolling mean {rolling_mean:.1f}% — "
            f"sustained demand exceeds single-instance capacity. "
            f"Prediction uncertainty increases under extreme conditions — "
            f"decision reinforced using behavioral signals and spike history."
        )
        action = "Triggering horizontal autoscale — provisioning additional compute instances"

    elif decision == "RESTART":
        if spike_count >= SPIKE_RESTART_COUNT:
            reason = (
                f"Graduated escalation triggered under {load_state} — {spike_count} spikes "
                f"(threshold: {SPIKE_RESTART_COUNT}). CPU {current_cpu:.1f}% is {trend_str} "
                f"with mean {rolling_mean:.1f}%. Attempting graceful restart before escalating "
                f"to on-call — standard SRE first-response procedure."
            )
        else:
            reason = (
                f"CPU oscillating with std={rolling_std:.1f}% under {load_state} "
                f"(threshold {RESTART_STD_THRESHOLD:.0f}%, mean {rolling_mean:.1f}%). "
                f"Pattern is {trend_str} — high variance indicates memory pressure, "
                f"connection pool exhaustion, or runaway background process."
            )
        action = "Initiating graceful service restart to clear process state"

    elif decision == "MONITOR":
        reason = (
            f"Server under {load_state} at {current_cpu:.1f}% ({trend_str}). "
            f"Predicted CPU {corrected_pred:.1f}% approaching critical zone "
            f"(threshold {MONITOR_CPU_THRESHOLD:.0f}%) with {spike_prob:.0%} spike probability — "
            f"escalation likely if trend continues."
        )
        action = "Increasing telemetry frequency — pre-positioning scale trigger if CPU crosses 80%"

    else:  # STABLE
        if trend in ("rapidly_falling", "falling"):
            reason = (
                f"CPU {current_cpu:.1f}% declining from {load_state} — "
                f"workload normalizing. Predicted {corrected_pred:.1f}% with {confidence:.0%} "
                f"confidence. No intervention required."
            )
        else:
            reason = (
                f"CPU {current_cpu:.1f}% under {load_state} "
                f"(mean {rolling_mean:.1f}%, std {rolling_std:.1f}%). "
                f"Predicted {corrected_pred:.1f}% at {confidence:.0%} confidence — "
                f"no anomalous patterns detected."
            )
        action = "Continuous monitoring active — no action required"

    return reason, action


# ─────────────────────────────────────────────
# MODULE 5 — COMBINED SIGNAL SCORING
# ─────────────────────────────────────────────

def compute_risk_score(
    current_cpu: float,
    corrected_pred: float,
    rolling_mean: float,
    rolling_std: float,
    spike_count: int,
    spike_prob: float,
    confidence: float,
    trend: str,
) -> float:
    """
    FIX 11 — Adaptive weights based on system state.
    When spike_count > 5, spike burden weight increases from 0.25 → 0.35
    and CPU pressure decreases to compensate. This means:
    "Weights adapt based on system state — spike history dominates
     under sustained instability, CPU pressure dominates otherwise."
    """
    cpu_pressure  = min(current_cpu / MAX_CPU, 1.0)
    pred_pressure = min(corrected_pred / MAX_CPU, 1.0)
    spike_burden  = min(spike_count / 10.0, 1.0)

    # FIX 11 — adaptive weights
    if spike_count > 5:
        w_cpu, w_pred, w_spike, w_prob = 0.25, 0.20, 0.40, 0.15
    else:
        w_cpu, w_pred, w_spike, w_prob = 0.35, 0.25, 0.25, 0.15

    trend_mult = {
        "rapidly_rising": 1.3, "rising": 1.15, "elevated": 1.1,
        "volatile": 1.2, "stable": 1.0, "falling": 0.85, "rapidly_falling": 0.7,
    }.get(trend, 1.0)

    raw = (
        w_cpu   * cpu_pressure  +
        w_pred  * pred_pressure +
        w_spike * spike_burden  +
        w_prob  * spike_prob
    ) * trend_mult

    return round(min(raw, 1.0), 4)


# ─────────────────────────────────────────────
# DECISION ENGINE
# ─────────────────────────────────────────────

class DecisionEngine:
    """
    Production-grade autonomous decision engine.

    Pipeline:
      raw predictions → correction layer → spike scoring
      → trend classification → risk scoring → decision
      → explanation engine → output record
    """

    def __init__(self):
        self._spike_tracker    = SpikeTracker()
        self._cooldown_tracker = CooldownTracker()
        self._lock             = threading.Lock()
        self._last_decisions: dict[str, dict] = {}
        # FIX 12 — decision memory for hysteresis (prevents flapping)
        self._last_decision_str: dict[str, str] = {}

    def evaluate(self, predictions: dict[str, dict]) -> dict[str, dict]:
        results = {}
        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

        for sid, pred in predictions.items():
            if pred.get("model_used") == "warming_up":
                results[sid] = self._warming_up_record(pred, ts)
                continue
            results[sid] = self._decide(sid, pred, ts)

        with self._lock:
            self._last_decisions.update(results)

        return results

    def get_last_decisions(self) -> dict[str, dict]:
        with self._lock:
            return dict(self._last_decisions)

    def _decide(self, server_id: str, pred: dict, ts: str) -> dict:
        # ── Extract raw inputs ─────────────────────────────────
        current_cpu   = float(pred.get("current_cpu",      0.0))
        raw_pred      = float(pred.get("predicted_cpu",    current_cpu))
        confidence    = float(pred.get("confidence",       0.5))
        features      = pred.get("features",               {})
        server_name   = pred.get("server_name",            server_id)
        model_used    = pred.get("model_used",             "unknown")

        rolling_mean  = float(features.get("rolling_mean_10", current_cpu))
        rolling_std   = float(features.get("rolling_std_10",  0.0))
        delta_1       = float(features.get("delta_1",         0.0))

        # ── Track spikes ───────────────────────────────────────
        if current_cpu > SPIKE_CPU_THRESHOLD:
            self._spike_tracker.record(server_id)

        spike_count = self._spike_tracker.count(server_id)
        spike_rate  = self._spike_tracker.spike_rate(server_id)

        # ── MODULE 1: Correct prediction ───────────────────────
        corrected_pred = correct_prediction(
            current_cpu, raw_pred, rolling_mean, rolling_std,
            spike_count, confidence,
        )

        # ── FIX 9: Prediction disagreement + model reliability ─
        prediction_disagreement = round(abs(raw_pred - corrected_pred), 2)
        model_reliability       = round(1.0 - (prediction_disagreement / MAX_CPU), 4)

        # ── FIX 13: Adjusted confidence — penalised by disagreement
        adjusted_conf = round(
            confidence * (1.0 - prediction_disagreement / 100.0), 4
        )

        # ── MODULE 2: Spike probability ────────────────────────
        spike_prob = compute_spike_probability(
            current_cpu, corrected_pred, rolling_mean, rolling_std,
            spike_count, adjusted_conf,
        )

        # ── MODULE 3: Trend ────────────────────────────────────
        trend = classify_trend(current_cpu, rolling_mean, rolling_std, delta_1)

        # ── MODULE 5: Risk score (adaptive weights) ────────────
        risk = compute_risk_score(
            current_cpu, corrected_pred, rolling_mean, rolling_std,
            spike_count, spike_prob, adjusted_conf, trend,
        )

        # ── FIX 4: Load state for honest explanations ──────────
        # Prevents "CPU 81% within normal range" embarrassment
        if current_cpu >= 80:
            load_state = "critical load"
        elif current_cpu >= 65:
            load_state = "elevated load"
        else:
            load_state = "normal load"

        # ── FIX 12: Hysteresis — read last decision ────────────
        last_decision = self._last_decision_str.get(server_id, "STABLE")

        # ── FIX 1: Hard reality override ───────────────────────
        # Cannot ignore actual load regardless of model prediction.
        # High CPU ALWAYS forces action — model uncertainty is irrelevant.
        if current_cpu >= 85:
            decision = "SCALE"
        elif current_cpu >= 75 and trend in ("rising", "rapidly_rising"):
            decision = "SCALE"
        else:
            decision = None  # will be set below

        if decision is None:

        if decision is None:
            # ── FIX 10: Decision logic — pattern-aware + hysteresis ─
            escalate_pattern = (
                spike_count >= SPIKE_ESCALATE_COUNT and
                spike_rate > 0.3 and
                current_cpu > 75 and
                trend in ("rapidly_rising", "rising", "elevated")
            )

            if escalate_pattern:
                decision = "ESCALATE"

            elif spike_count >= SPIKE_RESTART_COUNT:
                decision = "RESTART"

            elif (corrected_pred > SCALE_CPU_THRESHOLD and adjusted_conf > SCALE_CONF_THRESHOLD) \
                 or (risk > 0.82 and trend in ("rapidly_rising", "rising") and current_cpu > 70):
                decision = "SCALE"

            elif rolling_std > RESTART_STD_THRESHOLD and spike_count >= 1:
                decision = "RESTART"

            elif corrected_pred > MONITOR_CPU_THRESHOLD or risk > 0.55:
                decision = "MONITOR"

            else:
                decision = "STABLE"

        # FIX 12 — Hysteresis: prevent flapping
        if last_decision == "SCALE" and decision == "MONITOR" and risk > 0.6:
            decision = "SCALE"
        if last_decision == "ESCALATE" and decision in ("RESTART", "MONITOR") and spike_count >= SPIKE_RESTART_COUNT:
            decision = "ESCALATE"

        # Store decision for next tick
        self._last_decision_str[server_id] = decision

        # ── MODULE 4: Explanation ──────────────────────────────
        reason, action = build_explanation(
            decision, current_cpu, corrected_pred, rolling_mean,
            rolling_std, spike_count, spike_prob, confidence,
            trend, spike_rate, load_state,
        )

        severity = DECISIONS[decision]["severity"]
        color    = DECISIONS[decision]["color"]

        # ── Alert cooldown ─────────────────────────────────────
        alert_ready = False
        if severity in ("CRITICAL", "HIGH"):
            alert_ready = self._cooldown_tracker.can_alert(server_id)
            if alert_ready:
                self._cooldown_tracker.record(server_id)

        return {
            "server_id":                server_id,
            "server_name":              server_name,
            "current_cpu":              round(current_cpu,             2),
            "predicted_cpu":            corrected_pred,
            "raw_predicted_cpu":        round(raw_pred,                2),
            "confidence":               round(confidence,              4),
            "adjusted_confidence":      adjusted_conf,
            "spike_probability":        spike_prob,
            "prediction_disagreement":  prediction_disagreement,
            "model_reliability":        model_reliability,
            "decision":                 decision,
            "severity":                 severity,
            "color":                    color,
            "reason":                   reason,
            "action":                   action,
            "spike_count":              spike_count,
            "spike_rate":               round(spike_rate,              2),
            "risk_score":               risk,
            "trend":                    trend,
            "rolling_std":              round(rolling_std,             2),
            "rolling_mean":             round(rolling_mean,            2),
            "alert_ready":              alert_ready,
            "cooldown_remaining":       self._cooldown_tracker.seconds_remaining(server_id),
            "model_used":               model_used,
            "timestamp":                ts,
        }

    def _warming_up_record(self, pred: dict, ts: str) -> dict:
        curr = float(pred.get("current_cpu", 0.0))
        return {
            "server_id":               pred.get("server_id",   ""),
            "server_name":             pred.get("server_name", ""),
            "current_cpu":             curr,
            "predicted_cpu":           curr,
            "raw_predicted_cpu":       curr,
            "confidence":              0.0,
            "adjusted_confidence":     0.0,
            "spike_probability":       0.0,
            "prediction_disagreement": 0.0,
            "model_reliability":       1.0,
            "decision":                "STABLE",
            "severity":                "OK",
            "color":                   "#00C851",
            "reason":                  "Collecting baseline metrics — insufficient history for anomaly detection",
            "action":                  "Passive monitoring active — awaiting 30-reading baseline",
            "spike_count":             0,
            "spike_rate":              0.0,
            "risk_score":              0.0,
            "trend":                   "stable",
            "rolling_std":             0.0,
            "rolling_mean":            curr,
            "alert_ready":             False,
            "cooldown_remaining":      0,
            "model_used":              "warming_up",
            "timestamp":               ts,
        }


# ─────────────────────────────────────────────
# STANDALONE TEST — python decision_engine.py
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import server_simulator as sim
    from pipeline import PredictionPipeline

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print("CrashGuard AI — Decision Engine v4 Test")
    print("─" * 75)

    sim.start()
    pipeline = PredictionPipeline()
    pipeline.start()
    engine   = DecisionEngine()

    print("Warming up (90s)...\n")
    time.sleep(90)

    ICON = {"CRITICAL":"🔴","HIGH":"🟠","MEDIUM":"🟡","OK":"🟢"}

    for cycle in range(10):
        time.sleep(4)
        predictions = pipeline.get_predictions()
        if not predictions:
            print("  Waiting...")
            continue

        decisions = engine.evaluate(predictions)
        print(f"\n── Cycle {cycle+1} ── {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC")
        print(f"  {'SERVER':<28} {'CPU':>6} {'PRED':>6} {'RELIA':>6} {'RISK':>6} {'DECISION':<10} {'TREND'}")
        print(f"  {'─'*82}")

        for sid, d in decisions.items():
            icon = ICON.get(d["severity"], "⚪")
            print(
                f"  {d['server_name']:<28} "
                f"{d['current_cpu']:>5.1f}% "
                f"{d['predicted_cpu']:>5.1f}% "
                f"{d['model_reliability']:>5.2f}  "
                f"{d['risk_score']:>5.2f}  "
                f"{icon} {d['decision']:<8}  "
                f"{d['trend']}"
            )
            print(f"    → {d['reason'][:92]}...")
            print(f"    ⚙  {d['action']}")

    print("\n✅ Decision engine v4 test complete.")
