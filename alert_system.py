"""
alert_system.py — CrashGuard AI
Sends rich Slack block alerts for HIGH and CRITICAL decisions.

Features:
  - Rich Slack blocks (not plain text)
  - Server name, current CPU, predicted CPU
  - Confidence, decision, root cause, action
  - Dynamic insights: trend direction + pattern type
  - 5 minute cooldown per server independently
  - Only fires for HIGH and CRITICAL severity
  - Retry with exponential backoff (3 attempts)
  - Async send — never blocks the main thread
  - Graceful fallback if Slack not configured

Fixes applied:
  [2] Retry with exponential backoff (3 attempts, 1s/2s/4s)
  [3] Slack send runs in background thread — non-blocking
  [5] Dynamic insights: trend direction + pattern type in message
  [7] Removed unused variables (header, subtext)
"""

import os
import json
import time
import logging
import threading
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("crashguard.alerts")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SLACK_WEBHOOK_URL  = os.getenv("SLACK_WEBHOOK_URL", "")
ALERT_COOLDOWN     = 300   # 5 minutes per server
ALERTABLE_SEVERITY = {"CRITICAL", "HIGH"}
MAX_RETRIES        = 3     # FIX 2 — retry attempts
RETRY_BASE_SECONDS = 1.0   # FIX 2 — exponential backoff base

DECISION_EMOJI = {
    "STABLE":      "✅",
    "MONITOR":     "👀",
    "SCALE_READY": "⏳",
    "SCALE":       "📈",
    "RESTART":     "🔁",
    "ESCALATE":    "🚨",
}

SEVERITY_COLOR = {
    "CRITICAL": "#FF3B3B",
    "HIGH":     "#FF8C00",
    "WARNING":  "#FFD700",
    "INFO":     "#00BFFF",
}

# FIX 5 — pattern labels shown in Slack message
BEHAVIOR_LABEL = {
    "stable":      "Stable baseline",
    "gradual":     "Gradual climb",
    "burst":       "Sudden burst",
    "fluctuating": "Oscillating load",
    "critical":    "Sustained high load",
    "demo_trigger":"Manual trigger",
}


# ─────────────────────────────────────────────
# COOLDOWN TRACKER
# ─────────────────────────────────────────────

class AlertCooldown:
    def __init__(self, cooldown_seconds: int = ALERT_COOLDOWN):
        self._lock     = threading.Lock()
        self._cooldown = cooldown_seconds
        self._last: dict[str, float] = {}

    def can_send(self, server_id: str) -> bool:
        with self._lock:
            return (time.time() - self._last.get(server_id, 0.0)) >= self._cooldown

    def mark_sent(self, server_id: str):
        with self._lock:
            self._last[server_id] = time.time()

    def seconds_remaining(self, server_id: str) -> int:
        with self._lock:
            elapsed = time.time() - self._last.get(server_id, 0.0)
            return max(0, int(self._cooldown - elapsed))


# ─────────────────────────────────────────────
# FIX 5 — DYNAMIC INSIGHT BUILDER
# ─────────────────────────────────────────────

def _trend_direction(delta_1: float, delta_5: float) -> str:
    """Derive trend label from delta features."""
    if delta_1 > 3.0 and delta_5 > 5.0:
        return "📈 Rapidly rising"
    elif delta_1 > 1.0:
        return "↗ Rising"
    elif delta_1 < -3.0 and delta_5 < -5.0:
        return "📉 Rapidly falling"
    elif delta_1 < -1.0:
        return "↘ Falling"
    else:
        return "→ Stable"


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.85:
        return "Very high — model is certain"
    elif confidence >= 0.70:
        return "High — reliable prediction"
    elif confidence >= 0.50:
        return "Moderate — monitor closely"
    else:
        return "Low — treat as indicative"


def build_dynamic_insights(decision: dict) -> str:
    """
    FIX 5 — Build dynamic insight string for Slack message.
    Shows trend direction, pattern type, confidence interpretation.
    """
    features = decision.get("features", {})
    delta_1  = features.get("delta_1", 0.0)
    delta_5  = features.get("delta_5", 0.0)
    behavior = decision.get("behavior", "")

    trend    = _trend_direction(delta_1, delta_5)
    pattern  = BEHAVIOR_LABEL.get(behavior, "Unknown pattern")
    conf_lbl = _confidence_label(decision.get("confidence", 0.0))

    return (
        f"*Trend:* {trend}\n"
        f"*Pattern:* {pattern}\n"
        f"*Confidence:* {conf_lbl}"
    )


# ─────────────────────────────────────────────
# SLACK MESSAGE BUILDER
# ─────────────────────────────────────────────

def build_slack_blocks(decision: dict) -> dict:
    server      = decision["server_name"]
    curr_cpu    = decision["current_cpu"]
    pred_cpu    = decision["predicted_cpu"]
    confidence  = decision["confidence"]
    dec         = decision["decision"]
    severity    = decision["severity"]
    reason      = decision["reason"]
    action      = decision["action"]
    spike_count = decision["spike_count"]
    spike_prob  = decision.get("spike_probability", 0.0)
    model_used  = decision.get("model_used", "ensemble")
    ts          = decision["timestamp"]
    emoji       = DECISION_EMOJI.get(dec, "⚠️")
    color       = SEVERITY_COLOR.get(severity, "#888888")

    # FIX 5 — dynamic insights block
    insights = build_dynamic_insights(decision)

    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"CrashGuard AI — {dec} Alert",
                "emoji": True,
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"{emoji} *{server}*\nSeverity: *{severity}*"
            }
        },
        {"type": "divider"},
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Current CPU*\n`{curr_cpu:.1f}%`"},
                {"type": "mrkdwn", "text": f"*Predicted CPU*\n`{pred_cpu:.1f}%`"},
                {"type": "mrkdwn", "text": f"*Confidence*\n`{confidence:.0%}`"},
                {"type": "mrkdwn", "text": f"*Spike Probability*\n`{spike_prob:.0%}`"},
                {"type": "mrkdwn", "text": f"*Spikes (10 min)*\n`{spike_count}`"},
                {"type": "mrkdwn", "text": f"*Model*\n`{model_used}`"},
            ]
        },
        {"type": "divider"},
        # FIX 5 — dynamic insights section
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Dynamic Insights*\n{insights}"
            }
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Root Cause*\n{reason}"
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Autonomous Action*\n⚙ {action}"
            }
        },
        {"type": "divider"},
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        f"🤖 CrashGuard AI  |  {ts}  |  "
                        "Powered by LSTM + XGBoost Ensemble"
                    )
                }
            ]
        }
    ]

    return {
        "attachments": [
            {
                "color":  color,
                "blocks": blocks,
            }
        ]
    }


# ─────────────────────────────────────────────
# ALERT SENDER
# ─────────────────────────────────────────────

class AlertSystem:

    def __init__(self, webhook_url: str = SLACK_WEBHOOK_URL):
        self._webhook    = webhook_url
        self._cooldown   = AlertCooldown()
        self._enabled    = bool(webhook_url)
        self._lock       = threading.Lock()
        self._sent_count = 0
        self._fail_count = 0

        if self._enabled:
            logger.info("Alert system ready — Slack webhook configured.")
        else:
            logger.warning(
                "Alert system in DRY-RUN mode — "
                "set SLACK_WEBHOOK_URL to enable real alerts."
            )

    def process_decisions(self, decisions: dict[str, dict]) -> list[dict]:
        sent = []
        for sid, decision in decisions.items():
            result = self._maybe_send(sid, decision)
            if result:
                sent.append(result)
        return sent

    def _maybe_send(self, server_id: str, decision: dict) -> Optional[dict]:
        severity = decision.get("severity", "OK")
        if severity not in ALERTABLE_SEVERITY:
            return None
        if not self._cooldown.can_send(server_id):
            remaining = self._cooldown.seconds_remaining(server_id)
            logger.debug(f"Alert suppressed for {server_id} — cooldown {remaining}s")
            return None

        payload = build_slack_blocks(decision)
        self._cooldown.mark_sent(server_id)

        if self._enabled:
            # FIX 3 — send in background thread, never blocks main thread
            t = threading.Thread(
                target=self._send_with_retry,
                args=(payload,),
                daemon=True,
            )
            t.start()
        else:
            self._dry_run_log(decision)

        with self._lock:
            self._sent_count += 1

        return {
            "server_id": server_id,
            "decision":  decision["decision"],
            "severity":  severity,
            "sent":      True,
            "dry_run":   not self._enabled,
            "timestamp": decision["timestamp"],
        }

    def _send_with_retry(self, payload: dict):
        """
        FIX 2 — Retry with exponential backoff.
        Attempts: 1s → 2s → 4s between retries.
        """
        for attempt in range(1, MAX_RETRIES + 1):
            success = self._send_to_slack(payload)
            if success:
                return
            if attempt < MAX_RETRIES:
                wait = RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    f"Slack send failed (attempt {attempt}/{MAX_RETRIES}) "
                    f"— retrying in {wait:.0f}s"
                )
                time.sleep(wait)
            else:
                logger.error(f"Slack send failed after {MAX_RETRIES} attempts.")
                with self._lock:
                    self._fail_count += 1

    def _send_to_slack(self, payload: dict) -> bool:
        try:
            data = json.dumps(payload).encode("utf-8")
            req  = urllib.request.Request(
                self._webhook,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception as e:
            logger.error(f"Slack request error: {e}")
            return False

    def _dry_run_log(self, decision: dict):
        features = decision.get("features", {})
        delta_1  = features.get("delta_1", 0.0)
        delta_5  = features.get("delta_5", 0.0)
        trend    = _trend_direction(delta_1, delta_5)
        logger.info(
            f"[DRY-RUN] {decision['server_name']} | "
            f"{decision['decision']} | "
            f"CPU={decision['current_cpu']:.1f}% | "
            f"PRED={decision['predicted_cpu']:.1f}% | "
            f"Trend={trend} | "
            f"{decision['reason']}"
        )

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "sent":    self._sent_count,
                "failed":  self._fail_count,
                "enabled": self._enabled,
            }


# ─────────────────────────────────────────────
# STANDALONE TEST — python alert_system.py
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import server_simulator as sim
    from pipeline import PredictionPipeline
    from decision_engine import DecisionEngine

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print("CrashGuard AI — Alert System Test")
    print("─" * 55)

    webhook = os.getenv("SLACK_WEBHOOK_URL", "")
    if webhook:
        print("✅ Slack webhook configured — real alerts will fire")
    else:
        print("⚠️  No SLACK_WEBHOOK_URL — running in dry-run mode")
    print()

    sim.start()
    pipeline = PredictionPipeline()
    pipeline.start()
    engine   = DecisionEngine()
    alerts   = AlertSystem()

    print("Warming up (90s)...\n")
    time.sleep(90)

    for cycle in range(8):
        time.sleep(4)
        predictions = pipeline.get_predictions()
        if not predictions:
            continue

        decisions = engine.evaluate(predictions)
        sent      = alerts.process_decisions(decisions)

        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        print(f"\n── Cycle {cycle+1} ── {ts} UTC")

        for sid, d in decisions.items():
            icon = ("🔴" if d["severity"] == "CRITICAL" else
                    "🟠" if d["severity"] == "HIGH"     else
                    "🟡" if d["severity"] == "MEDIUM"   else "🟢")
            features = d.get("features", {})
            trend    = _trend_direction(
                features.get("delta_1", 0.0),
                features.get("delta_5", 0.0),
            )
            print(f"  {icon} {d['server_name']:<28} "
                  f"CPU={d['current_cpu']:>5.1f}%  "
                  f"{d['decision']:<10} "
                  f"trend={trend}")

        if sent:
            print(f"\n  📨 Alerts this cycle: {len(sent)}")
            for s in sent:
                mode = "DRY-RUN" if s["dry_run"] else "SENT"
                print(f"     [{mode}] {s['server_id']} → {s['decision']}")
        else:
            print(f"\n  📭 No alerts (cooldown or severity)")

    stats = alerts.get_stats()
    print(f"\n── Alert Stats ──")
    print(f"  Total sent:   {stats['sent']}")
    print(f"  Total failed: {stats['failed']}")
    print(f"  Mode:         {'LIVE' if stats['enabled'] else 'DRY-RUN'}")
    print("\n✅ Alert system test complete.")