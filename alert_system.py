"""
alert_system.py — CrashGuard AI (Production Grade)
Unified alert engine: Slack + Email + Cooldown + Layered Fallback.

Alerts are DERIVED from DecisionEngine output — zero independent logic.

Delivery architecture:
  try: send_slack()
  except: send_email()
  else: log_dry_run()

Email triggers ONLY on ESCALATE decisions.
Cooldown: 120s per server per channel.
"""

import os
import json
import time
import uuid
import logging
import threading
import smtplib
import urllib.request
import urllib.error
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("crashguard.alerts")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")

# Email config — Gmail App Password
SMTP_HOST     = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT     = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER     = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")  # Gmail App Password
EMAIL_TO      = os.getenv("ALERT_EMAIL_TO", "")

ALERT_COOLDOWN_SLACK = 120  # seconds per server
ALERT_COOLDOWN_EMAIL = 120  # seconds per server
MAX_RETRIES          = 3
RETRY_BASE_SECONDS   = 1.0

# Severity mapping — SINGLE SOURCE from decision engine
SEVERITY_MAP = {
    "ESCALATE":    "CRITICAL",
    "SCALE":       "HIGH",
    "SCALE_READY": "WARNING",
    "MONITOR":     "WARNING",
    "STABLE":      "INFO",
}

DECISION_EMOJI = {
    "STABLE":      "✅",
    "MONITOR":     "👀",
    "SCALE_READY": "⏳",
    "SCALE":       "📈",
    "ESCALATE":    "🚨",
}

SEVERITY_COLOR = {
    "CRITICAL": "#FF3B3B",
    "HIGH":     "#FF8C00",
    "WARNING":  "#FFD700",
    "INFO":     "#00BFFF",
}

# Only fire Slack/email for these
ALERTABLE_DECISIONS = {"ESCALATE", "SCALE"}
EMAIL_DECISIONS     = {"ESCALATE"}


# ─────────────────────────────────────────────
# COOLDOWN TRACKER (per-channel, per-server)
# ─────────────────────────────────────────────

class ChannelCooldown:
    """Per-server cooldown tracker for a single delivery channel."""

    def __init__(self, cooldown_seconds: int):
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
# STRUCTURED ALERT BUILDER
# ─────────────────────────────────────────────

def build_structured_alert(decision: dict) -> dict:
    """
    Build canonical alert from decision engine output.
    This is the SINGLE alert format used everywhere.
    """
    dec = decision.get("decision", "STABLE")
    return {
        "id":            str(uuid.uuid4()),
        "timestamp":     decision.get("timestamp", datetime.now(timezone.utc).isoformat()),
        "server_id":     decision.get("server_id", ""),
        "server_name":   decision.get("server_name", ""),
        "decision":      dec,
        "severity":      SEVERITY_MAP.get(dec, "INFO"),
        "cpu":           round(float(decision.get("current_cpu", 0)), 2),
        "predicted_cpu": round(float(decision.get("predicted_cpu", 0)), 2),
        "risk":          round(float(decision.get("crash_risk_5min", decision.get("risk_score", 0))), 4),
        "trend":         decision.get("trend", "stable"),
        "reason":        decision.get("reason", ""),
        "action":        decision.get("action", ""),
        "confidence":    round(float(decision.get("confidence", 0)), 4),
        "spike_count":   int(decision.get("spike_count", 0)),
    }


# ─────────────────────────────────────────────
# SLACK MESSAGE BUILDER
# ─────────────────────────────────────────────

def build_slack_blocks(alert: dict) -> dict:
    """Build Slack block kit payload from structured alert."""
    dec      = alert["decision"]
    severity = alert["severity"]
    emoji    = DECISION_EMOJI.get(dec, "⚠️")
    color    = SEVERITY_COLOR.get(severity, "#888888")

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
                "text": f"{emoji} *{alert['server_name']}*\nSeverity: *{severity}*"
            }
        },
        {"type": "divider"},
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Current CPU*\n`{alert['cpu']:.1f}%`"},
                {"type": "mrkdwn", "text": f"*Predicted CPU*\n`{alert['predicted_cpu']:.1f}%`"},
                {"type": "mrkdwn", "text": f"*Risk*\n`{alert['risk']:.0%}`"},
                {"type": "mrkdwn", "text": f"*Trend*\n`{alert['trend']}`"},
                {"type": "mrkdwn", "text": f"*Confidence*\n`{alert['confidence']:.0%}`"},
                {"type": "mrkdwn", "text": f"*Spikes (10 min)*\n`{alert['spike_count']}`"},
            ]
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Root Cause*\n{alert['reason']}"}
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Autonomous Action*\n⚙ {alert['action']}"}
        },
        {"type": "divider"},
        {
            "type": "context",
            "elements": [{
                "type": "mrkdwn",
                "text": f"🤖 CrashGuard AI  |  {alert['timestamp']}  |  LSTM + XGBoost Ensemble"
            }]
        }
    ]

    return {"attachments": [{"color": color, "blocks": blocks}]}


# ─────────────────────────────────────────────
# EMAIL BUILDER
# ─────────────────────────────────────────────

def build_email(alert: dict) -> MIMEMultipart:
    """Build MIME email for ESCALATE alerts."""
    subject = f"[CRITICAL] CrashGuard — {alert['server_name']} Escalation"

    body = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🚨 CRASHGUARD AI — CRITICAL ESCALATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Server:        {alert['server_name']} ({alert['server_id']})
Decision:      {alert['decision']}
Severity:      {alert['severity']}

Current CPU:   {alert['cpu']:.1f}%
Predicted CPU: {alert['predicted_cpu']:.1f}%
Crash Risk:    {alert['risk']:.0%}
Trend:         {alert['trend']}

Root Cause:
{alert['reason']}

Action:
{alert['action']}

Timestamp: {alert['timestamp']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This alert was generated by CrashGuard AI.
Automated SRE — LSTM + XGBoost Ensemble
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"]    = SMTP_USER
    msg["To"]      = EMAIL_TO
    msg.attach(MIMEText(body, "plain"))
    return msg


# ─────────────────────────────────────────────
# ALERT SYSTEM — UNIFIED DELIVERY ENGINE
# ─────────────────────────────────────────────

class AlertSystem:
    """
    Production alert system with layered delivery.
    
    Pipeline:
      DecisionEngine output → build_structured_alert()
      → cooldown check → try Slack → except email → else dry-run log
    
    Email fires ONLY on ESCALATE. Slack fires on ESCALATE + SCALE.
    """

    def __init__(self):
        self._slack_webhook   = SLACK_WEBHOOK_URL
        self._slack_enabled   = bool(SLACK_WEBHOOK_URL)
        self._email_enabled   = bool(SMTP_USER and SMTP_PASSWORD and EMAIL_TO)
        self._slack_cooldown  = ChannelCooldown(ALERT_COOLDOWN_SLACK)
        self._email_cooldown  = ChannelCooldown(ALERT_COOLDOWN_EMAIL)
        self._lock            = threading.Lock()
        self._sent_count      = 0
        self._fail_count      = 0
        self._email_sent      = 0
        self._dry_run_count   = 0
        self._alert_log: list[dict] = []

        if self._slack_enabled:
            logger.info("Alert system: Slack webhook configured.")
        else:
            logger.warning("Alert system: No Slack webhook — will attempt email fallback.")

        if self._email_enabled:
            logger.info(f"Alert system: Email configured ({SMTP_USER} → {EMAIL_TO}).")
        else:
            logger.warning("Alert system: No email configured — set SMTP_USER/SMTP_PASSWORD/ALERT_EMAIL_TO.")

    def process_decisions(self, decisions: dict[str, dict]) -> list[dict]:
        """Process all decisions from engine. Returns list of fired alerts."""
        fired = []
        for sid, decision in decisions.items():
            result = self._process_one(sid, decision)
            if result:
                fired.append(result)
        return fired

    def _process_one(self, server_id: str, decision: dict) -> Optional[dict]:
        """Process single decision through alert pipeline."""
        dec = decision.get("decision", "STABLE")

        # Only alert on actionable decisions
        if dec not in ALERTABLE_DECISIONS:
            return None

        alert = build_structured_alert(decision)

        # ── LAYERED DELIVERY ──────────────────────────────
        slack_sent = False
        email_sent = False
        dry_run    = False

        # Layer 1: Try Slack
        if dec in ALERTABLE_DECISIONS and self._slack_cooldown.can_send(server_id):
            if self._slack_enabled:
                try:
                    self._send_slack_async(alert)
                    self._slack_cooldown.mark_sent(server_id)
                    slack_sent = True
                except Exception as e:
                    logger.error(f"Slack send failed for {server_id}: {e}")
                    # Fall through to email
            
            # Layer 2: Email fallback (or primary for ESCALATE)
            if (not slack_sent or dec in EMAIL_DECISIONS) and dec in EMAIL_DECISIONS:
                if self._email_enabled and self._email_cooldown.can_send(server_id):
                    try:
                        self._send_email_async(alert)
                        self._email_cooldown.mark_sent(server_id)
                        email_sent = True
                    except Exception as e:
                        logger.error(f"Email send failed for {server_id}: {e}")

            # Layer 3: Dry-run log if nothing sent
            if not slack_sent and not email_sent:
                self._dry_run_log(alert)
                dry_run = True
        else:
            # Cooldown active
            remaining = self._slack_cooldown.seconds_remaining(server_id)
            logger.debug(f"Alert suppressed for {server_id} — cooldown {remaining}s")
            return None

        with self._lock:
            self._sent_count += 1
            if dry_run:
                self._dry_run_count += 1
            self._alert_log.append(alert)
            # Cap log at 500 entries
            if len(self._alert_log) > 500:
                self._alert_log = self._alert_log[-500:]

        return {
            "alert_id":   alert["id"],
            "server_id":  server_id,
            "decision":   dec,
            "severity":   alert["severity"],
            "slack_sent":  slack_sent,
            "email_sent":  email_sent,
            "dry_run":     dry_run,
            "timestamp":   alert["timestamp"],
        }

    # ── SLACK DELIVERY ──────────────────────────────────

    def _send_slack_async(self, alert: dict):
        payload = build_slack_blocks(alert)
        t = threading.Thread(
            target=self._send_slack_with_retry,
            args=(payload, alert["server_id"]),
            daemon=True,
        )
        t.start()

    def _send_slack_with_retry(self, payload: dict, server_id: str):
        for attempt in range(1, MAX_RETRIES + 1):
            success = self._send_to_slack(payload)
            if success:
                with self._lock:
                    self._sent_count += 1
                logger.info(f"Slack alert sent for {server_id}")
                return
            if attempt < MAX_RETRIES:
                wait = RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                logger.warning(f"Slack failed (attempt {attempt}/{MAX_RETRIES}) — retry in {wait:.0f}s")
                time.sleep(wait)
            else:
                logger.error(f"Slack failed after {MAX_RETRIES} attempts for {server_id}")
                with self._lock:
                    self._fail_count += 1

    def _send_to_slack(self, payload: dict) -> bool:
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self._slack_webhook,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception as e:
            logger.error(f"Slack request error: {e}")
            return False

    # ── EMAIL DELIVERY ──────────────────────────────────

    def _send_email_async(self, alert: dict):
        t = threading.Thread(
            target=self._send_email_with_retry,
            args=(alert,),
            daemon=True,
        )
        t.start()

    def _send_email_with_retry(self, alert: dict):
        for attempt in range(1, MAX_RETRIES + 1):
            success = self._send_email(alert)
            if success:
                with self._lock:
                    self._email_sent += 1
                logger.info(f"Email alert sent for {alert['server_id']}")
                return
            if attempt < MAX_RETRIES:
                wait = RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                logger.warning(f"Email failed (attempt {attempt}/{MAX_RETRIES}) — retry in {wait:.0f}s")
                time.sleep(wait)
            else:
                logger.error(f"Email failed after {MAX_RETRIES} attempts for {alert['server_id']}")
                with self._lock:
                    self._fail_count += 1

    def _send_email(self, alert: dict) -> bool:
        try:
            msg = build_email(alert)
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(msg)
            return True
        except Exception as e:
            logger.error(f"SMTP error: {e}")
            return False

    # ── DRY-RUN FALLBACK ────────────────────────────────

    def _dry_run_log(self, alert: dict):
        logger.info(
            f"[DRY-RUN] {alert['server_name']} | "
            f"{alert['decision']} ({alert['severity']}) | "
            f"CPU={alert['cpu']:.1f}% | "
            f"PRED={alert['predicted_cpu']:.1f}% | "
            f"Risk={alert['risk']:.0%} | "
            f"Trend={alert['trend']} | "
            f"{alert['reason'][:120]}"
        )

    # ── STATS ───────────────────────────────────────────

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "slack_sent":    self._sent_count,
                "email_sent":    self._email_sent,
                "failed":        self._fail_count,
                "dry_runs":      self._dry_run_count,
                "slack_enabled": self._slack_enabled,
                "email_enabled": self._email_enabled,
                "total_alerts":  len(self._alert_log),
            }

    def get_alert_log(self) -> list[dict]:
        with self._lock:
            return list(self._alert_log)


# ─────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print("CrashGuard AI — Alert System Test")
    print("─" * 55)

    alerts = AlertSystem()
    stats = alerts.get_stats()
    print(f"  Slack: {'LIVE' if stats['slack_enabled'] else 'DISABLED'}")
    print(f"  Email: {'LIVE' if stats['email_enabled'] else 'DISABLED'}")

    # Simulate an ESCALATE decision
    mock_decision = {
        "server_id": "server_e",
        "server_name": "Server E — Critical",
        "current_cpu": 91.5,
        "predicted_cpu": 93.2,
        "confidence": 0.85,
        "crash_risk_5min": 0.78,
        "risk_score": 0.82,
        "decision": "ESCALATE",
        "severity": "CRITICAL",
        "trend": "rapidly_rising",
        "spike_count": 12,
        "reason": "Sustained CPU >90% with rising trend — system crash imminent",
        "action": "Paging on-call engineer — SLA breach risk",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
    }

    result = alerts._process_one("server_e", mock_decision)
    if result:
        print(f"\n  Alert fired: {result}")
    else:
        print("\n  Alert suppressed (cooldown)")

    print(f"\n  Stats: {alerts.get_stats()}")
    print("\n✅ Alert system test complete.")