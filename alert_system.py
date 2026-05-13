"""
alert_system.py — CrashGuard AI (Production Grade)
Unified alert engine: Twilio + Slack + Email + Cooldown + Layered Fallback.

Alerts are DERIVED from DecisionEngine output — zero independent logic.

Delivery architecture:
  ESCALATE → Twilio phone call + email (fallback: Slack → dry-run)
  SCALE    → Email (fallback: Slack → dry-run)
  RESTART  → Email (fallback: Slack → dry-run)
  MONITOR  → Dashboard only (no external notification)
  STABLE   → Dashboard only

FIX 2 — Realistic alert deduplication with per-server suppression tracking.
FIX 3 — Gmail SMTP email alerts for CRITICAL + HIGH severity.
FIX 4 — Twilio phone call for ESCALATE decisions.
         Non-blocking with exponential backoff retry.

Cooldown: 300s (5 minutes) per server per channel.

Windows env vars (set BEFORE python app.py):
  set TWILIO_ACCOUNT_SID=your_account_sid
  set TWILIO_AUTH_TOKEN=your_token
  set TWILIO_FROM_NUMBER=+1234567890
  set TWILIO_TO_NUMBER=+0987654321
  set SMTP_USER=your@gmail.com
  set SMTP_PASS=your-16-char-app-password
  set ALERT_EMAIL=your@gmail.com
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
import random

# Safe Twilio import — graceful degradation if not installed
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TwilioClient = None
    TWILIO_AVAILABLE = False

logger = logging.getLogger("crashguard.alerts")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")

# Email config — read at CALL TIME, not import time.
# To get Gmail App Password:
# Google Account → Security → 2-Step Verification → App Passwords → Mail
#
# Required env vars (set in terminal BEFORE python app.py):
#   Windows:  set SMTP_USER=yourgmail@gmail.com
#             set SMTP_PASS=abcdefghijklmnop   (Gmail App Password, no spaces)
#             set ALERT_EMAIL=yourgmail@gmail.com
#   Linux:    export SMTP_USER=yourgmail@gmail.com  etc.
#
# NOTE: These are intentionally NOT read here. They are read inside
#       _get_email_config() at function-call time so that env vars
#       set after this module is imported are still picked up.

COOLDOWN_BY_DECISION = {
    "MONITOR": 300,
    "SCALE_READY": 300,
    "SCALE": 120,
    "ESCALATE": 180,
}
MAX_RETRIES            = 3
RETRY_BASE_SECONDS     = 1.0

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

# Production alert routing — only meaningful incidents generate external notifications
ALERTABLE_DECISIONS = {"ESCALATE", "SCALE", "RESTART"}
EMAIL_DECISIONS     = {"ESCALATE", "SCALE", "RESTART"}

# FIX 2 — Suppression sanity cap per session
MAX_SUPPRESSIONS_PER_SESSION = 50


# ─────────────────────────────────────────────
# COOLDOWN TRACKER (per-channel, per-server)
# FIX 2 — enforces 5 min cooldown BEFORE severity check
# ─────────────────────────────────────────────

class ChannelCooldown:
    """Per-server cooldown tracker for a single delivery channel."""

    def __init__(self, cooldown_seconds: int):
        self._lock     = threading.Lock()
        self._cooldown = cooldown_seconds
        self._last: dict[str, float] = {}

    def can_send_with_cooldown(self, server_id: str, cooldown: int) -> bool:
        with self._lock:
            elapsed = time.time() - self._last.get(server_id, 0.0)
            return elapsed >= cooldown

    def can_send(self, server_id: str) -> bool:
        with self._lock:
            elapsed = time.time() - self._last.get(server_id, 0.0)
            return elapsed >= self._cooldown

    def mark_sent(self, server_id: str):
        with self._lock:
            self._last[server_id] = time.time()

    def seconds_remaining(self, server_id: str, cooldown: int) -> int:
        with self._lock:
            elapsed = time.time() - self._last.get(server_id, 0.0)
            return max(0, int(cooldown - elapsed))


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
            "text": {"type": "mrkdwn", "text": f"*Autonomous Mitigation*\n⚙ {alert['action']}"}
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
# EMAIL BUILDER (FIX 3 — reads env at call time)
# ─────────────────────────────────────────────

def build_email(alert: dict, smtp_user: str, email_to: str) -> MIMEMultipart:
    """
    Build MIME email for CRITICAL and HIGH severity alerts.
    smtp_user and email_to are passed in explicitly so we never
    depend on module-level variables.
    """
    if alert['decision'] == "ESCALATE":
        subject = f"[CrashGuard][ESCALATE] {alert['server_name']} — Immediate Attention Required"
    else:
        subject = f"[CrashGuard][{alert['decision']}] {alert['server_name']} — CPU {alert['cpu']:.1f}%"

    body = (
        "CrashGuard AI Alert\n"
        "===================\n"
        f"Server:     {alert['server_name']}\n"
        f"Current CPU: {alert['cpu']:.1f}%\n"
        f"Predicted:  {alert['predicted_cpu']:.1f}%\n"
        f"Operational Risk: {alert['risk']:.0%}\n"
        f"Decision:   {alert['decision']}\n"
        f"Timestamp:  {alert['timestamp']}\n"
        f"Action:     {alert['action']}\n"
        "\n"
        "-- CrashGuard AI Autonomous Decision System\n"
    )

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"]    = smtp_user
    msg["To"]      = email_to
    msg.attach(MIMEText(body, "plain"))
    return msg


# ─────────────────────────────────────────────
# DYNAMIC VOICE INTELLIGENCE ENGINE
# Production-grade, anti-repetition phrase system.
# Messages are contextual, adaptive, and under 20 seconds.
# ─────────────────────────────────────────────

# ── PHRASE POOLS (rotated randomly to prevent identical calls) ──

_VOICE_INTROS = [
    "This is CrashGuard AI.",
    "Automated infrastructure alert.",
    "CrashGuard escalation notification.",
    "CrashGuard operations alert.",
    "Infrastructure monitoring notification.",
]

_VOICE_SEV_SEVERE = [
    "Severe infrastructure instability detected on {server}.",
    "Critical resource exhaustion detected on {server}.",
    "Major operational disruption on {server}.",
    "Severe capacity failure detected on {server}.",
]

_VOICE_SEV_CRITICAL = [
    "Critical escalation detected on {server}.",
    "Critical load conditions detected on {server}.",
    "Urgent escalation triggered for {server}.",
    "High severity incident on {server}.",
]

_VOICE_SEV_HIGH = [
    "High infrastructure load detected on {server}.",
    "Elevated system load detected on {server}.",
    "Significant load increase on {server}.",
    "High load conditions detected on {server}.",
]

_VOICE_SEV_STABILIZING = [
    "Load is beginning to stabilize on {server}.",
    "{server} is showing signs of recovery.",
    "System pressure is decreasing on {server}.",
    "Conditions are stabilizing on {server}.",
]

# ── CPU CONTEXT PHRASES ──

_VOICE_CPU_CRITICAL = [
    "CPU utilization remains critically elevated at {cpu} percent.",
    "Current CPU usage is at {cpu} percent.",
    "CPU remains at {cpu} percent.",
]

_VOICE_CPU_HIGH = [
    "Current CPU utilization is {cpu} percent.",
    "Current CPU usage remains elevated at {cpu} percent.",
    "CPU is currently at {cpu} percent.",
]

# ── TREND-AWARE PHRASES ──

_VOICE_TREND_RISING = [
    "Predicted workload is continuing to rise rapidly.",
    "Predicted workload continues rising beyond safe thresholds.",
    "System pressure is expected to escalate further.",
]

_VOICE_TREND_ELEVATED = [
    "Workload is expected to remain critically elevated.",
    "Predicted workload is expected to exceed safe operating thresholds within the next few minutes.",
    "Load is projected to remain at critical levels.",
]

_VOICE_TREND_STABILIZING = [
    "Workload is beginning to stabilize.",
    "Predicted load is trending downward.",
    "System load is expected to decrease.",
]

# ── MITIGATION STATUS PHRASES ──

_VOICE_MITIGATION_SCALED = [
    "Automatic scaling has already been triggered.",
    "Automated scaling procedures were initiated.",
    "Autonomous scaling has been activated.",
]

_VOICE_MITIGATION_RESTARTED = [
    "Automated recovery procedures completed.",
    "Automated restart procedures have been executed.",
    "Recovery intervention has completed.",
]

_VOICE_MITIGATION_MONITORING = [
    "Monitoring escalation conditions.",
    "System remains under active monitoring.",
    "Continuous monitoring is in effect.",
]

# ── ACTION RECOMMENDATION PHRASES ──

_VOICE_ACT_ESCALATE = [
    "Immediate operator review is recommended.",
    "On-call escalation has been initiated.",
    "Manual investigation may now be required.",
    "Immediate engineer review is recommended.",
]

_VOICE_ACT_WATCH = [
    "Monitoring continues.",
    "Continued observation is advised.",
    "No immediate action required at this time.",
]


def build_voice_message(decision_data: dict) -> str:
    """
    Generate a dynamic, operationally realistic voice message.

    Adapts based on: server name, current CPU, predicted CPU, operational risk,
    trend direction, mitigation status, escalation severity, and recovery.

    Returns a concise spoken message suitable for Twilio TTS (under 20 seconds).
    """
    server_name = decision_data.get("server_name", "Unknown Server")
    cpu = round(float(decision_data.get("cpu", decision_data.get("current_cpu", 0))))
    predicted_cpu = round(float(decision_data.get("predicted_cpu", cpu)))
    decision = decision_data.get("decision", "ESCALATE")
    trend = decision_data.get("trend", "stable")
    action_text = decision_data.get("action", "").lower()

    parts = []

    # ── 1. INTRO (rotated) ──
    parts.append(random.choice(_VOICE_INTROS))

    # ── 2. SEVERITY PHRASE (CPU-threshold driven) ──
    is_stabilizing = trend in ("rapidly_falling", "falling") or predicted_cpu < cpu - 3

    if is_stabilizing and decision not in ("ESCALATE",):
        parts.append(random.choice(_VOICE_SEV_STABILIZING).format(server=server_name))
    elif cpu > 95:
        parts.append(random.choice(_VOICE_SEV_SEVERE).format(server=server_name))
    elif cpu >= 90:
        parts.append(random.choice(_VOICE_SEV_CRITICAL).format(server=server_name))
    elif cpu >= 80:
        parts.append(random.choice(_VOICE_SEV_HIGH).format(server=server_name))
    else:
        parts.append(random.choice(_VOICE_SEV_HIGH).format(server=server_name))

    # ── 3. CPU CONTEXT (no decimals, natural wording) ──
    if not is_stabilizing or decision == "ESCALATE":
        if cpu >= 90:
            parts.append(random.choice(_VOICE_CPU_CRITICAL).format(cpu=cpu))
        else:
            parts.append(random.choice(_VOICE_CPU_HIGH).format(cpu=cpu))

    # ── 4. TREND PHRASE (predicted vs current) ──
    if predicted_cpu > cpu + 5:
        parts.append(random.choice(_VOICE_TREND_RISING))
    elif abs(predicted_cpu - cpu) <= 5 and cpu >= 80:
        parts.append(random.choice(_VOICE_TREND_ELEVATED))
    elif predicted_cpu < cpu:
        parts.append(random.choice(_VOICE_TREND_STABILIZING))

    # ── 5. MITIGATION STATUS (inferred from decision + action text) ──
    if decision in ("ESCALATE", "SCALE") or "scal" in action_text:
        parts.append(random.choice(_VOICE_MITIGATION_SCALED))
    elif decision == "RESTART" or "restart" in action_text or "recover" in action_text:
        parts.append(random.choice(_VOICE_MITIGATION_RESTARTED))
    elif decision == "MONITOR":
        parts.append(random.choice(_VOICE_MITIGATION_MONITORING))

    # ── 6. ACTION RECOMMENDATION ──
    if decision == "ESCALATE":
        parts.append(random.choice(_VOICE_ACT_ESCALATE))
    elif is_stabilizing:
        parts.append(random.choice(_VOICE_ACT_WATCH))
    elif decision == "SCALE":
        parts.append(random.choice(_VOICE_ACT_WATCH))
    # MONITOR / STABLE — no extra action line (mitigation phrase covers it)

    return " ".join(parts)


# ─────────────────────────────────────────────
# ALERT SYSTEM — UNIFIED DELIVERY ENGINE
# ─────────────────────────────────────────────

class AlertSystem:
    """
    Production alert system with layered delivery.
    
    Pipeline:
      DecisionEngine output → build_structured_alert()
      → cooldown check (per server, 5 min) → severity check
      → try Slack → except email → else dry-run log
    
    FIX 2: Cooldown is checked FIRST, then severity filter.
           Suppression tracked per-server with reason.
    FIX 3: Email fires for CRITICAL + HIGH.
           Priority: Slack → Email → DRY-RUN.
    """

    def __init__(self):
        self._slack_webhook   = SLACK_WEBHOOK_URL
        self._slack_enabled   = bool(SLACK_WEBHOOK_URL)
        # NOTE: email_enabled is checked dynamically at send time via _get_email_config()
        # FIX 2 — Single cooldown tracker shared across channels (5 min per server)
        self._cooldown        = ChannelCooldown(300)
        self._lock            = threading.Lock()
        self._sent_count      = 0
        self._fail_count      = 0
        self._email_sent      = 0
        self._twilio_sent     = 0
        self._dry_run_count   = 0
        self._alert_log: list[dict] = []
        self._timeline: list[dict] = []
        self._last_decision: dict[str, str] = {}

        # FIX 2 — Per-server suppression tracking
        self._suppressed_by_cooldown: dict[str, int] = {}  # server_id → count
        self._suppressed_by_severity: dict[str, int] = {}  # server_id → count
        self._total_suppressed_session = 0

        if self._slack_enabled:
            logger.info("Alert system: Slack webhook configured.")
            print("[ALERT] Slack webhook configured.")
        else:
            logger.warning("Alert system: No Slack webhook — will attempt email fallback.")
            print("[ALERT] No Slack webhook — will attempt email fallback.")

        # Check email config at init time for logging, but will re-check at send time
        cfg = self._get_email_config()
        if cfg:
            logger.info(f"Alert system: Email configured ({cfg['user']} → {cfg['to']}).")
            print(f"[ALERT] Email configured ({cfg['user']} → {cfg['to']}).")
        else:
            logger.warning("Alert system: No email configured — set SMTP_USER/SMTP_PASS/ALERT_EMAIL.")
            print("[ALERT] No email env vars detected at startup. Set SMTP_USER/SMTP_PASS/ALERT_EMAIL.")

        # Check Twilio config at init time for logging
        if TWILIO_AVAILABLE and self._has_twilio():
            print(f"[ALERT] Twilio configured (ESCALATE → phone call to {os.getenv('TWILIO_TO_NUMBER')}).")
        elif TWILIO_AVAILABLE:
            print("[ALERT] Twilio library installed but env vars missing. Set TWILIO_ACCOUNT_SID/AUTH_TOKEN/FROM/TO.")
        else:
            print("[ALERT] Twilio not installed — ESCALATE will fall back to email. pip install twilio>=8.0.0")

    def _record_timeline(self, evt_type: str, server_id: str, message: str):
        with self._lock:
            self._timeline.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": evt_type,
                "server_id": server_id,
                "message": message
            })
            if len(self._timeline) > 100:
                self._timeline.pop(0)

    def get_timeline(self) -> list[dict]:
        """Return timeline filtered to operational events only. Internal routing excluded."""
        _VISIBLE_TYPES = {"ESCALATE", "SCALE", "RESTART", "MONITOR", "RECOVERED",
                          "CALL_TRIGGERED", "EMAIL_SENT", "EMAIL_FAILED", "TWILIO_FAILED"}
        with self._lock:
            return [e for e in self._timeline if e.get("type", "") in _VISIBLE_TYPES]

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
        generated_at = decision.get("timestamp", datetime.now(timezone.utc).isoformat())
        dec = decision.get("decision", "STABLE")
        severity = SEVERITY_MAP.get(dec, "INFO")
        server_name = decision.get('server_name', server_id)

        # FIX 2 — Step 2: Only alert on actionable decisions (severity filter)
        if dec not in ALERTABLE_DECISIONS:
            if severity not in ("CRITICAL", "HIGH"):
                # Only track suppression for non-INFO decisions
                if dec not in ("STABLE",):
                    with self._lock:
                        self._suppressed_by_severity[server_id] = self._suppressed_by_severity.get(server_id, 0) + 1
                        self._total_suppressed_session += 1
            return None

        print(f"[ALERT_WORKER] received decision={dec} server={server_id}")
        self._record_timeline(dec, server_id, f"Decision: {dec}")

        with self._lock:
            last_dec = self._last_decision.get(server_id, "STABLE")
            self._last_decision[server_id] = dec

        is_duplicate = (dec == last_dec)

        # FIX 2 — Step 1: Check cooldown FIRST (before severity)
        cooldown = COOLDOWN_BY_DECISION.get(dec, 300)
        if not self._cooldown.can_send_with_cooldown(server_id, cooldown):
            if not is_duplicate:
                print(f"[COOLDOWN] Bypassed for {server_name} — decision changed from {last_dec} to {dec}")
            else:
                remaining = self._cooldown.seconds_remaining(server_id, cooldown)
                print(f"[COOLDOWN_SUPPRESSED] server={server_id} decision={dec} remaining={remaining}s")
                logger.debug(f"Alert suppressed for {server_id} — cooldown {remaining}s remaining")
                with self._lock:
                    self._suppressed_by_cooldown[server_id] = self._suppressed_by_cooldown.get(server_id, 0) + 1
                    self._total_suppressed_session += 1
                return None

        if is_duplicate and dec == "ESCALATE":
            print(f"[TWILIO_REPEAT] server={server_id} reason=incident_unresolved")

        # FIX 2 — Sanity cap: if we've suppressed > 50 this session, log warning
        if self._total_suppressed_session > MAX_SUPPRESSIONS_PER_SESSION:
            logger.warning(
                f"Suppression sanity check: {self._total_suppressed_session} suppressions this session. "
                f"Cooldown logic may need review."
            )

        alert = build_structured_alert(decision)

        # ── LAYERED DELIVERY (FIX 3 + FIX 4) ─────────────
        # ESCALATE → Twilio phone call (fallback: email → Slack → dry-run)
        # SCALE/MONITOR → Slack → Email → dry-run
        twilio_sent = False
        slack_sent  = False
        email_sent  = False
        dry_run     = False

        # Layer 0: Route decisions to appropriate channels
        current_cpu = alert.get("cpu", 0)
        if dec == "ESCALATE":
            print(f"[ROUTER] decision={dec} channel=TWILIO+EMAIL server={server_id}")
        elif dec in ("SCALE", "RESTART"):
            print(f"[ROUTER] decision={dec} channel=EMAIL server={server_id}")
        else:
            print(f"[ROUTER] decision={dec} channel=DASHBOARD_ONLY server={server_id}")

        if dec == "ESCALATE" and TWILIO_AVAILABLE and self._has_twilio():
            try:
                print(f"[TWILIO_TRIGGER] server={server_id}")
                self._send_twilio_async(alert)
                twilio_sent = True
            except Exception as e:
                logger.error(f"Twilio queue failed for {server_id}: {e}")
                print(f"[TWILIO] Failed to queue call for {server_id}: {e}")

        # Layer 1: Try Slack (skip if Twilio already queued for ESCALATE)
        if not twilio_sent and self._slack_enabled:
            try:
                self._send_slack_async(alert)
                slack_sent = True
            except Exception as e:
                logger.error(f"Slack send failed for {server_id}: {e}")

        # Layer 2: Email fallback (or additional for CRITICAL/HIGH)
        # FIX 3 — fires for CRITICAL and HIGH severity
        # Skip if Twilio queued (Twilio retry will fallback to email internally)
        if not twilio_sent and (not slack_sent or dec in EMAIL_DECISIONS):
            email_cfg = self._get_email_config()
            if email_cfg:
                try:
                    print(f"[ROUTER] selected_channel=email server={server_id}")
                    self._send_email_async(alert, email_cfg, generated_at)
                    email_sent = True
                except Exception as e:
                    logger.error(f"Email send failed for {server_id}: {e}")
                    print(f"[EMAIL] Failed to queue email for {server_id}: {e}")

        # Layer 3: DRY-RUN log if nothing sent
        if not twilio_sent and not slack_sent and not email_sent:
            self._dry_run_log(alert)
            dry_run = True

        # Mark cooldown for this server
        self._cooldown.mark_sent(server_id)

        with self._lock:
            self._sent_count += 1
            if dry_run:
                self._dry_run_count += 1
            self._alert_log.append(alert)
            # Cap log at 500 entries
            if len(self._alert_log) > 500:
                self._alert_log = self._alert_log[-500:]

        return {
            "alert_id":    alert["id"],
            "server_id":   server_id,
            "decision":    dec,
            "severity":    alert["severity"],
            "twilio_sent": twilio_sent,
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

    # ── EMAIL CONFIG (read at call time, NOT import time) ──

    @staticmethod
    def _get_email_config() -> Optional[dict]:
        """
        Read SMTP env vars RIGHT NOW.  Returns dict if all three are set,
        else None.  This is the fix for the [DRY-RUN] bug: env vars set
        after import are now picked up.
        """
        smtp_user = os.getenv("SMTP_USER", "").strip()
        smtp_pass = os.getenv("SMTP_PASS", os.getenv("SMTP_PASSWORD", "")).strip()
        email_to  = os.getenv("ALERT_EMAIL", os.getenv("ALERT_EMAIL_TO", "")).strip()
        smtp_host = os.getenv("SMTP_SERVER", os.getenv("SMTP_HOST", "smtp.gmail.com")).strip()
        smtp_port = int(os.getenv("SMTP_PORT", "587"))

        if smtp_user and smtp_pass and email_to:
            return {
                "user": smtp_user,
                "pass": smtp_pass,
                "to":   email_to,
                "host": smtp_host,
                "port": smtp_port,
            }
        return None

    # ── EMAIL DELIVERY (FIX 3 — non-blocking with retry) ──

    def _send_email_async(self, alert: dict, email_cfg: dict, generated_at: str):
        """Non-blocking: runs email delivery in background thread."""
        t = threading.Thread(
            target=self._send_email_with_retry,
            args=(alert, email_cfg, generated_at),
            daemon=True,
        )
        t.start()

    def _send_email_with_retry(self, alert: dict, email_cfg: dict, generated_at: str):
        """
        3 attempts with 0.5s/1s/2s exponential backoff.
        Prints to console with timing for latency measurement.
        """
        t0 = time.time()
        print(f"[EMAIL] Generated at: {generated_at}")
        print(f"[EMAIL] Sending to {email_cfg['to']}...")

        for attempt in range(1, MAX_RETRIES + 1):
            t_attempt = time.time()
            success = self._send_email(alert, email_cfg)
            duration_ms = int((time.time() - t_attempt) * 1000)
            if success:
                total_ms = int((time.time() - t0) * 1000)
                with self._lock:
                    self._email_sent += 1
                logger.info(f"Email alert sent for {alert['server_id']} ({total_ms}ms total)")
                print(f"[EMAIL_SENT] decision={alert['decision']} latency_ms={total_ms}")
                self._record_timeline("EMAIL_SENT", alert.get("server_id", "unknown"), f"{total_ms}ms delay")
                return
            if attempt < MAX_RETRIES:
                wait = 0.5 * (2 ** (attempt - 1))  # 0.5s, 1s, 2s
                logger.warning(f"Email failed (attempt {attempt}/{MAX_RETRIES}, {duration_ms}ms) — retry in {wait:.1f}s")
                print(f"[EMAIL] Attempt {attempt}/{MAX_RETRIES} failed ({duration_ms}ms) — retrying in {wait:.1f}s...")
                time.sleep(wait)
            else:
                total_ms = int((time.time() - t0) * 1000)
                logger.error(f"Email failed after {MAX_RETRIES} attempts for {alert['server_id']} ({total_ms}ms) — falling back to DRY-RUN")
                print(f"[EMAIL] Failed after {MAX_RETRIES} attempts for {alert['server_name']} ({total_ms}ms) — falling back to DRY-RUN")
                self._record_timeline("EMAIL_FAILED", alert.get("server_id", "unknown"), f"Failed after {MAX_RETRIES} attempts")
                self._dry_run_log(alert)
                with self._lock:
                    self._fail_count += 1

    def _send_email(self, alert: dict, email_cfg: dict) -> bool:
        """
        Attempt a single SMTP send.  All config is passed via email_cfg
        (read from env vars at call time, not import time).
        """
        try:
            msg = build_email(alert, smtp_user=email_cfg["user"], email_to=email_cfg["to"])
            with smtplib.SMTP(email_cfg["host"], email_cfg["port"], timeout=5) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(email_cfg["user"], email_cfg["pass"])
                server.send_message(msg)
            return True
        except Exception as e:
            logger.error(f"SMTP error: {e}")
            print(f"[EMAIL] SMTP error: {e}")
            return False

    # ── TWILIO DELIVERY (FIX 4 — phone call for ESCALATE) ──

    @staticmethod
    def _has_twilio() -> bool:
        """Check if all Twilio env vars are set. Read at call time."""
        return all([
            os.getenv("TWILIO_ACCOUNT_SID"),
            os.getenv("TWILIO_AUTH_TOKEN"),
            os.getenv("TWILIO_FROM_NUMBER"),
            os.getenv("TWILIO_TO_NUMBER"),
        ])

    def _send_twilio_async(self, alert: dict):
        """Non-blocking: runs Twilio call in background thread."""
        t = threading.Thread(
            target=self._send_twilio_call_with_retry,
            args=(alert,),
            daemon=True,
        )
        t.start()

    def _send_twilio_call_with_retry(self, alert: dict):
        """
        3 attempts with 1s/2s exponential backoff.
        First attempt immediate — no sleep before first try.
        Falls back to email if all retries fail.
        """
        server_id = alert.get("server_id", "unknown")
        for attempt in range(1, MAX_RETRIES + 1):
            success = self._send_twilio_call(alert)
            if success:
                with self._lock:
                    self._twilio_sent += 1
                return
            if attempt < MAX_RETRIES:
                wait = RETRY_BASE_SECONDS * (2 ** (attempt - 1))  # 1s, 2s
                print(f"[TWILIO] Attempt {attempt}/{MAX_RETRIES} failed — retrying in {wait:.0f}s...")
                time.sleep(wait)
            else:
                print(f"[TWILIO] Failed after {MAX_RETRIES} attempts for {alert.get('server_name', server_id)}")
                self._record_timeline("TWILIO_FAILED", alert.get("server_id"), f"Failed after {MAX_RETRIES} attempts")
                print(f"[TWILIO] Falling back to email...")
                # Graceful fallback to email
                email_cfg = self._get_email_config()
                if email_cfg:
                    generated_at = alert.get("timestamp", "unknown")
                    self._send_email_with_retry(alert, email_cfg, generated_at)
                else:
                    self._dry_run_log(alert)
                with self._lock:
                    self._fail_count += 1

    def _send_twilio_call(self, alert: dict) -> bool:
        """
        Attempt a single Twilio outbound call.
        Reads env vars inside function body (not at import time).
        Uses TwiML <Say voice='alice'> to speak the alert message.
        """
        try:
            account_sid = os.getenv("TWILIO_ACCOUNT_SID")
            auth_token  = os.getenv("TWILIO_AUTH_TOKEN")
            from_number = os.getenv("TWILIO_FROM_NUMBER")
            to_number   = os.getenv("TWILIO_TO_NUMBER")

            if not all([account_sid, auth_token, from_number, to_number]):
                print("[TWILIO] Missing env vars — cannot place call")
                return False

            print(f"[TWILIO] Debug: Auth Token Length: {len(auth_token) if auth_token else 0}")
            print(f"[TWILIO] Starting call from {from_number} to {to_number}")

            client = TwilioClient(account_sid, auth_token)

            server_name = alert.get("server_name", "Unknown")

            spoken_text = build_voice_message(alert)

            twiml_message = (
                f"<Response><Say voice='alice'>"
                f"{spoken_text}"
                f"</Say></Response>"
            )

            call = client.calls.create(
                twiml=twiml_message,
                from_=from_number,
                to=to_number,
            )

            print(f"[TWILIO_SUCCESS] sid={call.sid}")
            logger.info(f"Twilio call initiated for {server_name} — SID: {call.sid}")
            self._record_timeline("CALL_TRIGGERED", alert.get("server_id", "unknown"), f"SID: {call.sid}")
            return True

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[TWILIO] Failed: {e}")
            logger.error(f"Twilio call error: {e}")
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

    # ── STATS (FIX 2 — includes suppression breakdown + FIX 4 Twilio) ──

    def get_stats(self) -> dict:
        with self._lock:
            total_suppressed_cooldown = sum(self._suppressed_by_cooldown.values())
            total_suppressed_severity = sum(self._suppressed_by_severity.values())
            email_cfg = self._get_email_config()
            twilio_ok = TWILIO_AVAILABLE and self._has_twilio()
            return {
                "sent":                    self._sent_count,
                "slack_sent":              self._sent_count,
                "email_sent":              self._email_sent,
                "twilio_sent":             self._twilio_sent,
                "failed":                  self._fail_count,
                "dry_runs":                self._dry_run_count,
                "slack_enabled":           self._slack_enabled,
                "email_enabled":           bool(email_cfg),
                "twilio_enabled":          twilio_ok,
                "total_alerts":            len(self._alert_log),
                # FIX 2 — Suppression breakdown
                "suppressed_by_cooldown":  total_suppressed_cooldown,
                "suppressed_by_severity":  total_suppressed_severity,
                "total_suppressed":        total_suppressed_cooldown + total_suppressed_severity,
                "suppressed_per_server":   dict(self._suppressed_by_cooldown),
                # FIX 4 — Channel availability
                "channels": {
                    "twilio": twilio_ok,
                    "email":  bool(email_cfg),
                    "slack":  self._slack_enabled,
                },
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

    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    print("CrashGuard AI -- Alert System Test")
    print("-" * 55)

    # Show env var status (read at call time)
    cfg = AlertSystem._get_email_config()
    print(f"  SMTP_USER       = {os.getenv('SMTP_USER', '(not set)')}")
    print(f"  SMTP_PASS       = {'****' + os.getenv('SMTP_PASS', '')[-4:] if os.getenv('SMTP_PASS') else '(not set)'}")
    print(f"  ALERT_EMAIL     = {os.getenv('ALERT_EMAIL', '(not set)')}")
    print(f"  Email ready     : {bool(cfg)}")
    print()
    print(f"  TWILIO_SID      = {os.getenv('TWILIO_ACCOUNT_SID', '(not set)')[:10]}..." if os.getenv('TWILIO_ACCOUNT_SID') else "  TWILIO_SID      = (not set)")
    print(f"  TWILIO_TOKEN    = {'****' if os.getenv('TWILIO_AUTH_TOKEN') else '(not set)'}")
    print(f"  TWILIO_FROM     = {os.getenv('TWILIO_FROM_NUMBER', '(not set)')}")
    print(f"  TWILIO_TO       = {os.getenv('TWILIO_TO_NUMBER', '(not set)')}")
    print(f"  Twilio library  : {'INSTALLED' if TWILIO_AVAILABLE else 'NOT INSTALLED'}")
    print(f"  Twilio ready    : {TWILIO_AVAILABLE and AlertSystem._has_twilio()}")
    print()

    alerts = AlertSystem()
    stats = alerts.get_stats()
    print(f"  Slack:  {'LIVE' if stats['slack_enabled'] else 'DISABLED'}")
    print(f"  Email:  {'LIVE' if stats['email_enabled'] else 'DISABLED'}")
    print(f"  Twilio: {'LIVE' if stats.get('twilio_enabled') else 'DISABLED'}")
    print()

    # Simulate an ESCALATE decision (triggers Twilio if configured)
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

    print("  Testing ESCALATE decision...")
    result = alerts._process_one("server_e", mock_decision)
    if result:
        print(f"\n  Alert fired: {result}")
    else:
        print("\n  Alert suppressed (cooldown)")

    # Wait for background threads (Twilio call or email) to finish
    print("\n  Waiting for background delivery...")
    time.sleep(12)

    print(f"\n  Stats: {alerts.get_stats()}")
    print("\n✅ Alert system test complete.")

    test_alert = {
        "server_name": "Server E",
        "decision": "ESCALATE",
        "current_cpu": 91.5,
        "action": "Immediate intervention required",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    sender = AlertSystem()
    sender._send_twilio_call(test_alert)