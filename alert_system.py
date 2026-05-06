"""
alert_system.py — CrashGuard AI (Production Grade)
Unified alert engine: Slack + Email + Cooldown + Layered Fallback.

Alerts are DERIVED from DecisionEngine output — zero independent logic.

Delivery architecture:
  try: send_slack()
  except: send_email()
  else: log_dry_run()

FIX 2 — Realistic alert deduplication with per-server suppression tracking.
FIX 3 — Gmail SMTP email alerts for CRITICAL + HIGH severity.
         Priority: Slack → Email → DRY-RUN log.
         Non-blocking with exponential backoff retry.

Cooldown: 300s (5 minutes) per server per channel.
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

# FIX 2 — Cooldown set to 5 minutes (300s) per server per channel
ALERT_COOLDOWN_SECONDS = 300  # 5 minutes per server
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

# FIX 3 — Email fires for CRITICAL and HIGH severity (ESCALATE + SCALE)
ALERTABLE_DECISIONS = {"ESCALATE", "SCALE"}
EMAIL_DECISIONS     = {"ESCALATE", "SCALE"}  # FIX 3: both CRITICAL and HIGH

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

    def can_send(self, server_id: str) -> bool:
        with self._lock:
            elapsed = time.time() - self._last.get(server_id, 0.0)
            return elapsed >= self._cooldown

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
# EMAIL BUILDER (FIX 3 — reads env at call time)
# ─────────────────────────────────────────────

def build_email(alert: dict, smtp_user: str, email_to: str) -> MIMEMultipart:
    """
    Build MIME email for CRITICAL and HIGH severity alerts.
    smtp_user and email_to are passed in explicitly so we never
    depend on module-level variables.
    """
    subject = f"[CrashGuard] {alert['decision']} — {alert['server_name']} — CPU {alert['cpu']:.1f}%"

    body = (
        "CrashGuard AI Alert\n"
        "===================\n"
        f"Server:     {alert['server_name']}\n"
        f"Decision:   {alert['decision']}\n"
        f"Severity:   {alert['severity']}\n"
        f"Current CPU: {alert['cpu']:.1f}%\n"
        f"Predicted:  {alert['predicted_cpu']:.1f}%\n"
        f"Confidence: {alert['confidence']:.0%}\n"
        f"Crash Risk: {alert['risk']:.0%}\n"
        "\n"
        f"Reason: {alert['reason']}\n"
        f"Action: {alert['action']}\n"
        "\n"
        f"Timestamp: {alert['timestamp']}\n"
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
        self._cooldown        = ChannelCooldown(ALERT_COOLDOWN_SECONDS)
        self._lock            = threading.Lock()
        self._sent_count      = 0
        self._fail_count      = 0
        self._email_sent      = 0
        self._dry_run_count   = 0
        self._alert_log: list[dict] = []

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
            print("[ALERT] ⚠ No email env vars detected at startup. Set SMTP_USER/SMTP_PASS/ALERT_EMAIL.")

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
        severity = SEVERITY_MAP.get(dec, "INFO")

        # FIX 2 — Step 1: Check cooldown FIRST (before severity)
        if not self._cooldown.can_send(server_id):
            remaining = self._cooldown.seconds_remaining(server_id)
            logger.debug(f"Alert suppressed for {server_id} — cooldown {remaining}s remaining")
            with self._lock:
                self._suppressed_by_cooldown[server_id] = self._suppressed_by_cooldown.get(server_id, 0) + 1
                self._total_suppressed_session += 1
            return None

        # FIX 2 — Step 2: Only alert on actionable decisions (severity filter)
        if dec not in ALERTABLE_DECISIONS:
            if severity not in ("CRITICAL", "HIGH"):
                # Only track suppression for non-INFO decisions
                if dec not in ("STABLE",):
                    with self._lock:
                        self._suppressed_by_severity[server_id] = self._suppressed_by_severity.get(server_id, 0) + 1
                        self._total_suppressed_session += 1
            return None

        # FIX 2 — Sanity cap: if we've suppressed > 50 this session, log warning
        if self._total_suppressed_session > MAX_SUPPRESSIONS_PER_SESSION:
            logger.warning(
                f"Suppression sanity check: {self._total_suppressed_session} suppressions this session. "
                f"Cooldown logic may need review."
            )

        alert = build_structured_alert(decision)

        # ── LAYERED DELIVERY (FIX 3) ──────────────────────
        # Priority: Slack first → Email second → DRY-RUN third
        slack_sent = False
        email_sent = False
        dry_run    = False

        # Layer 1: Try Slack
        if self._slack_enabled:
            try:
                self._send_slack_async(alert)
                slack_sent = True
            except Exception as e:
                logger.error(f"Slack send failed for {server_id}: {e}")

        # Layer 2: Email fallback (or additional for CRITICAL/HIGH)
        # FIX 3 — fires for CRITICAL and HIGH severity
        # Re-read env vars NOW so late-set vars are picked up
        if not slack_sent or dec in EMAIL_DECISIONS:
            email_cfg = self._get_email_config()
            if email_cfg:
                try:
                    self._send_email_async(alert, email_cfg)
                    email_sent = True
                except Exception as e:
                    logger.error(f"Email send failed for {server_id}: {e}")
                    print(f"[EMAIL] ❌ Failed to queue email for {server_id}: {e}")

        # Layer 3: DRY-RUN log if nothing sent
        if not slack_sent and not email_sent:
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

    def _send_email_async(self, alert: dict, email_cfg: dict):
        """Non-blocking: runs email delivery in background thread."""
        t = threading.Thread(
            target=self._send_email_with_retry,
            args=(alert, email_cfg),
            daemon=True,
        )
        t.start()

    def _send_email_with_retry(self, alert: dict, email_cfg: dict):
        """
        3 attempts with 0.5s/1s/2s exponential backoff.
        Prints to console with timing for latency measurement.
        """
        t0 = time.time()
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
                print(f"[EMAIL] ✅ Sent successfully to {email_cfg['to']} for {alert['server_name']} ({total_ms}ms)")
                return
            if attempt < MAX_RETRIES:
                wait = 0.5 * (2 ** (attempt - 1))  # 0.5s, 1s, 2s
                logger.warning(f"Email failed (attempt {attempt}/{MAX_RETRIES}, {duration_ms}ms) — retry in {wait:.1f}s")
                print(f"[EMAIL] ⚠ Attempt {attempt}/{MAX_RETRIES} failed ({duration_ms}ms) — retrying in {wait:.1f}s...")
                time.sleep(wait)
            else:
                total_ms = int((time.time() - t0) * 1000)
                logger.error(f"Email failed after {MAX_RETRIES} attempts for {alert['server_id']} ({total_ms}ms) — falling back to DRY-RUN")
                print(f"[EMAIL] ❌ Failed after {MAX_RETRIES} attempts for {alert['server_name']} ({total_ms}ms) — falling back to DRY-RUN")
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
            with smtplib.SMTP(email_cfg["host"], email_cfg["port"], timeout=10) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(email_cfg["user"], email_cfg["pass"])
                server.send_message(msg)
            return True
        except Exception as e:
            logger.error(f"SMTP error: {e}")
            print(f"[EMAIL] ❌ SMTP error: {e}")
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

    # ── STATS (FIX 2 — includes suppression breakdown) ──

    def get_stats(self) -> dict:
        with self._lock:
            total_suppressed_cooldown = sum(self._suppressed_by_cooldown.values())
            total_suppressed_severity = sum(self._suppressed_by_severity.values())
            email_cfg = self._get_email_config()
            return {
                "slack_sent":              self._sent_count,
                "email_sent":              self._email_sent,
                "failed":                  self._fail_count,
                "dry_runs":                self._dry_run_count,
                "slack_enabled":           self._slack_enabled,
                "email_enabled":           bool(email_cfg),
                "total_alerts":            len(self._alert_log),
                # FIX 2 — Suppression breakdown
                "suppressed_by_cooldown":  total_suppressed_cooldown,
                "suppressed_by_severity":  total_suppressed_severity,
                "total_suppressed":        total_suppressed_cooldown + total_suppressed_severity,
                "suppressed_per_server":   dict(self._suppressed_by_cooldown),
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

    # Show env var status (read at call time)
    cfg = AlertSystem._get_email_config()
    print(f"  SMTP_USER  = {os.getenv('SMTP_USER', '(not set)')}")
    print(f"  SMTP_PASS  = {'****' + os.getenv('SMTP_PASS', '')[-4:] if os.getenv('SMTP_PASS') else '(not set)'}")
    print(f"  ALERT_EMAIL= {os.getenv('ALERT_EMAIL', '(not set)')}")
    print(f"  Email ready: {bool(cfg)}")
    print()

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

    # Wait for background email thread to finish
    time.sleep(8)

    print(f"\n  Stats: {alerts.get_stats()}")
    print("\n✅ Alert system test complete.")