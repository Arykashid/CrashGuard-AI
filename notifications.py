"""
Slack Notifications Module
Sends CPU spike alerts to Slack channel.

SECURITY FIX:
  Webhook URL is now read from environment variable SLACK_WEBHOOK_URL.
  Never hardcode secrets in source code — they get committed to GitHub
  and exposed publicly.

  To set it:
    export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
  Or add to .env file (python-dotenv):
    SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
"""

import requests
import json
import os
from datetime import datetime


# ================= SLACK CONFIG =================
# Read from environment — never hardcode
# Set via: export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")


# ================= SEND ALERT =================
def send_slack_alert(level, predicted_cpu, current_cpu,
                     action, webhook_url=None):
    """
    Sends CPU spike alert to Slack.

    level:         HIGH / MEDIUM / LOW
    predicted_cpu: float 0–1
    current_cpu:   float 0–1
    action:        string describing recommended action
    webhook_url:   optional override (e.g. from Streamlit sidebar input)
    """
    url = webhook_url or SLACK_WEBHOOK_URL

    if not url:
        print("⚠️  No Slack webhook configured. "
              "Set SLACK_WEBHOOK_URL env variable.")
        return False

    emoji         = "🔴" if level == "HIGH" else "🟡" if level == "MEDIUM" else "🟢"
    time_estimate = "~2 minutes" if level == "HIGH" else "~5 minutes"

    message = {
        "text": f"{emoji} *CPU SPIKE PREDICTED — {level}*",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} CPU SPIKE PREDICTED — {level}"
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": "*Server:*\nnode-1"},
                    {"type": "mrkdwn", "text": f"*Current CPU:*\n{current_cpu:.1%}"},
                    {"type": "mrkdwn",
                     "text": f"*Predicted CPU:*\n{predicted_cpu:.1%} "
                             f"(in {time_estimate})"},
                    {"type": "mrkdwn",
                     "text": f"*Time:*\n{datetime.now().strftime('%H:%M:%S')}"}
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Suggested Action:*\n{action}"
                }
            },
            {"type": "divider"},
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "🛡️ CrashGuard AI — CPU Workload Forecasting System"
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(
            url,
            data=json.dumps(message),
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        return response.status_code == 200
    except Exception as e:
        print(f"Slack error: {e}")
        return False


def send_test_message(webhook_url=None):
    """Sends a test message to verify the webhook is working."""
    url = webhook_url or SLACK_WEBHOOK_URL
    if not url:
        return False
    message = {
        "text": "✅ CrashGuard AI connected! Slack alerts are working."
    }
    try:
        response = requests.post(
            url,
            data=json.dumps(message),
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        return response.status_code == 200
    except Exception:
        return False
