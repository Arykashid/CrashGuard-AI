import re
import sys
import os

with open("alert_system.py", "r", encoding="utf-8") as f:
    code = f.read()

# 1. ALERT <-> DECISION MISMATCH
# Remove hardcoded colors/severities in alert_system.py and rely strictly on decision_engine's severity
code = re.sub(
    r'DECISION_EMOJI = \{[\s\S]*?\}',
    '''DECISION_EMOJI = {
    "SCALE":    "📈",
    "SCALE_READY":"⏳",
    "ESCALATE": "🚨",
    "MONITOR":  "👁",
    "STABLE":   "✅",
}''',
    code
)

code = re.sub(
    r'SEVERITY_COLOR = \{[\s\S]*?\}',
    '''SEVERITY_COLOR = {
    "CRITICAL": "#FF3B3B",
    "HIGH":     "#FF8C00",
    "WARNING":  "#FFD700",
    "INFO":     "#00BFFF",
}''',
    code
)

with open("alert_system.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Updated alert_system.py")
