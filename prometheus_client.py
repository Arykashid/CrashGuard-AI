"""
Prometheus Integration Module
Reads CPU metrics from Prometheus server.
Falls back to psutil if Prometheus not available.

Companies use Prometheus to collect metrics
from ALL their servers in one place.
"""

import requests
import psutil
import numpy as np
from datetime import datetime


# ================= CONFIG =================
PROMETHEUS_URL = "http://localhost:9090"  # Change to company's Prometheus URL


# ================= CHECK PROMETHEUS =================
def is_prometheus_available(url=PROMETHEUS_URL):
    """Check if Prometheus is running."""
    try:
        response = requests.get(f"{url}/-/healthy", timeout=3)
        return response.status_code == 200
    except Exception:
        return False


# ================= GET CPU FROM PROMETHEUS =================
def get_cpu_from_prometheus(url=PROMETHEUS_URL, instance="localhost:9100"):
    """
    Gets CPU usage from Prometheus.
    Query: 1 - avg(rate(node_cpu_seconds_total{mode='idle'}[1m]))
    """
    try:
        query = "1 - avg(rate(node_cpu_seconds_total{mode='idle'}[1m]))"
        response = requests.get(
            f"{url}/api/v1/query",
            params={"query": query},
            timeout=5
        )
        data = response.json()

        if data["status"] == "success" and data["data"]["result"]:
            value = float(data["data"]["result"][0]["value"][1])
            return float(np.clip(value, 0.0, 1.0))
        return None
    except Exception:
        return None


def get_cpu_history_from_prometheus(
    url=PROMETHEUS_URL,
    minutes=10
):
    """Gets CPU history from Prometheus for last N minutes."""
    try:
        import time
        end = time.time()
        start = end - (minutes * 60)

        query = "1 - avg(rate(node_cpu_seconds_total{mode='idle'}[1m]))"
        response = requests.get(
            f"{url}/api/v1/query_range",
            params={
                "query": query,
                "start": start,
                "end": end,
                "step": "15s"
            },
            timeout=5
        )
        data = response.json()

        if data["status"] == "success" and data["data"]["result"]:
            values = data["data"]["result"][0]["values"]
            return [float(np.clip(float(v[1]), 0.0, 1.0)) for v in values]
        return []
    except Exception:
        return []


def get_all_servers_cpu(url=PROMETHEUS_URL):
    """Gets CPU for ALL servers monitored by Prometheus."""
    try:
        query = "1 - avg by (instance) (rate(node_cpu_seconds_total{mode='idle'}[1m]))"
        response = requests.get(
            f"{url}/api/v1/query",
            params={"query": query},
            timeout=5
        )
        data = response.json()

        servers = []
        if data["status"] == "success":
            for result in data["data"]["result"]:
                instance = result["metric"].get("instance", "unknown")
                cpu = float(np.clip(float(result["value"][1]), 0.0, 1.0))
                servers.append({
                    "instance": instance,
                    "cpu": cpu,
                    "status": (
                        "Critical" if cpu > 0.85
                        else "Warning" if cpu > 0.65
                        else "Normal"
                    )
                })
        return servers
    except Exception:
        return []


# ================= SMART CPU READER =================
def get_cpu_smart(prometheus_url=PROMETHEUS_URL):
    """
    Smart CPU reader:
    - If Prometheus is available → use Prometheus (real server data)
    - If not → fall back to psutil (local machine)
    """
    if is_prometheus_available(prometheus_url):
        cpu = get_cpu_from_prometheus(prometheus_url)
        if cpu is not None:
            return cpu, "prometheus"

    # Fallback to psutil
    return psutil.cpu_percent(interval=0.1) / 100.0, "psutil"


def get_data_source_status():
    """Returns current data source info."""
    if is_prometheus_available():
        servers = get_all_servers_cpu()
        return {
            "source": "Prometheus",
            "status": "✅ Connected",
            "servers_monitored": len(servers),
            "url": PROMETHEUS_URL
        }
    else:
        return {
            "source": "psutil (local)",
            "status": "✅ Active",
            "servers_monitored": 1,
            "url": "localhost"
        }
