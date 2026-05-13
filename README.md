# CrashGuard AI

### Autonomous CPU Workload Forecasting and Infrastructure Decision System

![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-orange?logo=tensorflow)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-blue)
![Twilio](https://img.shields.io/badge/Twilio-Voice_Escalation-red?logo=twilio)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

---

CrashGuard predicts CPU overload events before they happen and responds autonomously — scaling infrastructure, restarting services, or escalating to on-call engineers via phone call — without waiting for human intervention.

---

## Screenshots

### Dashboard
![Dashboard — live decision banner, CPU chart with prediction overlay, fleet status, incident timeline](assets/screenshots/dashboard.png)

### Systems
![Systems — per-server CPU utilization, trend direction, live decision badges](assets/screenshots/systems.png)

### Alerts
![Alerts — alert log with decision routing, severity tags, suppression stats](assets/screenshots/alerts.png)

### Models
![Models — calibration proof, live trust indicators, ensemble pipeline diagram](assets/screenshots/models.png)

### Predictions
![Predictions — per-server forecasts, operational risk scores, recommended actions](assets/screenshots/predictions.png)

---

## Architecture

```
Raw CPU Telemetry (2s tick rate)
         │
         ▼
┌─────────────────────┐
│   Feature Engine     │  15 signals: lags, rolling stats,
│                      │  cyclical time encoding, delta, spike_flag
└──────────┬──────────┘
           │
           ▼
┌──────────────────────────────┐
│   LSTM + XGBoost Ensemble    │  MC Dropout uncertainty bounds
│   60/40 dynamic weighting    │  Temperature scaling T = 9.5518
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│   Decision Engine v5         │  Graduated escalation
│   Risk score + hysteresis    │  Confidence gating
│   Adaptive weights           │  State machine enforcement
└──────┬───────────────────────┘
       │
  ┌────┴────────────────┐
  ▼                     ▼
Autoscale/Restart    📞 Twilio Call
                     📧 Email Alert
```

---

## Model Performance

| Metric | Value |
|---|---|
| Ensemble RMSE | 0.1578 |
| XGBoost RMSE | 0.1337 |
| LSTM RMSE | 0.2328 |
| Diebold-Mariano p-value | 0.0086 (ensemble significantly outperforms individual models) |
| 80% CI Coverage | 80.0% (calibrated — matches claimed interval) |
| Calibration Temperature | T = 9.5518 (Platt scaling) |
| Training Data | ~60,000 records from Google Cluster Workload Traces |
| Engineered Features | 15 signals per timestep |
| MC Dropout Samples | 15 forward passes per prediction |

---

## Decision Engine

| Decision | Trigger Condition | Action |
|---|---|---|
| **STABLE** | Risk < 0.4, no spikes | Dashboard monitoring only |
| **MONITOR** | Predicted CPU > 65% or risk > 0.4 | Increase telemetry frequency |
| **RESTART** | 3+ spikes in 10 min, high volatility | Graceful service restart |
| **SCALE** | CPU ≥ 85% or predicted > 81% at 65%+ confidence | Horizontal autoscale |
| **ESCALATE** | SCALE persisted 30s OR CPU > 90% | Phone call + email to on-call |

**Key properties:**

- **Graduated escalation:** STABLE → MONITOR → SCALE → ESCALATE (no level skipping)
- **Hysteresis:** prevents decision flapping within 60-second windows
- **Confidence gating:** >70% model confidence for autonomous action; <50% forces human review
- **Hard override:** CPU ≥ 90% bypasses all gates → immediate ESCALATE

---

## Alert Routing

| Decision | Channel | Cooldown |
|---|---|---|
| **ESCALATE** | Twilio phone call + Email | 60 seconds |
| **SCALE** | Email only | 5 minutes |
| **RESTART** | Email only | 5 minutes |
| **MONITOR** | Dashboard only | — |
| **STABLE** | No alert | — |

All channels include per-server cooldown enforcement to prevent alert fatigue. Alerts are suppressed (not lost) during cooldown — suppression counts are visible on the Alerts page.

---

## Dashboard Pages

1. **Dashboard** — Live decision banner, CPU chart with prediction overlay, incident timeline, fleet CPU bars, system health panel
2. **Systems** — Per-server CPU utilization, trend direction, decision badges, model source
3. **Alerts** — Alert log with severity tags, suppression stats, cooldown/deduplication metrics
4. **Models** — Calibration proof (80% CI bar), live trust indicators (PASS/WARN/FAIL), end-to-end pipeline diagram
5. **Predictions** — Per-server forecasts, confidence intervals, operational risk scores, recommended actions

---

## Quick Start

### Option 1: Python

```bash
git clone https://github.com/Arykashid/CrashGuard-AI
cd CrashGuard-AI
pip install -r requirements.txt
cp .env.example .env    # fill in credentials (optional — runs in dry-run mode without them)
python app.py
# Open http://localhost:5000
```

### Option 2: Docker (recommended)

```bash
git clone https://github.com/Arykashid/CrashGuard-AI
cd CrashGuard-AI
cp .env.example .env    # fill in credentials
docker-compose up --build
# Open http://localhost:5000
```

---

## Demo Walkthrough

1. Open `http://localhost:5000` — wait ~90 seconds for LSTM model warmup
2. Dashboard shows all 5 servers with live CPU bars and decisions
3. Click **Burst C** in the Demo Controls panel (bottom-right)
4. Watch the hero banner escalate: **STABLE → MONITOR → SCALE → ESCALATE**
5. Terminal logs show `[TWILIO] ✅ Call initiated` — phone rings within 10 seconds
6. Navigate to **Alerts** — see the alert log with severity, routing, and suppression stats
7. Navigate to **Predictions** — watch operational risk rise then fall as the server recovers
8. Click **Normalize A** — system recovers autonomously, decision returns to STABLE
9. Navigate to **Models** — verify calibration proof and live trust indicators

---

## Project Structure

```
CrashGuard-AI/
├── app.py                    # Flask backend + REST API
├── server_simulator.py       # 5-server workload simulator
├── feature_engine.py         # 15-signal feature pipeline
├── pipeline.py               # LSTM + XGBoost ensemble inference
├── decision_engine.py        # Autonomous decision engine v5
├── alert_system.py           # Multi-channel alert dispatch
├── crashguard_dashboard.html # 5-page SPA dashboard
├── assets/
│   ├── style.css             # Dashboard styles
│   ├── dashboard.js          # Dashboard logic
│   └── screenshots/          # README screenshots
├── models/                   # Trained model artifacts
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .env.example
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `DEMO_MODE` | No | `1` = deterministic demo, `0` = live random. Default: `1` |
| `TWILIO_ACCOUNT_SID` | No | Twilio account identifier |
| `TWILIO_AUTH_TOKEN` | No | Twilio authentication token |
| `TWILIO_FROM_NUMBER` | No | Twilio phone number (caller) |
| `TWILIO_TO_NUMBER` | No | On-call engineer phone number |
| `SMTP_USER` | No | Gmail address for sending alerts |
| `SMTP_PASS` | No | Gmail App Password (16 characters) |
| `ALERT_EMAIL` | No | Alert destination email |
| `SLACK_WEBHOOK_URL` | No | Slack incoming webhook URL |

All alert channels are optional. When credentials are absent, the system runs in **dry-run mode** — decisions are logged to the console but no external notifications are sent.

---

## Cloud Deployment

CrashGuard runs as a single Docker container. Deploy directly from the repository on any platform that supports Docker:

### Render

1. Connect your GitHub repo at [render.com](https://render.com)
2. Select **Web Service** → **Docker** runtime
3. Set environment variables in the Render dashboard
4. Deploy — Render builds from the Dockerfile automatically

### Railway

1. Connect your GitHub repo at [railway.app](https://railway.app)
2. Railway auto-detects the Dockerfile
3. Add environment variables in the Railway dashboard
4. Deploy — accessible via generated URL

**Startup command** (if needed): `python app.py`
**Port**: `5000`
**Health endpoint**: `GET /health`

---

## Why CrashGuard vs Reactive Monitoring

- **Predictive, not reactive.** Tools like Datadog and Grafana alert *after* a threshold is breached. CrashGuard forecasts the failure trajectory 60 seconds ahead and acts *before* impact.
- **Autonomous first-response.** No human required for initial scaling or restart. Mean time to response drops from minutes to seconds.
- **Self-aware model.** The system exposes prediction reliability in real time and gates autonomous actions when model confidence is low — preventing unsafe automated interventions.

---

## Limitations

- Model trained on [Google Cluster Workload Traces](https://github.com/google/cluster-data) — performance on workloads with different characteristics (e.g., GPU-bound, IO-heavy) has not been validated.
- The 5-server simulator generates synthetic CPU patterns for demonstration. Production deployment requires integration with real telemetry sources (e.g., `psutil`, Prometheus, or cloud provider APIs).
- Trial Twilio accounts are limited to verified phone numbers only.

---

## Future Work

- Real telemetry ingestion via `psutil` or Prometheus scraping
- PostgreSQL persistence for incident history and audit logs
- Kubernetes HPA integration for live infrastructure scaling
- Multi-cluster fleet monitoring
- Online model retraining pipeline on production data

---

## Citation

Training data: [Google Cluster Workload Traces v2](https://github.com/google/cluster-data) (Reiss et al., 2011)

---

## Deployment Checklist

### Local

- [ ] `python app.py` starts without errors
- [ ] `http://localhost:5000` loads the dashboard
- [ ] `GET /health` returns `200 OK`
- [ ] All 5 sidebar pages render correctly
- [ ] Demo Controls panel visible (bottom-right)
- [ ] Burst C triggers ESCALATE within 3 minutes
- [ ] Phone rings when ESCALATE fires (requires Twilio credentials)
- [ ] Email arrives for SCALE decision (requires SMTP credentials)

### Docker

- [ ] `docker-compose up --build` completes successfully
- [ ] `http://localhost:5000` loads the dashboard
- [ ] `GET /health` returns `200 OK`
- [ ] Environment variables injected from `.env`
