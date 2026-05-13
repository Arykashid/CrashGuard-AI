# CrashGuard AI

### Autonomous CPU Workload Forecasting and Infrastructure Decision System

![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Backend-black?logo=flask)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-orange?logo=tensorflow)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-blue?logo=xgboost)
![Twilio](https://img.shields.io/badge/Twilio-Escalation-red?logo=twilio)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

---

CrashGuard predicts CPU failures before they happen and responds autonomously — scaling infrastructure, restarting services, or escalating to on-call engineers — without waiting for human intervention.

---

## Architecture

```
Raw CPU Telemetry (2s tick rate)
         │
         ▼
┌─────────────────────┐
│   Feature Engine     │  15 signals: lags, rolling stats,
│                      │  cyclical time, delta, spike_flag
└──────────┬──────────┘
           │
           ▼
┌──────────────────────────────┐
│   LSTM + XGBoost Ensemble    │  MCDropout uncertainty bounds
│   60/40 dynamic weighting    │  Temperature scaling T=9.5518
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

## Model Metrics

| Metric | Value |
|---|---|
| Ensemble RMSE | 0.1578 |
| XGBoost RMSE | 0.1337 |
| LSTM RMSE | 0.2328 |
| DM Test p-value | 0.0086 |
| 80% CI Coverage | 80.0% (perfectly calibrated) |
| Calibration Temperature | T = 9.5518 |
| Training Samples | ~60,000 Google Cluster records |
| Features | 15 engineered signals |
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

- **Graduated escalation:** STABLE → MONITOR → SCALE → ESCALATE (no skipping)
- **Hysteresis:** prevents decision flapping within 60 seconds
- **Confidence gating:** >70% confidence for autonomous action, <50% forces human escalation
- **Hard override:** CPU ≥ 90% bypasses all gates and forces ESCALATE immediately

---

## Alert Routing

| Decision | Channel | Cooldown |
|---|---|---|
| **ESCALATE** | Twilio phone call + Email | 60 seconds |
| **SCALE** | Email only | 5 minutes |
| **RESTART** | Email only | 5 minutes |
| **MONITOR** | Dashboard only | — |
| **STABLE** | No alert | — |

---

## Dashboard Pages

1. **Dashboard** — Live decision banner, CPU chart with prediction zone, incident timeline, fleet status, system health
2. **Systems** — Per-server CPU bars, trend labels, CRITICAL RISK badges, live decision badges
3. **Alerts** — Evidence at detection, escalation paths, fallback branches, root cause hypothesis
4. **Models** — Calibration proof (80% CI), trust indicators (PASS/WARN/FAIL), pipeline diagram
5. **Predictions** — Per-server forecasts, CI ranges, crash risk scores, recommended actions

---

## Quick Start

### Option 1: Direct Python

```bash
git clone https://github.com/Arykashid/CrashGuard-AI
cd CrashGuard-AI
pip install -r requirements.txt
cp .env.example .env
# Fill in .env with your credentials
set DEMO_MODE=1   # Windows
python app.py
# Open http://localhost:5000
```

### Option 2: Docker (recommended)

```bash
cp .env.example .env
# Fill in .env with your credentials
docker-compose up
# Open http://localhost:5000
```

---

## Demo Walkthrough

1. Open `http://localhost:5000` — wait 90 seconds for model warmup
2. Dashboard shows all 5 servers with live CPU bars
3. Click **Burst C** in Demo Controls panel
4. Watch hero banner escalate: **MONITOR → RESTART → SCALE → ESCALATE**
5. Terminal shows `[TWILIO] ✅ Call initiated` — phone rings within 10 seconds
6. Go to **Alerts** page — see Evidence at Detection, Thresholds Crossed, Fallback branches
7. Go to **Predictions** page — watch crash risk rise then fall
8. Click **Normalize A** — system recovers autonomously
9. Go to **Models** page — verify calibration proof and trust indicators

---

## Project Structure

```
CrashGuard-AI/
├── app.py                    # Flask backend, API endpoints
├── server_simulator.py       # 5 server behavior profiles
├── feature_engine.py         # 15-signal feature pipeline
├── pipeline.py               # LSTM + XGBoost ensemble
├── decision_engine.py        # Autonomous decision engine v5
├── alert_system.py           # Multi-channel alert dispatch
├── crashguard_dashboard.html # 5-page SPA dashboard
├── assets/
│   ├── style.css
│   └── dashboard.js
├── models/                   # Trained model artifacts
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── requirements.txt
└── README.md
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `DEMO_MODE` | No | `1` = stable demo, `0` = live random |
| `TWILIO_ACCOUNT_SID` | No | Twilio account identifier |
| `TWILIO_AUTH_TOKEN` | No | Twilio authentication token |
| `TWILIO_FROM_NUMBER` | No | Your Twilio phone number |
| `TWILIO_TO_NUMBER` | No | On-call engineer number |
| `SMTP_USER` | No | Gmail address for alerts |
| `SMTP_PASS` | No | Gmail App Password (16 chars) |
| `ALERT_EMAIL` | No | Alert destination email |

---

## Why CrashGuard vs Reactive Monitoring

- **Predictive, not reactive.** Reactive tools (Datadog, Grafana) alert *after* a failure occurs. CrashGuard predicts the failure trajectory 60 seconds ahead and acts *before* impact.
- **Autonomous response.** No human required for first-response scaling or restart. Mean time to response drops from minutes to seconds.
- **Self-aware model.** The system exposes prediction reliability in real time and gates autonomous actions when model confidence is low — preventing unsafe interventions.

---

## Limitations

- Model trained on Google Cluster Workload Traces — performance on workloads with different characteristics may vary.
- Simulator generates synthetic CPU patterns for demo; production deployment requires real telemetry integration.
- Trial Twilio account limited to verified phone numbers.

---

## Future Work

- Real telemetry ingestion via `psutil` or Prometheus scraping
- PostgreSQL persistence for incident history
- Kubernetes HPA integration for actual infrastructure scaling
- Multi-cluster support
- Model retraining pipeline on live data

---

## Citation

Training data: [Google Cluster Workload Traces v2](https://github.com/google/cluster-data) (Reiss et al., 2011)

---

## Deployment Checklist

### Local

- [ ] `python app.py` starts without errors
- [ ] `http://localhost:5000` loads dashboard
- [ ] `/health` returns 200
- [ ] All 5 sidebar pages work
- [ ] Demo Controls visible and clickable
- [ ] Burst C triggers ESCALATE within 3 minutes
- [ ] Phone rings when ESCALATE fires
- [ ] Email arrives for SCALE decision

### Docker

- [ ] `docker-compose up` builds successfully
- [ ] `http://localhost:5000` loads dashboard
- [ ] `/health` returns 200
- [ ] All env vars passed correctly
