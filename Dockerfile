# ─────────────────────────────────────────────
# CrashGuard AI — Production Container
# ─────────────────────────────────────────────
FROM python:3.12-slim

# curl required for Docker healthcheck
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 5000

# Runtime defaults
ENV DEMO_MODE=1
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

CMD ["python", "app.py"]