# Multi-stage Dockerfile for FinSight Multi-Agent Financial Research Analyst
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir --user .

# Production Runtime Stage
FROM python:3.11-slim AS runner

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# Create non-root user
RUN useradd -m -u 1000 finsight && chown -R finsight:finsight /app
USER finsight

# Copy application files
COPY --chown=finsight:finsight config/ ./config/
COPY --chown=finsight:finsight src/ ./src/
COPY --chown=finsight:finsight ui/ ./ui/
COPY --chown=finsight:finsight demo/ ./demo/
COPY --chown=finsight:finsight docs/ ./docs/
COPY --chown=finsight:finsight scripts/ ./scripts/

# Seed reproducible offline fixtures rather than relying on local or LFS pointer files.
RUN FINSIGHT_OFFLINE_DATA=1 python scripts/seed_market_data.py

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default startup runs both FastAPI backend and Streamlit UI
CMD ["sh", "-c", "uvicorn finsight.api.server:app --host 0.0.0.0 --port 8000 & streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0"]
