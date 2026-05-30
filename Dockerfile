FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# System deps (libgomp1 is required by xgboost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Source + trained artifacts
COPY src ./src
COPY frontend ./frontend
COPY artifacts ./artifacts
COPY deploy/entrypoint.sh ./deploy/entrypoint.sh
RUN chmod +x ./deploy/entrypoint.sh

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s \
    CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz').status==200 else 1)"

# 엔트리포인트: LLM 이 켜져 있으면 Drive 어댑터를 먼저 받고 서버 기동.
CMD ["sh", "./deploy/entrypoint.sh"]
