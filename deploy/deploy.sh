#!/usr/bin/env bash
# Cloud Run 일괄 배포 스크립트
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project)}"
REGION="${REGION:-asia-northeast3}"
SERVICE="${SERVICE:-css-rating2}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "PROJECT_ID 환경변수가 비어 있습니다. gcloud config set project <id>" >&2
  exit 1
fi

echo "▶ Building & deploying ${SERVICE} → ${PROJECT_ID} (${REGION})"

gcloud builds submit \
  --config deploy/cloudbuild.yaml \
  --substitutions=_SERVICE=${SERVICE},_REGION=${REGION}

URL=$(gcloud run services describe "${SERVICE}" --region "${REGION}" --format='value(status.url)')
echo "✅ Service URL: ${URL}"
echo "▶ Smoke test: ${URL}/healthz"
curl -sf "${URL}/healthz" && echo "  → OK"
