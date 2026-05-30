#!/usr/bin/env bash
# Cloud Run GPU 배포용 서비스 계정 1회 설정.
#
# 이 스크립트가 하는 일:
#   1) Drive API 활성화
#   2) 전용 런타임 서비스 계정(css-rating2-run) 생성
#   3) 그 SA 이메일을 출력 → 이 이메일로 Drive 의 Qwen3_fintech 폴더를
#      "뷰어"로 공유해야 함 (수동, Drive 웹 UI에서)
#
# 키 파일을 만들지 않으므로(ADC 사용) 키 유출 위험이 없다.
#
# 사용:
#   PROJECT_ID=my-proj bash deploy/setup_service_account.sh
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
SA_NAME="${SA_NAME:-css-rating2-run}"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

if [ -z "${PROJECT_ID}" ]; then
  echo "PROJECT_ID 가 비어 있습니다. PROJECT_ID=<id> 로 지정하거나 gcloud config set project <id>" >&2
  exit 1
fi

echo "▶ 프로젝트: ${PROJECT_ID}"

echo "▶ Drive API 활성화"
gcloud services enable drive.googleapis.com --project "${PROJECT_ID}"

echo "▶ Cloud Run / Build API 활성화"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com --project "${PROJECT_ID}"

if gcloud iam service-accounts describe "${SA_EMAIL}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
  echo "▶ 서비스 계정 이미 존재: ${SA_EMAIL}"
else
  echo "▶ 서비스 계정 생성: ${SA_EMAIL}"
  gcloud iam service-accounts create "${SA_NAME}" \
    --project "${PROJECT_ID}" \
    --display-name "CSS Rating2 Cloud Run runtime (Drive read)"
fi

cat <<EOF

============================================================
✅ 서비스 계정 준비 완료

다음 단계 (수동, 1회):

1) Google Drive 웹에서 'Qwen3_fintech' 폴더 우클릭 → 공유 →
   아래 이메일을 '뷰어'로 추가:

        ${SA_EMAIL}

2) 배포 (GPU, 서울 미지원이라 us-central1):

   gcloud builds submit --config deploy/cloudbuild_gpu.yaml \\
     --substitutions _REGION=us-central1,_RUNTIME_SA=${SA_EMAIL}

   → 컨테이너가 ADC(런타임 SA)로 Drive 의 어댑터를 받아 Qwen3-14B 로 채점합니다.
============================================================
EOF
