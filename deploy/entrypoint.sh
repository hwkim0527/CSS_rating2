#!/usr/bin/env sh
# 컨테이너 시작 엔트리포인트.
# CSS_ENABLE_LLM=1 이고 CSS_LLM_DRIVE_FOLDER_ID 가 있으면, 웹 서버 기동 전에
# Google Drive(Qwen3_fintech)에서 LoRA 어댑터를 미리 내려받는다.
# (받지 못해도 앱은 기동 — XGBoost 경로는 항상 동작.)
set -eu

if [ "${CSS_ENABLE_LLM:-0}" = "1" ] && [ -n "${CSS_LLM_DRIVE_FOLDER_ID:-}" ]; then
  echo "▶ LLM 활성화됨 — Drive 에서 어댑터 사전 다운로드 시도"
  python -m src.web.download_model || \
    echo "⚠ 어댑터 사전 다운로드 실패 — 첫 추론 시 재시도하거나 XGBoost 로 동작합니다."
else
  echo "▶ LLM 비활성화(또는 폴더 ID 미설정) — XGBoost 경로로 기동합니다."
fi

exec uvicorn src.web.app:app --host 0.0.0.0 --port "${PORT:-8080}"
