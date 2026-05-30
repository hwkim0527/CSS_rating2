#!/usr/bin/env sh
# 컨테이너 시작 엔트리포인트.
#
# 중요: Cloud Run 은 컨테이너가 빨리 PORT 를 열어 헬스체크에 응답하길 요구한다.
# 따라서 어댑터 다운로드로 기동을 막지 않는다 — uvicorn 을 즉시 띄우고,
# 다운로드는 백그라운드로 미리 받아둔다(첫 추론 지연 완화). 받지 못해도
# llm_scoring 이 첫 추론 시 다시 시도하며, XGBoost 경로는 항상 동작한다.
set -eu

# --- GPU 진단 (CUDA 드라이버/torch 호환 확인용) ---
echo "▶ GPU 진단:"
python - <<'PYDIAG' || echo "  (torch 진단 실패 — 위 메시지 참고)"
import torch
print("  torch:", torch.__version__)
print("  CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  device:", torch.cuda.get_device_name(0))
    print("  capability:", torch.cuda.get_device_capability(0))
PYDIAG

if [ "${CSS_ENABLE_LLM:-0}" = "1" ] && [ -n "${CSS_LLM_DRIVE_FOLDER_ID:-}" ]; then
  echo "▶ LLM 활성화됨 — Drive 어댑터를 백그라운드로 사전 다운로드"
  (
    python -m src.web.download_model \
      && echo "✔ 어댑터 사전 다운로드 완료" \
      || echo "⚠ 어댑터 사전 다운로드 실패 — 첫 추론 시 재시도(또는 XGBoost)"
  ) &
else
  echo "▶ LLM 비활성화(또는 폴더 ID 미설정) — XGBoost 경로로 기동합니다."
fi

echo "▶ 웹 서버 기동: 0.0.0.0:${PORT:-8080}"
exec uvicorn src.web.app:app --host 0.0.0.0 --port "${PORT:-8080}"
