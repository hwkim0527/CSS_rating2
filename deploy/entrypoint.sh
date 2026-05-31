#!/usr/bin/env sh
# Container entrypoint. Open port fast; prefetch adapter in background; base
# model loads lazily on first inference. Must be LF (CRLF breaks dash: set -eu).
set -eu

echo "GPU check:"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || echo "(torch check failed)"

[ "${CSS_ENABLE_LLM:-0}" = "1" ] && [ -n "${CSS_LLM_DRIVE_FOLDER_ID:-}" ] && { echo "prefetching adapter from Drive"; python -m src.web.download_model >/tmp/adapter_dl.log 2>&1 & } || echo "LLM off or no folder id - XGBoost only"
# 주의: 베이스 모델(~29GB) 프리페치를 여기서 백그라운드로 추가하면, 그 다운로드가
# 끝나기 전(config.json 은 맨 마지막에 기록됨) 들어온 요청의 lazy 로드가 같은
# 디렉터리에 동시 다운로드를 시작해 샤드가 손상될 수 있다. 베이스 워밍은 앱 레벨
# 조율(warming/ready 플래그 + 락 + ready 전 503)로 처리해야 안전하다(미구현).

echo "starting web server on 0.0.0.0:${PORT:-8080}"
exec uvicorn src.web.app:app --host 0.0.0.0 --port "${PORT:-8080}"
