#!/usr/bin/env sh
# Container entrypoint. Open port fast; prefetch adapter in background; base
# model loads lazily on first inference. Must be LF (CRLF breaks dash: set -eu).
set -eu

echo "GPU check:"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || echo "(torch check failed)"

[ "${CSS_ENABLE_LLM:-0}" = "1" ] && [ -n "${CSS_LLM_DRIVE_FOLDER_ID:-}" ] && { echo "prefetching adapter from Drive"; python -m src.web.download_model >/tmp/adapter_dl.log 2>&1 & } || echo "LLM off or no folder id - XGBoost only"

echo "starting web server on 0.0.0.0:${PORT:-8080}"
exec uvicorn src.web.app:app --host 0.0.0.0 --port "${PORT:-8080}"
