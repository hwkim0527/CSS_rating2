#!/usr/bin/env sh
# Container entrypoint. Open the port fast; LLM warm-up (adapter+base download +
# 4bit load) is started in-app at FastAPI startup as a single coordinated
# background thread (src.web.llm_scoring.ensure_warming_started) — NOT here, to
# avoid concurrent-download races. Must be LF (CRLF breaks dash: set -eu).
set -eu

echo "GPU check:"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || echo "(torch check failed)"

# 모델 다운로드(어댑터 + 베이스 ~29GB)와 4bit 로딩은 entrypoint 에서 하지 않는다.
# 앱(src.web.app)이 startup 에서 단일 백그라운드 스레드로 한 번만 조율해 받는다
# (src.web.llm_scoring.ensure_warming_started) — 동시 다운로드 race 와 콜드 요청
# 연결 끊김을 막기 위함. 그래서 여기선 포트만 즉시 연다.
echo "starting web server on 0.0.0.0:${PORT:-8080} (LLM warm-up runs in-app)"
exec uvicorn src.web.app:app --host 0.0.0.0 --port "${PORT:-8080}"
