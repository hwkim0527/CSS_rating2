"""FastAPI 신용평가 웹 서비스."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.utils.config import REPO_ROOT
from src.web.schemas import CompareResponse, ScoreRequest, ScoreResponse
from src.web.scoring import load_metrics, score_one

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("app")

TEMPLATES_DIR = REPO_ROOT / "frontend" / "templates"
STATIC_DIR = REPO_ROOT / "frontend" / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # LLM 이 켜져 있으면 기동 직후 백그라운드 워밍을 시작한다(서버는 즉시 서빙 시작).
    # 베이스 모델 다운로드+로딩이 단일 스레드로 한 번만 일어나 콜드 요청 끊김/동시
    # 다운로드 race 를 막는다. CPU 호스트(LLM off)에서는 no-op.
    try:
        from src.web.llm_scoring import LLM_ENABLED, ensure_warming_started

        if LLM_ENABLED:
            log.info("LLM warm-up 시작 (startup): %s", ensure_warming_started())
    except Exception:  # noqa: BLE001
        log.exception("LLM warm-up 시작 실패 (서버는 계속 기동)")
    yield


app = FastAPI(
    title="CSS Rating 2 — AI 신용평가시스템",
    description="개인 신용정보를 입력하면 부실 확률을 산출합니다.",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/compare", response_class=HTMLResponse)
def compare_page(request: Request) -> HTMLResponse:
    metrics = load_metrics()
    return templates.TemplateResponse(
        "compare.html", {"request": request, "metrics": metrics}
    )


@app.post("/api/score", response_model=ScoreResponse)
def api_score(req: ScoreRequest) -> ScoreResponse:
    try:
        result = score_one(req.model_dump(), model_key="xgboost")
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:  # pragma: no cover
        log.exception("scoring failed")
        raise HTTPException(status_code=500, detail=f"scoring failed: {e}") from e
    return ScoreResponse(**result)


@app.post("/api/score_baseline", response_model=ScoreResponse)
def api_score_baseline(req: ScoreRequest) -> ScoreResponse:
    try:
        result = score_one(req.model_dump(), model_key="logistic")
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return ScoreResponse(**result)


@app.post("/api/score_llm", response_model=ScoreResponse)
def api_score_llm(req: ScoreRequest) -> ScoreResponse:
    try:
        result = score_one(req.model_dump(), model_key="llm")
    except (FileNotFoundError, RuntimeError) as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return ScoreResponse(**result)


@app.get("/api/llm_status")
def api_llm_status() -> dict:
    from src.web.llm_scoring import ADAPTER_DIR, LLM_ENABLED, llm_available, warm_status
    ws = warm_status()
    return {
        "enabled_flag": LLM_ENABLED,
        "adapter_path": str(ADAPTER_DIR),
        "adapter_present": ADAPTER_DIR.exists() and any(ADAPTER_DIR.iterdir()) if ADAPTER_DIR.exists() else False,
        "available_for_inference": llm_available(),
        "warm_status": ws["status"],     # cold | warming | ready | error | disabled
        "warm_error": ws["error"],
    }


@app.get("/api/compare", response_model=CompareResponse)
def api_compare() -> JSONResponse:
    metrics = load_metrics()
    if not metrics:
        raise HTTPException(status_code=503, detail="metrics.json이 없습니다. 학습을 먼저 실행하세요.")
    return JSONResponse(metrics)
