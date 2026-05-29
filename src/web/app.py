"""FastAPI 신용평가 웹 서비스."""
from __future__ import annotations

import logging
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

app = FastAPI(
    title="CSS Rating 2 — AI 신용평가시스템",
    description="개인 신용정보를 입력하면 부실 확률을 산출합니다.",
    version="1.0.0",
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


@app.get("/api/compare", response_model=CompareResponse)
def api_compare() -> JSONResponse:
    metrics = load_metrics()
    if not metrics:
        raise HTTPException(status_code=503, detail="metrics.json이 없습니다. 학습을 먼저 실행하세요.")
    return JSONResponse(metrics)
