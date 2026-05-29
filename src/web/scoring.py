"""모델 로딩 및 점수 산출."""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.utils.config import (
    ARTIFACTS_DIR,
    CATEGORICAL_FEATURES,
    METRICS_PATH,
    NUMERIC_FEATURES,
)

log = logging.getLogger("scoring")

MODEL_PATH = ARTIFACTS_DIR / "xgboost.joblib"
BASELINE_PATH = ARTIFACTS_DIR / "baseline_logistic.joblib"


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"학습된 모델이 없습니다: {MODEL_PATH}. python -m src.models.train 을 먼저 실행하세요."
        )
    log.info("Loading model: %s", MODEL_PATH)
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_baseline():
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(BASELINE_PATH)
    return joblib.load(BASELINE_PATH)


@lru_cache(maxsize=1)
def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def to_dataframe(payload: dict) -> pd.DataFrame:
    """Pydantic 입력 → 학습 시 사용한 컬럼 순서의 DataFrame."""
    row = {}
    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        row[col] = payload.get(col)
    df = pd.DataFrame([row])
    return df


def credit_score_from_prob(p: float) -> int:
    """부실확률 → FICO 풍 신용점수 (300~850)."""
    p = float(np.clip(p, 1e-4, 1 - 1e-4))
    # log-odds 매핑: 낮은 확률 → 높은 점수
    logit = np.log((1 - p) / p)
    score = 550 + 60 * logit  # rough scale; clamp below
    return int(np.clip(round(score), 300, 850))


def risk_grade_from_prob(p: float) -> tuple[str, str]:
    # Thresholds calibrated to a ~21.6% base default rate (Lending Club test set).
    # Below base rate = better than average; above = worse.
    if p < 0.08:
        return "A", "최우량 (부실위험 매우 낮음)"
    if p < 0.15:
        return "B", "우량 (부실위험 낮음)"
    if p < 0.25:
        return "C", "보통 (평균 수준)"
    if p < 0.40:
        return "D", "주의 (부실위험 높음)"
    return "E", "고위험 (부실위험 매우 높음)"


def explain_top_factors(payload: dict, prob: float) -> list[dict]:
    """규칙 기반 위험 요인 간단 설명 (학습된 SHAP 없이도 동작)."""
    factors: list[dict] = []
    dti = payload.get("dti") or 0
    if dti >= 30:
        factors.append({
            "factor": "DTI",
            "value": dti,
            "impact": "위험",
            "note": f"부채/소득 비율이 {dti:.1f}%로 높습니다 (권장 ≤ 25%).",
        })
    elif dti <= 15:
        factors.append({
            "factor": "DTI",
            "value": dti,
            "impact": "긍정",
            "note": f"DTI {dti:.1f}%로 양호합니다.",
        })

    util = payload.get("revol_util") or 0
    if util >= 70:
        factors.append({
            "factor": "회전신용 활용률",
            "value": util,
            "impact": "위험",
            "note": f"활용률 {util:.0f}%로 높습니다 (권장 ≤ 30%).",
        })

    inq = payload.get("inq_last_6mths") or 0
    if inq >= 3:
        factors.append({
            "factor": "최근 6개월 신용조회",
            "value": inq,
            "impact": "위험",
            "note": f"{inq}건의 신용조회는 신용도에 부정적입니다.",
        })

    delinq = payload.get("delinq_2yrs") or 0
    if delinq >= 1:
        factors.append({
            "factor": "최근 2년 연체",
            "value": delinq,
            "impact": "위험",
            "note": f"{delinq}건의 연체 기록이 있습니다.",
        })

    bk = payload.get("pub_rec_bankruptcies") or 0
    if bk >= 1:
        factors.append({
            "factor": "파산 기록",
            "value": bk,
            "impact": "위험",
            "note": f"{bk}건의 파산 기록이 점수에 큰 영향을 줍니다.",
        })

    inc = payload.get("annual_inc") or 0
    if inc >= 100_000:
        factors.append({
            "factor": "연소득",
            "value": inc,
            "impact": "긍정",
            "note": f"연소득 ${inc:,.0f}은 상환 능력에 긍정적입니다.",
        })

    history = payload.get("credit_history_years") or 0
    if history >= 15:
        factors.append({
            "factor": "신용 이력",
            "value": history,
            "impact": "긍정",
            "note": f"{history:.1f}년의 신용 이력은 양호한 지표입니다.",
        })

    if not factors:
        factors.append({
            "factor": "종합",
            "value": prob,
            "impact": "중립",
            "note": "특이한 위험/긍정 요인 없이 평균 수준입니다.",
        })
    return factors


def score_one(payload: dict, model_key: str = "xgboost") -> dict:
    model = load_baseline() if model_key == "logistic" else load_model()
    df = to_dataframe(payload)
    prob = float(model.predict_proba(df)[0, 1])
    grade, grade_kr = risk_grade_from_prob(prob)
    return {
        "default_probability": prob,
        "credit_score": credit_score_from_prob(prob),
        "risk_grade": grade,
        "risk_grade_kr": grade_kr,
        "model_name": "XGBoost" if model_key == "xgboost" else "Logistic Regression",
        "top_factors": explain_top_factors(payload, prob),
    }
