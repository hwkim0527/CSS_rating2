"""End-to-end smoke tests — exercise scoring API + compare endpoint."""
from __future__ import annotations

from fastapi.testclient import TestClient

from src.web.app import app
from src.web.scoring import (
    credit_score_from_prob,
    risk_grade_from_prob,
)

client = TestClient(app)

SAMPLE_GOOD = {
    "loan_amnt": 8000, "installment": 250, "int_rate": 8.5, "term": "36 months",
    "purpose": "credit_card", "annual_inc": 120000, "emp_length": "10",
    "home_ownership": "MORTGAGE", "verification_status": "Verified", "addr_state": "CA",
    "dti": 12.0, "delinq_2yrs": 0, "inq_last_6mths": 0, "open_acc": 10, "pub_rec": 0,
    "revol_bal": 3000, "revol_util": 12, "total_acc": 25, "mort_acc": 2,
    "pub_rec_bankruptcies": 0, "credit_history_years": 18.0,
    "application_type": "Individual", "initial_list_status": "w",
}

SAMPLE_BAD = {
    **SAMPLE_GOOD,
    "annual_inc": 28000, "dti": 45.0, "delinq_2yrs": 4, "inq_last_6mths": 6,
    "pub_rec": 2, "revol_util": 95, "pub_rec_bankruptcies": 1, "int_rate": 28.0,
    "credit_history_years": 3.0,
}


def test_healthz() -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_index_renders() -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "신용평가" in r.text


def test_score_endpoint_good_applicant() -> None:
    r = client.post("/api/score", json=SAMPLE_GOOD)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 <= body["default_probability"] <= 1
    assert 300 <= body["credit_score"] <= 850
    assert body["risk_grade"] in {"A", "B", "C", "D", "E"}


def test_score_endpoint_bad_applicant_has_higher_prob() -> None:
    good = client.post("/api/score", json=SAMPLE_GOOD).json()
    bad = client.post("/api/score", json=SAMPLE_BAD).json()
    assert bad["default_probability"] > good["default_probability"]


def test_score_endpoint_rejects_missing_fields() -> None:
    r = client.post("/api/score", json={"loan_amnt": 1000})
    assert r.status_code == 422


def test_compare_endpoint() -> None:
    r = client.get("/api/compare")
    assert r.status_code == 200
    body = r.json()
    assert "models" in body
    assert "logistic_regression" in body["models"]
    assert "xgboost" in body["models"]
    assert body["models"]["xgboost"]["auc"] > 0.6


def test_compare_page_renders() -> None:
    r = client.get("/compare")
    assert r.status_code == 200
    assert "모델 비교" in r.text or "AUC" in r.text


def test_credit_score_mapping_monotonic() -> None:
    s_low = credit_score_from_prob(0.05)
    s_mid = credit_score_from_prob(0.30)
    s_high = credit_score_from_prob(0.80)
    assert s_low > s_mid > s_high


def test_risk_grade_thresholds() -> None:
    assert risk_grade_from_prob(0.05)[0] == "A"
    assert risk_grade_from_prob(0.60)[0] == "E"
