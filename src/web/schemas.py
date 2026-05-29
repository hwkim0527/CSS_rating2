"""Pydantic input schema for /api/score."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    """개인이 입력하는 신용평가 신청 정보."""

    # === Loan request ===
    loan_amnt: float = Field(..., ge=500, le=100_000, description="대출 신청 금액 (USD)")
    installment: float = Field(..., ge=0, description="월 상환금 (USD)")
    int_rate: float = Field(..., ge=0, le=40, description="이자율 (%)")
    term: Literal["36 months", "60 months"] = Field(..., description="상환 기간")
    purpose: str = Field(..., description="대출 목적")

    # === Borrower ===
    annual_inc: float = Field(..., ge=0, description="연소득 (USD)")
    emp_length: str = Field(default="10", description="근속 연수 (0~10, 10=10년 이상)")
    home_ownership: str = Field(..., description="주거 형태: RENT/OWN/MORTGAGE/OTHER")
    verification_status: str = Field(
        default="Not Verified", description="소득 검증 상태"
    )
    addr_state: str = Field(..., min_length=2, max_length=2, description="거주 주 코드")

    # === Credit history ===
    dti: float = Field(..., ge=0, le=100, description="부채 대비 소득 비율 (DTI %)")
    delinq_2yrs: int = Field(default=0, ge=0, description="최근 2년 연체 건수")
    inq_last_6mths: int = Field(default=0, ge=0, description="최근 6개월 신용조회 건수")
    open_acc: int = Field(default=5, ge=0, description="현재 보유 신용계좌 수")
    pub_rec: int = Field(default=0, ge=0, description="공적 기록 건수")
    revol_bal: float = Field(default=0, ge=0, description="회전신용 잔액 (USD)")
    revol_util: float = Field(default=30.0, ge=0, le=200, description="회전신용 활용률 (%)")
    total_acc: int = Field(default=10, ge=0, description="누적 신용계좌 수")
    mort_acc: int = Field(default=0, ge=0, description="모기지 계좌 수")
    pub_rec_bankruptcies: int = Field(default=0, ge=0, description="파산 기록 건수")
    credit_history_years: float = Field(..., ge=0, le=80, description="신용 이력 (년)")

    application_type: Literal["Individual", "Joint App"] = Field(
        default="Individual", description="신청 유형"
    )
    initial_list_status: Literal["w", "f"] = Field(default="w", description="초기 분류 상태")


class ScoreResponse(BaseModel):
    default_probability: float = Field(..., description="부실 확률 (0~1)")
    credit_score: int = Field(..., description="신용점수 (300~850 환산)")
    risk_grade: str = Field(..., description="A(최우량) ~ E(고위험)")
    risk_grade_kr: str = Field(..., description="등급 한국어 설명")
    model_name: str = Field(..., description="사용된 모델명")
    top_factors: list[dict] = Field(default_factory=list, description="주요 위험 요인")


class CompareResponse(BaseModel):
    dataset: dict
    models: dict
    comparison_vs_baseline: dict
