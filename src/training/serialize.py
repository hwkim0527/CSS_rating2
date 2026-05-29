"""Tabular row → natural-language prompt for LLM training/inference."""
from __future__ import annotations

import pandas as pd

PURPOSE_KR = {
    "debt_consolidation": "부채 통합",
    "credit_card": "신용카드 대환",
    "home_improvement": "주택 개량",
    "major_purchase": "대형 구매",
    "small_business": "사업 자금",
    "car": "자동차",
    "medical": "의료",
    "vacation": "여행",
    "moving": "이사",
    "house": "주택 구매",
    "wedding": "결혼",
    "renewable_energy": "재생에너지",
    "educational": "교육",
    "other": "기타",
}

HOME_KR = {"RENT": "임대", "MORTGAGE": "모기지", "OWN": "자가", "OTHER": "기타", "NONE": "없음", "ANY": "기타"}
VERIF_KR = {"Not Verified": "미검증", "Verified": "검증완료", "Source Verified": "원천 검증"}

INSTRUCTION = (
    "다음 신청자 정보를 보고 12개월 내 채무불이행(부실) 여부를 정상/부실 중 하나로 판정하세요."
)


def _safe_float(v, default: float = 0.0) -> float:
    try:
        f = float(v)
        if pd.isna(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _safe_int(v, default: int = 0) -> int:
    return int(_safe_float(v, float(default)))


def row_to_prompt(row: dict) -> str:
    purpose = PURPOSE_KR.get(str(row.get("purpose", "other")), str(row.get("purpose", "기타")))
    home = HOME_KR.get(str(row.get("home_ownership", "OTHER")), str(row.get("home_ownership", "기타")))
    verif = VERIF_KR.get(str(row.get("verification_status", "Not Verified")), "미검증")

    fields = [
        f"- 대출 신청 금액: ${_safe_float(row.get('loan_amnt')):,.0f}",
        f"- 상환 기간: {row.get('term', '')}",
        f"- 월 상환금: ${_safe_float(row.get('installment')):,.0f}",
        f"- 대출 목적: {purpose}",
        f"- 연소득: ${_safe_float(row.get('annual_inc')):,.0f}",
        f"- 근속: {row.get('emp_length', 'NA')}년",
        f"- 주거: {home}",
        f"- 소득 검증: {verif}",
        f"- 거주 주: {row.get('addr_state', '')}",
        f"- DTI: {_safe_float(row.get('dti')):.1f}%",
        f"- 최근 2년 연체: {_safe_int(row.get('delinq_2yrs'))}건",
        f"- 최근 6개월 신용조회: {_safe_int(row.get('inq_last_6mths'))}건",
        f"- 보유 신용계좌: {_safe_int(row.get('open_acc'))}개",
        f"- 회전신용 활용률: {_safe_float(row.get('revol_util')):.1f}%",
        f"- 회전신용 잔액: ${_safe_float(row.get('revol_bal')):,.0f}",
        f"- 누적 신용계좌: {_safe_int(row.get('total_acc'))}개",
        f"- 모기지 계좌: {_safe_int(row.get('mort_acc'))}개",
        f"- 파산 기록: {_safe_int(row.get('pub_rec_bankruptcies'))}건",
        f"- 신용 이력: {_safe_float(row.get('credit_history_years')):.1f}년",
    ]
    body = "\n".join(fields)
    return f"{INSTRUCTION}\n\n[신청자 정보]\n{body}\n\n[판정]"


def label_to_word(y: int) -> str:
    return "부실" if int(y) == 1 else "정상"


def dataframe_to_jsonl(df: pd.DataFrame, target: str = "default") -> list[dict]:
    """HuggingFace SFT 포맷 (instruction/output 쌍)."""
    out = []
    for _, row in df.iterrows():
        rec = row.to_dict()
        out.append({
            "instruction": row_to_prompt(rec),
            "output": label_to_word(rec[target]),
        })
    return out
