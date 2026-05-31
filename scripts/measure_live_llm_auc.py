"""워밍된 라이브 /api/score_llm 에 test.parquet 표본을 순차 POST 해 실측 AUC/KS 계산.

advisor 권고: 2개 표본으로 단정하지 말고 stratified 표본으로 배포 모델이 실제로
위험을 구분하는지 확정한다. 동시 요청은 콜드 인스턴스에서 race 를 유발하므로 순차.

사용:
    python -m scripts.measure_live_llm_auc --url https://.../api/score_llm --n 50
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO = Path(__file__).resolve().parents[1]

INT_FIELDS = [
    "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
    "total_acc", "mort_acc", "pub_rec_bankruptcies",
]
FIELDS = [
    "loan_amnt", "installment", "term", "purpose", "annual_inc", "emp_length",
    "home_ownership", "verification_status", "addr_state", "dti", "delinq_2yrs",
    "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "mort_acc", "pub_rec_bankruptcies", "credit_history_years", "application_type",
    "initial_list_status",
]


def row_to_payload(row: dict) -> dict:
    import pandas as pd
    p = {}
    for k in FIELDS:
        v = row[k]
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue  # NaN/결측 → 필드 생략 → Pydantic 기본값 적용(웹앱과 동일)
        if k in INT_FIELDS:
            p[k] = int(round(float(v)))
        elif k in {"loan_amnt", "installment", "annual_inc", "dti", "revol_bal",
                   "revol_util", "credit_history_years"}:
            p[k] = float(v)
        else:
            p[k] = str(v)
    return p


def post(url: str, payload: dict, timeout: float):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def auc_ks(y_true, y_score):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    auc = float(roc_auc_score(y_true, y_score))
    order = np.argsort(-y_score)
    ys = y_true[order]
    pos_cum = np.cumsum(ys) / max(ys.sum(), 1)
    neg_cum = np.cumsum(1 - ys) / max((1 - ys).sum(), 1)
    ks = float(np.max(np.abs(pos_cum - neg_cum)))
    return auc, ks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="/api/score_llm 의 전체 URL")
    ap.add_argument("--n", type=int, default=50, help="클래스당 표본 수 (총 2N)")
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--out", default=str(REPO / "claudedocs" / "live_llm_eval.json"))
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_parquet(REPO / "data" / "processed" / "test.parquet")
    pos = df[df["default"] == 1].sample(args.n, random_state=42)
    neg = df[df["default"] == 0].sample(args.n, random_state=42)
    sample = pd.concat([pos, neg]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"표본: {len(sample)}건 (부실 {args.n} / 정상 {args.n})  →  {args.url}")

    rows = []
    t0 = time.time()
    for i, (_, r) in enumerate(sample.iterrows()):
        payload = row_to_payload(r.to_dict())
        try:
            resp = post(args.url, payload, args.timeout)
            prob = float(resp["default_probability"])
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}] 실패: {type(e).__name__}: {e}")
            continue
        rows.append({"label": int(r["default"]), "prob": prob})
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(sample)} 완료 (마지막 prob={prob:.4f}, {time.time()-t0:.0f}s)")

    if len(rows) < 10:
        print("유효 응답이 너무 적어 AUC 계산 불가.")
        return 1

    y_true = [x["label"] for x in rows]
    y_score = [x["prob"] for x in rows]
    auc, ks = auc_ks(y_true, y_score)
    probs_sorted = sorted(y_score)
    summary = {
        "evaluated": len(rows),
        "auc": auc,
        "ks": ks,
        "prob_min": min(y_score),
        "prob_max": max(y_score),
        "prob_mean": sum(y_score) / len(y_score),
        "prob_spread": max(y_score) - min(y_score),
        "note": "배포(구 프롬프트 빌드) Qwen3-14B QLoRA 라이브 실측",
    }
    print("\n=== 라이브 LLM 실측 ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"확률 분포 예시(정렬): {[round(p,3) for p in probs_sorted[:5]]} ... {[round(p,3) for p in probs_sorted[-5:]]}")
    Path(args.out).write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"저장 → {args.out}")
    print(f"\n해석 기준: AUC≈0.5 또는 prob_spread 작음 → 위험 구분 못 함(미학습). "
          f"AUC>0.65 → 의미 있는 구분.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
