"""LLM 채점 경로의 GPU-free 검증.

- 프롬프트 일관성(build_chat_text): torch 불필요 → 어디서나 실행.
- score_with_llm 수학 경로(softmax/인덱싱): torch 필요 → 없으면 skip.
  실제 14B 모델은 mock 하고 logits 텐서만 직접 주입해 부실확률 계산을 검증한다.
"""
from __future__ import annotations

import pytest

from src.training.serialize import SYSTEM_PROMPT, build_chat_text, row_to_prompt

SAMPLE = {
    "loan_amnt": 12000, "installment": 406, "term": "36 months",
    "purpose": "debt_consolidation", "annual_inc": 65000, "emp_length": "5",
    "home_ownership": "RENT", "verification_status": "Not Verified", "addr_state": "CA",
    "dti": 18.5, "delinq_2yrs": 0, "inq_last_6mths": 0, "open_acc": 9, "pub_rec": 0,
    "revol_bal": 9500, "revol_util": 42, "total_acc": 22, "mort_acc": 1,
    "pub_rec_bankruptcies": 0, "credit_history_years": 12.5,
    "application_type": "Individual", "initial_list_status": "w",
}


# ── 프롬프트 일관성 (torch 불필요) ──────────────────────────────────────────

def test_inference_prompt_is_strict_prefix_of_training_prompt() -> None:
    """추론 프롬프트 = 학습 프롬프트에서 정답 토큰만 제거한 것이어야 한다."""
    instr = row_to_prompt(SAMPLE)
    train_pos = build_chat_text(instr, "부실")
    train_neg = build_chat_text(instr, "정상")
    infer = build_chat_text(instr)
    assert train_pos.startswith(infer)
    assert train_neg.startswith(infer)
    assert train_pos[len(infer):] == "부실<|im_end|>"
    assert train_neg[len(infer):] == "정상<|im_end|>"


def test_training_format_matches_shared_builder() -> None:
    """llm_finetune.format_sample 이 build_chat_text 와 글자까지 동일한지."""
    from src.training.llm_finetune import format_sample

    ex = {"instruction": row_to_prompt(SAMPLE), "output": "부실"}
    assert format_sample(ex) == build_chat_text(ex["instruction"], ex["output"])


def test_system_prompt_is_single_source() -> None:
    """system 프롬프트가 추론 헤드에 그대로 포함돼야 한다(단일 진실 원천)."""
    head = build_chat_text("X")
    assert SYSTEM_PROMPT in head
    assert head.endswith("<|im_start|>assistant\n")


# ── score_with_llm 수학 경로 (torch 필요) ───────────────────────────────────

class _FakeEnc(dict):
    def to(self, _device):  # tok(...).to(model.device)
        return self


class _FakeTok:
    def __call__(self, _text, return_tensors=None):  # noqa: ARG002
        import torch
        return _FakeEnc(input_ids=torch.tensor([[1, 2, 3]]))


class _FakeModel:
    """model(**inputs).logits 만 제공하는 최소 mock."""
    device = "cpu"

    def __init__(self, logits):
        self._logits = logits

    def __call__(self, **_kwargs):
        out = type("Out", (), {})()
        out.logits = self._logits
        return out


def _patch_load(monkeypatch, pos_logit: float, neg_logit: float, pos_id=5, neg_id=3, vocab=8):
    import torch
    from src.web import llm_scoring

    logits = torch.full((1, 4, vocab), -20.0)
    logits[0, -1, pos_id] = pos_logit
    logits[0, -1, neg_id] = neg_logit
    monkeypatch.setattr(
        llm_scoring, "_load_llm",
        lambda: (_FakeModel(logits), _FakeTok(), pos_id, neg_id),
    )


def test_score_with_llm_returns_high_prob_when_pos_dominates(monkeypatch) -> None:
    pytest.importorskip("torch")
    from src.web.llm_scoring import score_with_llm

    _patch_load(monkeypatch, pos_logit=6.0, neg_logit=0.0)
    p = score_with_llm(SAMPLE)
    assert 0.0 <= p <= 1.0
    assert p > 0.9, f"부실 logit 우세인데 P(부실)={p}"


def test_score_with_llm_returns_low_prob_when_neg_dominates(monkeypatch) -> None:
    pytest.importorskip("torch")
    from src.web.llm_scoring import score_with_llm

    _patch_load(monkeypatch, pos_logit=0.0, neg_logit=6.0)
    p = score_with_llm(SAMPLE)
    assert 0.0 <= p <= 1.0
    assert p < 0.1, f"정상 logit 우세인데 P(부실)={p}"


def test_score_with_llm_symmetric_at_equal_logits(monkeypatch) -> None:
    pytest.importorskip("torch")
    from src.web.llm_scoring import score_with_llm

    _patch_load(monkeypatch, pos_logit=2.0, neg_logit=2.0)
    p = score_with_llm(SAMPLE)
    assert abs(p - 0.5) < 1e-5, f"동일 logit 인데 P(부실)={p}"


# ── 엔드포인트/UI 통합 (torch 불필요, CSS_ENABLE_LLM 미설정 기준) ────────────

def _client():
    from fastapi.testclient import TestClient
    from src.web.app import app
    return TestClient(app)


def test_llm_status_endpoint_shape() -> None:
    r = _client().get("/api/llm_status")
    assert r.status_code == 200
    body = r.json()
    for key in ("enabled_flag", "adapter_path", "adapter_present", "available_for_inference"):
        assert key in body, f"누락된 키: {key}"
    # 기본(LLM 비활성) 환경에서는 추론 불가여야 한다.
    assert body["available_for_inference"] is False


def test_score_llm_returns_503_when_unavailable() -> None:
    """LLM 미가용 시 깨끗한 503(서비스 불가) — 500 이 아니어야 graceful degradation."""
    r = _client().post("/api/score_llm", json=SAMPLE)
    assert r.status_code == 503, r.text
    assert "LLM" in r.json()["detail"] or "어댑터" in r.json()["detail"]


def test_index_has_model_selector() -> None:
    html = _client().get("/").text
    assert "평가 모델 선택" in html
    assert "Qwen3-14B sLLM" in html
    assert 'value="llm"' in html
