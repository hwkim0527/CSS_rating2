"""GPU 없이 가능한 LLM 추론 경로 정적/반(半)정적 검증.

실제 14B forward pass 는 GPU 호스트(Colab/GCP)에서만 가능하지만, 추론을 깨뜨리는
대표적 silent-failure 들은 GPU 없이 미리 잡을 수 있다. 이 스크립트가 검증하는 것:

  1) 토크나이저 단일 토큰 분류 — '부실'/'정상' 의 첫 토큰 ID 가 서로 다른가.
     (같으면 logit 비교가 무의미 → llm_scoring/llm_eval 의 assert 가 터진다.)
     학습·추론 모두 base(Qwen/Qwen3-14B) 토크나이저를 쓰므로 그 토크나이저로 확인.
  2) 프롬프트 포맷 일관성 — 학습(format_sample)과 추론(score_with_llm)/평가(llm_eval)
     의 system 프롬프트가 동일해야 학습된 가중치가 의도대로 동작한다.
  3) 어댑터 config 호환성 — Drive 의 어댑터는 peft 0.19.1 로 저장됐다. 배포/로컬의
     peft 가 그 config(신규 필드 포함)를 파싱할 수 있는지 실제 LoraConfig.from_pretrained
     로 확인한다(배포 컨테이너에서 터지는 것을 미리 방지).

사용:  PYTHONUTF8=1 python -m scripts.verify_llm_local
종료코드: 0=모든 검증 통과, 1=하나 이상 실패.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
from pathlib import Path

# Windows 콘솔(cp949)에서 한글 출력 시 UnicodeEncodeError 방지.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

BASE_MODEL = os.environ.get("CSS_LLM_BASE", "Qwen/Qwen3-14B")

# Drive 의 학습 어댑터(Qwen3_fintech)에서 그대로 가져온 adapter_config.json.
# peft 0.19.1 로 저장되어 신규 필드(qalora_group_size, target_parameters,
# alora_invocation_tokens, lora_ga_config, arrow_config, use_qalora,
# trainable_token_indices, ensure_weight_tying, use_bdlora, peft_version)를 포함한다.
ADAPTER_CONFIG_B64 = (
    "ewogICJhbG9yYV9pbnZvY2F0aW9uX3Rva2VucyI6IG51bGwsCiAgImFscGhhX3BhdHRlcm4iOiB7fSwK"
    "ICAiYXJyb3dfY29uZmlnIjogbnVsbCwKICAiYXV0b19tYXBwaW5nIjogbnVsbCwKICAiYmFzZV9tb2Rl"
    "bF9uYW1lX29yX3BhdGgiOiAiUXdlbi9Rd2VuMy0xNEIiLAogICJiaWFzIjogIm5vbmUiLAogICJjb3Jk"
    "YV9jb25maWciOiBudWxsLAogICJlbnN1cmVfd2VpZ2h0X3R5aW5nIjogZmFsc2UsCiAgImV2YV9jb25m"
    "aWciOiBudWxsLAogICJleGNsdWRlX21vZHVsZXMiOiBudWxsLAogICJmYW5faW5fZmFuX291dCI6IGZh"
    "bHNlLAogICJpbmZlcmVuY2VfbW9kZSI6IHRydWUsCiAgImluaXRfbG9yYV93ZWlnaHRzIjogdHJ1ZSwK"
    "ICAibGF5ZXJfcmVwbGljYXRpb24iOiBudWxsLAogICJsYXllcnNfcGF0dGVybiI6IG51bGwsCiAgImxh"
    "eWVyc190b190cmFuc2Zvcm0iOiBudWxsLAogICJsb2Z0cV9jb25maWciOiB7fSwKICAibG9yYV9hbHBo"
    "YSI6IDE2LAogICJsb3JhX2JpYXMiOiBmYWxzZSwKICAibG9yYV9kcm9wb3V0IjogMC4wNSwKICAibG9y"
    "YV9nYV9jb25maWciOiBudWxsLAogICJtZWdhdHJvbl9jb25maWciOiBudWxsLAogICJtZWdhdHJvbl9j"
    "b3JlIjogIm1lZ2F0cm9uLmNvcmUiLAogICJtb2R1bGVzX3RvX3NhdmUiOiBudWxsLAogICJwZWZ0X3R5"
    "cGUiOiAiTE9SQSIsCiAgInBlZnRfdmVyc2lvbiI6ICIwLjE5LjEiLAogICJxYWxvcmFfZ3JvdXBfc2l6"
    "ZSI6IDE2LAogICJyIjogOCwKICAicmFua19wYXR0ZXJuIjoge30sCiAgInJldmlzaW9uIjogbnVsbCwK"
    "ICAidGFyZ2V0X21vZHVsZXMiOiBbCiAgICAidl9wcm9qIiwKICAgICJnYXRlX3Byb2oiLAogICAgImtf"
    "cHJvaiIsCiAgICAib19wcm9qIiwKICAgICJ1cF9wcm9qIiwKICAgICJkb3duX3Byb2oiLAogICAgInFf"
    "cHJvaiIKICBdLAogICJ0YXJnZXRfcGFyYW1ldGVycyI6IG51bGwsCiAgInRhc2tfdHlwZSI6ICJDQVVT"
    "QUxfTE0iLAogICJ0cmFpbmFibGVfdG9rZW5faW5kaWNlcyI6IG51bGwsCiAgInVzZV9iZGxvcmEiOiBu"
    "dWxsLAogICJ1c2VfZG9yYSI6IGZhbHNlLAogICJ1c2VfcWFsb3JhIjogZmFsc2UsCiAgInVzZV9yc2xv"
    "cmEiOiBmYWxzZQp9"
)

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, PASS if ok else FAIL, detail))
    print(f"[{PASS if ok else FAIL}] {name}" + (f" — {detail}" if detail else ""))


def check_tokenizer() -> None:
    """'부실'/'정상' 첫 토큰이 서로 다른지 (단일 토큰 분류 전제)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    pos = tok.encode("부실", add_special_tokens=False)
    neg = tok.encode("정상", add_special_tokens=False)
    detail = f"부실 first={pos[0]} (ids={pos}), 정상 first={neg[0]} (ids={neg})"
    record("tokenizer.단일토큰_구분(부실≠정상 first id)", pos[0] != neg[0], detail)


def check_prompt_consistency() -> None:
    """학습(format_sample)과 추론(build_chat_text)이 같은 헤드를 공유하는지 — 함수 레벨.

    추론 프롬프트는 학습 프롬프트에서 정답 토큰만 뺀 것이어야 한다(strict prefix).
    """
    from src.training.llm_finetune import format_sample
    from src.training.serialize import build_chat_text

    instr, ans = "테스트 신청자 정보", "부실"
    train_text = format_sample({"instruction": instr, "output": ans})
    expected_train = build_chat_text(instr, ans)   # 학습 형태
    infer_text = build_chat_text(instr)            # 추론 형태(평가도 동일 함수 사용)

    ok = (
        train_text == expected_train
        and train_text.startswith(infer_text)
        and train_text[len(infer_text):] == f"{ans}<|im_end|>"
    )
    detail = (
        "format_sample==build_chat_text(.,ans) 이고 추론은 학습의 strict prefix"
        if ok else f"train={train_text!r}\n      infer={infer_text!r}"
    )
    record("프롬프트.학습_추론_헤드_일치(build_chat_text 공유)", ok, detail)


def check_adapter_config_loads() -> None:
    """peft 0.19.1 로 저장된 adapter_config.json 을 현재 peft 가 파싱하는가."""
    import peft
    from peft import PeftConfig

    cfg_json = base64.b64decode(ADAPTER_CONFIG_B64).decode("utf-8")
    parsed = json.loads(cfg_json)
    saved_ver = parsed.get("peft_version", "?")
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "adapter_config.json").write_text(cfg_json, encoding="utf-8")
        try:
            cfg = PeftConfig.from_pretrained(d)
            ok = (
                getattr(cfg, "base_model_name_or_path", None) == "Qwen/Qwen3-14B"
                and getattr(cfg, "r", None) == 8
                and getattr(cfg, "lora_alpha", None) == 16
            )
            detail = (
                f"installed peft={peft.__version__} 가 saved peft={saved_ver} config 파싱 OK "
                f"(base={cfg.base_model_name_or_path}, r={cfg.r}, alpha={cfg.lora_alpha})"
            )
            record("어댑터config.로드호환(peft)", ok, detail)
        except Exception as e:  # noqa: BLE001
            record(
                "어댑터config.로드호환(peft)",
                False,
                f"installed peft={peft.__version__} 가 saved peft={saved_ver} config 파싱 실패: "
                f"{type(e).__name__}: {e}  → 배포 requirements 의 peft 하한을 {saved_ver} 로 올리세요.",
            )


def check_prompt_builds() -> None:
    """row_to_prompt + score_with_llm 프롬프트 조립이 예외 없이 동작하는가(문자열 레벨)."""
    from src.training.serialize import row_to_prompt

    sample = {
        "loan_amnt": 12000, "installment": 406, "int_rate": 13.5, "term": "36 months",
        "purpose": "debt_consolidation", "annual_inc": 65000, "emp_length": "5",
        "home_ownership": "RENT", "verification_status": "Not Verified", "addr_state": "CA",
        "dti": 18.5, "delinq_2yrs": 0, "inq_last_6mths": 0, "open_acc": 9, "pub_rec": 0,
        "revol_bal": 9500, "revol_util": 42, "total_acc": 22, "mort_acc": 1,
        "pub_rec_bankruptcies": 0, "credit_history_years": 12.5,
        "application_type": "Individual", "initial_list_status": "w",
    }
    p = row_to_prompt(sample)
    ok = "[신청자 정보]" in p and "[판정]" in p and "$12,000" in p
    record("프롬프트.row_to_prompt_조립", ok, f"len={len(p)}")


def main() -> int:
    print(f"=== LLM 로컬 검증 (base={BASE_MODEL}) ===")
    for fn in (check_prompt_builds, check_prompt_consistency, check_adapter_config_loads, check_tokenizer):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            record(fn.__name__, False, f"예외: {type(e).__name__}: {e}")

    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    print(f"\n=== 요약: {len(results)-n_fail}/{len(results)} 통과, 실패 {n_fail} ===")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
