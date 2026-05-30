"""선택적 LLM 채점 경로 (Qwen3-14B QLoRA). 무거운 의존성은 어댑터가 실제로
디스크에 있을 때만 지연 임포트한다.

배포 시 학습된 어댑터를 가져오는 방법 (둘 중 하나):

  (A) Google Drive 에서 자동 다운로드 — 학습된 모델이 hwkim0527 의 Drive
      "Colab Notebooks/Qwen3_fintech" 에 있을 때:
          export CSS_ENABLE_LLM=1
          # gdown(간편) 방식:
          export CSS_LLM_DRIVE_FOLDER_ID=<Qwen3_fintech 폴더 ID>
          # 또는 서비스 계정(비공개·권장) 방식:
          export CSS_LLM_GDRIVE_SA_JSON=/path/sa.json
          export CSS_LLM_DRIVE_FOLDER_ID=<폴더 ID>
      → 첫 추론 시 artifacts/qwen3_lora 로 내려받아 캐시한다.

  (B) 수동 배치 — 어댑터 폴더를 직접 artifacts/qwen3_lora 에 복사 후
          export CSS_ENABLE_LLM=1
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

from src.training.serialize import row_to_prompt
from src.utils.config import ARTIFACTS_DIR

log = logging.getLogger("llm_scoring")

# 어댑터 위치 — 환경변수로 덮어쓸 수 있다 (예: Drive 마운트 경로 직접 지정).
ADAPTER_DIR = Path(os.environ.get("CSS_LLM_ADAPTER_DIR", str(ARTIFACTS_DIR / "qwen3_lora")))
BASE_MODEL = os.environ.get("CSS_LLM_BASE", "Qwen/Qwen3-14B")
LLM_ENABLED = os.environ.get("CSS_ENABLE_LLM", "0") == "1"
# Drive 자동 다운로드 트리거 (설정되면 어댑터가 없을 때 내려받음).
DRIVE_FOLDER_ID = os.environ.get("CSS_LLM_DRIVE_FOLDER_ID", "")


def _adapter_present() -> bool:
    return ADAPTER_DIR.exists() and any(ADAPTER_DIR.iterdir())


def ensure_adapter() -> bool:
    """어댑터가 없고 Drive 폴더 ID 가 설정돼 있으면 내려받는다.

    Returns: 어댑터가 사용 가능하면 True.
    """
    if _adapter_present():
        return True
    if not DRIVE_FOLDER_ID:
        return False
    log.info("어댑터가 없어 Google Drive 에서 다운로드합니다: folder_id=%s", DRIVE_FOLDER_ID)
    from src.web.download_model import download_adapter_from_drive

    download_adapter_from_drive(ADAPTER_DIR)
    return _adapter_present()


def llm_available() -> bool:
    """저렴한 확인 — 플래그가 켜져 있고, 어댑터가 있거나 Drive 에서 받을 수 있을 때 True."""
    if not LLM_ENABLED:
        return False
    return _adapter_present() or bool(DRIVE_FOLDER_ID)


@lru_cache(maxsize=1)
def _load_llm():
    """Returns (model, tokenizer, pos_token_id, neg_token_id) or raises."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not ensure_adapter():
        raise FileNotFoundError(
            f"LLM 어댑터를 찾을 수 없습니다: {ADAPTER_DIR}. "
            "CSS_LLM_DRIVE_FOLDER_ID 를 설정해 Drive 에서 받거나, 어댑터를 직접 배치하세요."
        )

    log.info("Loading LLM: base=%s adapter=%s", BASE_MODEL, ADAPTER_DIR)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
    model.eval()
    # 학습 때와 동일하게 라벨을 첫 토큰으로 읽는다 (Qwen3 <think> 출력 억제됨).
    pos = tok.encode("부실", add_special_tokens=False)[0]
    neg = tok.encode("정상", add_special_tokens=False)[0]
    # 두 라벨의 첫 토큰이 같으면 logit 비교가 무의미해진다 (silent failure 방지).
    if pos == neg:
        raise RuntimeError(
            f"'부실'/'정상' 의 첫 토큰이 동일합니다 (id={pos}). "
            f"이 토크나이저({BASE_MODEL})에서는 단일 토큰 분류가 불가능합니다."
        )
    return model, tok, pos, neg


def score_with_llm(payload: dict) -> float:
    """Returns default probability in [0,1]."""
    import torch

    model, tok, pos_id, neg_id = _load_llm()
    prompt = (
        f"<|im_start|>system\n당신은 신용평가 전문가입니다. 신청자 정보를 보고 부실 또는 정상을 한 단어로 답하세요.<|im_end|>\n"
        f"<|im_start|>user\n{row_to_prompt(payload)}<|im_end|>\n<|im_start|>assistant\n"
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
    probs = torch.softmax(logits[[neg_id, pos_id]], dim=-1)
    return float(probs[1].item())
