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

베이스 모델(Qwen3-14B, ~28GB)도 Drive 에서 받기 (HF Hub 429 우회):
    export CSS_LLM_BASE_DRIVE_FOLDER_ID=<Qwen3_base 폴더 ID>
  설정 시 HF 대신 Drive 에서 받아 artifacts/qwen3_base 로 캐시하고 그 로컬
  경로로 로드한다. 미설정 시 BASE_MODEL(HF repo id)을 그대로 사용한다.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from functools import lru_cache
from pathlib import Path

from src.training.serialize import build_chat_text, row_to_prompt
from src.utils.config import ARTIFACTS_DIR

log = logging.getLogger("llm_scoring")

# 어댑터 위치 — 환경변수로 덮어쓸 수 있다 (예: Drive 마운트 경로 직접 지정).
ADAPTER_DIR = Path(os.environ.get("CSS_LLM_ADAPTER_DIR", str(ARTIFACTS_DIR / "qwen3_lora")))
BASE_MODEL = os.environ.get("CSS_LLM_BASE", "Qwen/Qwen3-14B")
LLM_ENABLED = os.environ.get("CSS_ENABLE_LLM", "0") == "1"
# Drive 자동 다운로드 트리거 (설정되면 어댑터가 없을 때 내려받음).
DRIVE_FOLDER_ID = os.environ.get("CSS_LLM_DRIVE_FOLDER_ID", "")
# 베이스 모델(Qwen3-14B)도 Drive 에서 받을 폴더 ID. 설정되면 HF Hub 대신 Drive
# 에서 받아 로컬 경로로 로드한다(GCP IP 대역의 HF 429 완전 우회).
BASE_DRIVE_FOLDER_ID = os.environ.get("CSS_LLM_BASE_DRIVE_FOLDER_ID", "")
BASE_DIR = Path(os.environ.get("CSS_LLM_BASE_DIR", str(ARTIFACTS_DIR / "qwen3_base")))


def _adapter_present() -> bool:
    # '디렉터리 비어있지 않음'은 부분 다운로드도 완료로 오인한다. 완료 센티넬이
    # 있거나(다운로드 완료), 수동 배치 시 핵심 두 파일(config+weights)이 모두 있을 때만 True.
    if (ADAPTER_DIR / ".download_complete").exists():
        return True
    return (ADAPTER_DIR / "adapter_config.json").exists() and (
        ADAPTER_DIR / "adapter_model.safetensors"
    ).exists()


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


def resolve_base_model() -> str:
    """베이스 모델 경로/이름을 돌려준다.

    CSS_LLM_BASE_DRIVE_FOLDER_ID 가 설정돼 있으면 Drive 에서 베이스 모델을 받아
    그 로컬 경로를 반환한다(HF 우회). 아니면 BASE_MODEL(HF repo id)을 그대로 반환.
    """
    if not BASE_DRIVE_FOLDER_ID:
        return BASE_MODEL
    # 완료 센티넬 기준(config.json 만 보면 샤드 누락 부분 다운로드를 캐시로 오인).
    if (BASE_DIR / ".download_complete").exists():
        log.info("베이스 모델 로컬 캐시 사용: %s", BASE_DIR)
        return str(BASE_DIR)
    log.info("베이스 모델을 Google Drive 에서 다운로드합니다: folder_id=%s", BASE_DRIVE_FOLDER_ID)
    from src.web.download_model import download_base_from_drive

    download_base_from_drive(BASE_DIR)
    return str(BASE_DIR)


def llm_available() -> bool:
    """저렴한 확인 — 플래그가 켜져 있고, 어댑터가 있거나 Drive 에서 받을 수 있을 때 True."""
    if not LLM_ENABLED:
        return False
    return _adapter_present() or bool(DRIVE_FOLDER_ID)


# ── 워밍(warm-up) 조율 ──────────────────────────────────────────────────────
# 베이스(Qwen3-14B ~29GB) + 어댑터 다운로드와 4bit GPU 로딩은 수 분 걸린다. 이걸
# 요청 핸들러 안에서 하면 (1) 클라이언트 연결이 끊기고 (2) 동시 요청이 각자 다운로드
# 하는 race 가 난다. 따라서 **단일 백그라운드 스레드**가 한 번만 받아서 로딩하고,
# 그 전까지 요청은 즉시 503("워밍 중")으로 돌려보낸다(다운로드를 절대 트리거 안 함).
# 단일 GPU forward 직렬화용 세마포어 — _warm_lock 과 별개. ready 이후 concurrency>1
# 에서 여러 요청이 같은 model 객체로 동시 forward 하면 활성/KV 메모리가 중첩돼 단일
# L4 에서 CUDA OOM 위험이 있다. forward 직전에만 잡아 직렬화한다(다운로드/상태조회는 막지 않음).
_infer_sem = threading.BoundedSemaphore(1)

_warm_lock = threading.Lock()
# cold | warming | ready | error.  error_ts: 마지막 실패 시각(monotonic) — 백오프용.
_warm_state = {"status": "cold", "error": None, "error_ts": 0.0}
# 결정적 실패(손상 캐시·미공유 폴더 등)에서 매 요청 재다운로드 폭주를 막는 재시도 간격(초).
_WARM_ERROR_BACKOFF_S = 60.0


def _sanitize_error(e: Exception) -> str:
    """클라이언트 노출용 오류 문자열 — 비공개 Drive folder_id/내부 경로를 마스킹한다."""
    msg = f"{type(e).__name__}: {e}"
    msg = re.sub(r"folder_id=[\w-]+", "folder_id=***", msg)
    msg = re.sub(r"/(?:app|content|home)/[^\s'\"]*", "<path>", msg)
    return msg[:300]


def warm_status() -> dict:
    """현재 워밍 상태의 스냅샷 {'status':..., 'error':...}."""
    with _warm_lock:
        return {"status": _warm_state["status"], "error": _warm_state["error"]}


def _gpu_cleanup() -> None:
    """워밍 실패 시 부분 적재된 GPU 메모리를 명시적으로 회수(반복 실패 누수 방지)."""
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _warm_worker() -> None:
    """단일 워밍 스레드 본체: 모델을 받아 GPU 에 로딩하고 lru_cache 를 채운다."""
    try:
        _load_llm()  # ensure_adapter + resolve_base(다운로드) + 4bit 로딩 + 토큰 검증
        with _warm_lock:
            _warm_state["status"] = "ready"
            _warm_state["error"] = None
        log.info("LLM 워밍 완료 (status=ready)")
    except Exception as e:  # noqa: BLE001
        with _warm_lock:
            _warm_state["status"] = "error"
            _warm_state["error"] = _sanitize_error(e)
            _warm_state["error_ts"] = time.monotonic()
        log.exception("LLM 워밍 실패")
        _gpu_cleanup()


def _start_warm_locked() -> None:
    """_warm_lock 을 잡은 상태에서 호출 — warming 으로 전이하고 단일 스레드를 띄운다."""
    _warm_state["status"] = "warming"
    _warm_state["error"] = None
    threading.Thread(target=_warm_worker, name="llm-warmup", daemon=True).start()


def ensure_warming_started() -> str:
    """워밍을 (필요하면) 백그라운드로 시작하고 즉시 현재 상태를 반환한다(블로킹 없음).

    - cold: 곧바로 워밍 시작.
    - warming/ready: 그대로 둔다(중복 다운로드 방지).
    - error: 마지막 실패로부터 _WARM_ERROR_BACKOFF_S 가 지났을 때만 재시도. 그 전에는
      error 를 유지해 호출자가 실제 오류를 보게 하고, 결정적 실패의 재다운로드 폭주를 막는다.
    """
    if not LLM_ENABLED:
        return "disabled"
    with _warm_lock:
        st = _warm_state["status"]
        if st == "cold":
            _start_warm_locked()
        elif st == "error":
            if time.monotonic() - _warm_state.get("error_ts", 0.0) >= _WARM_ERROR_BACKOFF_S:
                _start_warm_locked()
        return _warm_state["status"]


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

    base_ref = resolve_base_model()
    log.info("Loading LLM: base=%s adapter=%s", base_ref, ADAPTER_DIR)

    # 토크나이저를 먼저 로드하고 라벨 토큰을 검증한다. 결정적 실패(단일 토큰 분류
    # 불가)를 ~9GB GPU 적재 *이전*에 차단해 VRAM 낭비/누수를 막는다.
    tok = AutoTokenizer.from_pretrained(base_ref, trust_remote_code=True)
    # 학습 때와 동일하게 라벨을 첫 토큰으로 읽는다 (Qwen3 <think> 출력 억제됨).
    pos = tok.encode("부실", add_special_tokens=False)[0]
    neg = tok.encode("정상", add_special_tokens=False)[0]
    if pos == neg:  # 첫 토큰이 같으면 logit 비교가 무의미 (silent failure 방지).
        raise RuntimeError(
            f"'부실'/'정상' 의 첫 토큰이 동일합니다 (id={pos}). "
            f"이 토크나이저({BASE_MODEL})에서는 단일 토큰 분류가 불가능합니다."
        )

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = None
    try:
        base = AutoModelForCausalLM.from_pretrained(
            base_ref,
            quantization_config=bnb,
            device_map={"": 0},
            trust_remote_code=True,
            attn_implementation="eager",
        )
        model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
    except Exception:
        # 적재 도중 실패 시 부분 적재된 GPU 메모리를 회수한 뒤 전파(반복 실패 누수 방지).
        del base
        _gpu_cleanup()
        raise
    model.eval()
    return model, tok, pos, neg


def score_with_llm(payload: dict) -> float:
    """Returns default probability in [0,1]."""
    import torch

    model, tok, pos_id, neg_id = _load_llm()
    # 프롬프트는 학습(llm_finetune.format_sample)과 동일한 serialize.build_chat_text 사용.
    # 가중치가 그 포맷에 고정돼 있으므로 한 글자라도 어긋나면 첫 토큰 분포가 흔들린다.
    prompt = build_chat_text(row_to_prompt(payload))
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    # 단일 GPU forward 직렬화 — concurrency>1 에서 동시 forward 의 메모리 중첩 OOM 방지.
    with _infer_sem:
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1]
    probs = torch.softmax(logits[[neg_id, pos_id]], dim=-1)
    return float(probs[1].item())
