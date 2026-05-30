"""학습된 LoRA 어댑터로 테스트셋 AUC/KS 계산 → artifacts/metrics.json 갱신.

Colab/GPU에서 실행 (어댑터는 Drive 의 Qwen3_fintech 폴더):
    python -m src.training.llm_eval \
        --model_name Qwen/Qwen3-14B \
        --adapter_dir "/content/drive/MyDrive/Colab Notebooks/Qwen3_fintech" \
        --test_file data/llm_seed/test.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger("llm_eval")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen3-14B")
    parser.add_argument("--adapter_dir", default="artifacts/qwen3_lora")
    parser.add_argument("--test_file", default="data/llm_seed/test.jsonl")
    parser.add_argument("--metrics_path", default="artifacts/metrics.json")
    parser.add_argument("--max_samples", type=int, default=1000)
    args = parser.parse_args()

    try:
        import numpy as np
        import torch
        from peft import PeftModel
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            f1_score,
            roc_auc_score,
        )
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except Exception as e:
        log.error("필수 라이브러리 누락: %s", e)
        return

    log.info("Loading model + adapter")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
    )
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    device_index = torch.cuda.current_device() if torch.cuda.is_available() else 0
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name, quantization_config=bnb, device_map={"": device_index}, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    # token IDs for "정상" and "부실"
    POS_TOK = tok.encode("부실", add_special_tokens=False)[0]
    NEG_TOK = tok.encode("정상", add_special_tokens=False)[0]
    log.info("POS token id=%s, NEG token id=%s", POS_TOK, NEG_TOK)
    if POS_TOK == NEG_TOK:
        raise RuntimeError(
            f"'부실'/'정상' 의 첫 토큰이 동일합니다 (id={POS_TOK}). "
            f"이 토크나이저({args.model_name})에서는 단일 토큰 분류가 불가능합니다."
        )

    examples = [json.loads(line) for line in open(args.test_file, encoding="utf-8")]
    if args.max_samples and args.max_samples < len(examples):
        examples = examples[: args.max_samples]
    log.info("Evaluating %d examples", len(examples))

    y_true, y_score = [], []
    for ex in examples:
        prompt = (
            f"<|im_start|>system\n당신은 신용평가 전문가입니다. 신청자 정보를 보고 부실 또는 정상을 한 단어로 답하세요.<|im_end|>\n"
            f"<|im_start|>user\n{ex['instruction']}<|im_end|>\n<|im_start|>assistant\n"
        )
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1]
        probs = torch.softmax(logits[[NEG_TOK, POS_TOK]], dim=-1)
        p_pos = probs[1].item()
        y_score.append(p_pos)
        y_true.append(1 if ex["output"] == "부실" else 0)

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    auc = float(roc_auc_score(y_true, y_score))

    order = np.argsort(-y_score)
    ys = y_true[order]
    pos_cum = np.cumsum(ys) / max(ys.sum(), 1)
    neg_cum = np.cumsum(1 - ys) / max((1 - ys).sum(), 1)
    ks = float(np.max(np.abs(pos_cum - neg_cum)))

    y_pred = (y_score >= 0.5).astype(int)
    metrics = {
        "auc": auc,
        "ks": ks,
        "average_precision": float(average_precision_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "positive_rate": float(y_true.mean()),
        "evaluated_samples": len(y_true),
    }
    log.info("LLM metrics: %s", metrics)

    mp = Path(args.metrics_path)
    mp.parent.mkdir(parents=True, exist_ok=True)
    llm_block = {
        "label_kr": "Qwen3-14B QLoRA (LLM)",
        "status": "trained",
        **metrics,
    }
    if mp.exists():
        payload = json.loads(mp.read_text(encoding="utf-8"))
        payload.setdefault("models", {})["llm_qwen3_14b"] = llm_block
        base_auc = payload.get("models", {}).get("logistic_regression", {}).get("auc")
        if base_auc:
            delta = auc - base_auc
            payload.setdefault("comparison_vs_baseline", {})
            payload["comparison_vs_baseline"]["llm_auc_delta"] = delta
            payload["comparison_vs_baseline"]["llm_auc_pct_improvement"] = (delta / base_auc) * 100
            payload["comparison_vs_baseline"]["llm_meets_10pct_goal"] = delta >= 0.05
        mp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("Updated %s (merged into existing)", mp)
    else:
        # Standalone — save just the LLM metrics for later local merge.
        payload = {"models": {"llm_qwen3_14b": llm_block}}
        mp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("Wrote standalone LLM metrics: %s", mp)


if __name__ == "__main__":
    main()
