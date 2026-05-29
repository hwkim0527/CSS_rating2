"""Qwen2.5-7B QLoRA 파인튜닝 — GCP (A100/L4) 환경용.

선행:
    1) python -m src.data.preprocess
    2) python -m src.training.build_llm_dataset
    3) pip install -r src/training/requirements_llm.txt
    4) huggingface-cli login  (Qwen 모델 다운로드용)

실행 (Vertex AI Custom Job 또는 GCE GPU):
    python -m src.training.llm_finetune \
        --model_name Qwen/Qwen2.5-7B-Instruct \
        --train_file data/llm/train.jsonl \
        --val_file data/llm/val.jsonl \
        --output_dir artifacts/qwen25_lora \
        --num_epochs 3 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --learning_rate 2e-4
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

log = logging.getLogger("llm_finetune")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train_file", default="data/llm_seed/train.jsonl")
    parser.add_argument("--val_file", default="data/llm_seed/val.jsonl")
    parser.add_argument("--output_dir", default="artifacts/qwen25_lora")
    # Defaults are tuned for "efficient" mode (≈1.5–2h on T4 16GB, ~$0 on Colab free).
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Truncate training set to this many samples (for fast iteration).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Skip training; verify dependencies and dataset only.",
    )
    return parser.parse_args()


def format_sample(example: dict) -> str:
    return (
        f"<|im_start|>system\n당신은 신용평가 전문가입니다. 신청자 정보를 보고 부실(1) 또는 정상(0)을 한 단어로 답하세요.<|im_end|>\n"
        f"<|im_start|>user\n{example['instruction']}<|im_end|>\n"
        f"<|im_start|>assistant\n{example['output']}<|im_end|>"
    )


def main() -> None:
    args = parse_args()
    log.info("Args: %s", vars(args))

    # Lazy import — these are heavy. Allows dry_run to skip.
    try:
        import torch  # noqa: F401
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from trl import SFTConfig, SFTTrainer
    except Exception as e:
        log.error("필수 라이브러리 누락: %s", e)
        log.error("GCP GPU 환경에서 requirements_llm.txt 를 먼저 설치하세요.")
        if args.dry_run:
            log.info("--dry_run 모드: 의존성 미설치만 확인했습니다.")
            return
        raise

    train_path = Path(args.train_file)
    val_path = Path(args.val_file)
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"학습 데이터셋이 없습니다: {train_path}, {val_path}. "
            "먼저 'python -m src.training.build_llm_dataset' 를 실행하세요."
        )

    ds = load_dataset(
        "json",
        data_files={"train": str(train_path), "validation": str(val_path)},
    )
    if args.max_train_samples is not None and args.max_train_samples < len(ds["train"]):
        ds["train"] = ds["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
    ds = ds.map(lambda ex: {"text": format_sample(ex)})
    log.info("Dataset sizes: train=%d val=%d", len(ds["train"]), len(ds["validation"]))

    if args.dry_run:
        log.info("Sample prompt:\n%s", ds["train"][0]["text"][:600])
        log.info("Dry-run OK. 실제 학습은 GPU 환경에서 실행하세요.")
        return

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=500,
        save_total_limit=3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        seed=args.seed,
        report_to="none",
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        args=sft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    log.info("Adapter saved → %s", args.output_dir)

    # Persist a summary metric stub so the web app can pick it up.
    metrics_stub = {
        "model": args.model_name,
        "epochs": args.num_epochs,
        "train_examples": len(ds["train"]),
        "val_examples": len(ds["validation"]),
        "final_train_loss": trainer.state.log_history[-1].get("loss"),
    }
    Path(args.output_dir, "training_summary.json").write_text(
        json.dumps(metrics_stub, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Wrote training_summary.json")


if __name__ == "__main__":
    main()
