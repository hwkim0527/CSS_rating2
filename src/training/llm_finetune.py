"""Qwen3-14B QLoRA 파인튜닝 — Colab(T4/L4/A100) 및 GCP 환경용.

Qwen2.5-7B → Qwen3-14B 로 업그레이드. Qwen3 는 같은 크기 대비 약 1.5세대
높은 성능을 내며(14B ≈ Qwen2.5-32B급), 추론·수치 처리에서 이점이 큽니다.

⚠️ Qwen3 아키텍처는 transformers>=4.51.0 필요.
⚠️ 단일 토큰 분류(부실/정상)를 위해 assistant 턴이 라벨로 바로 시작하도록
   학습합니다. 이렇게 하면 Qwen3 의 기본 <think> 출력이 억제되어, 추론 시
   첫 토큰 로짓만으로 부실 확률을 안정적으로 읽을 수 있습니다.

선행:
    1) python -m src.data.preprocess
    2) python -m src.training.build_llm_dataset
    3) pip install -r src/training/requirements_llm.txt
    (Qwen3 는 공개 모델이라 huggingface-cli login 불필요)

실행 (Colab — 체크포인트를 Google Drive 에 저장하며 끊겨도 재개):
    python -m src.training.llm_finetune \
        --model_name Qwen/Qwen3-14B \
        --train_file data/llm_seed/train.jsonl \
        --val_file data/llm_seed/val.jsonl \
        --output_dir "/content/drive/MyDrive/Colab Notebooks/Qwen3_fintech" \
        --num_epochs 1 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --learning_rate 2e-4 \
        --resume_from_checkpoint auto
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
    parser.add_argument("--model_name", default="Qwen/Qwen3-14B")
    parser.add_argument("--train_file", default="data/llm_seed/train.jsonl")
    parser.add_argument("--val_file", default="data/llm_seed/val.jsonl")
    parser.add_argument("--output_dir", default="artifacts/qwen3_lora")
    # Defaults tuned for Qwen3-14B QLoRA on a 16GB T4 (batch 1 + accum 8).
    # On L4/A100 the Colab notebook raises batch size and seq length.
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--save_steps", type=int, default=50,
                        help="N 스텝마다 체크포인트 저장 (output_dir 가 Drive 면 Drive 에 누적).")
    parser.add_argument("--save_total_limit", type=int, default=2,
                        help="최신 체크포인트 N개만 유지 (Drive 용량 보호).")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Truncate training set to this many samples (for fast iteration).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume_from_checkpoint",
        default="auto",
        help="'auto'=output_dir 안의 최신 checkpoint-* 에서 자동 재개(없으면 처음부터), "
             "경로 지정 시 해당 체크포인트에서 재개, 'none'=항상 처음부터.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Skip training; verify dependencies and dataset only.",
    )
    return parser.parse_args()


def _resolve_resume(output_dir: str, resume_arg: str):
    """--resume_from_checkpoint 값을 trainer.train() 인자로 변환.

    'auto'  → output_dir 안에 checkpoint-* 가 있으면 최신 것, 없으면 None
    'none'  → None (처음부터)
    그 외   → 그 경로 그대로 사용
    """
    if resume_arg == "none":
        return None
    if resume_arg != "auto":
        return resume_arg
    out = Path(output_dir)
    if not out.exists():
        return None
    ckpts = [p for p in out.glob("checkpoint-*") if p.is_dir()]
    if not ckpts:
        return None
    latest = max(ckpts, key=lambda p: int(p.name.split("-")[-1]))
    log.info("재개할 체크포인트 발견: %s", latest)
    return str(latest)


def format_sample(example: dict) -> str:
    # 프롬프트는 serialize 의 단일 진실 원천을 사용 — 추론/평가와 글자까지 동일 보장.
    from src.training.serialize import build_chat_text

    return build_chat_text(example["instruction"], example["output"])


def main() -> None:
    args = parse_args()
    log.info("Args: %s", vars(args))

    # Lazy import — these are heavy. Allows dry_run to skip.
    try:
        import torch  # noqa: F401
        # We replace Accelerator.prepare_model entirely (see below) — no env
        # var dance needed. ACCELERATE_TORCH_DEVICE breaks because Vertex's
        # PyTorch container has torch_xla installed and setting it triggers
        # XLA initialization.
        import os as _os
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from trl import SFTConfig, SFTTrainer

        # Replace Accelerator.prepare_model entirely. The original raises
        # ValueError when bnb is loaded + accelerator.device is `cuda` (no
        # index) but model is on `cuda:0`. Calling original with
        # device_placement=False still triggers the same check.
        # For our single-GPU case, the model is already on the GPU and no
        # device movement is needed. We just register it.
        from accelerate import Accelerator

        def _patched_prepare_model(self, model, device_placement=None, evaluation_mode=False):
            self._models.append(model)
            if evaluation_mode:
                model = model.eval()
            return model

        Accelerator.prepare_model = _patched_prepare_model
        log.info("Replaced Accelerator.prepare_model with single-GPU bypass")
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

    import torch as _torch
    # Single-GPU only. Hardcode device 0 to avoid any race between
    # `current_device()` and accelerate's own device tracking.
    device_index = 0
    # T4 does not have native bf16 (despite PyTorch's is_bf16_supported() returning
    # True on Turing+ with CUDA 11+). Force fp16 unless GPU is Ampere (SM>=8.0).
    if _torch.cuda.is_available():
        gpu_name = _torch.cuda.get_device_name(device_index)
        sm_major = _torch.cuda.get_device_capability(device_index)[0]
        use_bf16 = sm_major >= 8  # Ampere and newer (A100, L4, H100 ...)
    else:
        gpu_name = "none"
        use_bf16 = False
    compute_dtype = _torch.bfloat16 if use_bf16 else _torch.float16
    log.info("compute_dtype=%s | device=cuda:%d | bf16=%s | GPU=%s",
             compute_dtype, device_index, use_bf16, gpu_name)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map={"": device_index},   # hardcoded single GPU
        torch_dtype=compute_dtype,
        trust_remote_code=True,
        # Eager attention sidesteps the SDPA dtype mismatch bug that fires
        # when prepare_model_for_kbit_training upcasts layer norms to fp32
        # while keeping the attention mask in bf16/fp16. Slightly slower than
        # SDPA but reliable for QLoRA + gradient checkpointing.
        attn_implementation="eager",
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Verify ALL parameters landed on the right device before training.
    model_devices = {p.device for p in model.parameters() if p.device.type != "meta"}
    log.info("Model parameter devices: %s", model_devices)
    assert all(d == _torch.device("cuda", device_index) for d in model_devices), \
        f"Model parameters spread across devices: {model_devices}"

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    # Note: do NOT call get_peft_model here; pass peft_config to SFTTrainer so
    # it wires accelerate + peft + 4-bit in the correct order internally.

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        seed=args.seed,
        report_to="none",
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        args=sft_config,
        peft_config=lora_config,
    )
    trainer.model.print_trainable_parameters()

    resume = _resolve_resume(args.output_dir, args.resume_from_checkpoint)
    if resume:
        log.info("체크포인트에서 학습 재개: %s", resume)
    else:
        log.info("처음부터 학습 시작 (재개할 체크포인트 없음)")
    trainer.train(resume_from_checkpoint=resume)

    trainer.save_model(args.output_dir)
    log.info("최종 어댑터 저장 완료 → %s", args.output_dir)

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
