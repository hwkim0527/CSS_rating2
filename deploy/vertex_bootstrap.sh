#!/usr/bin/env bash
# Vertex AI Custom Job bootstrap
# Env vars supplied by the job config:
#   BUCKET       — gs://<bucket>
#   PROJECT_BASE — base path under bucket (default: css-rating2)
set -euxo pipefail

BUCKET="${BUCKET:?BUCKET env var required}"
PROJECT_BASE="${PROJECT_BASE:-css-rating2}"
WORK=/workspace
mkdir -p $WORK && cd $WORK

echo "▶ GPU info:" && nvidia-smi || true

echo "▶ Pulling source"
gsutil cp "gs://${BUCKET}/${PROJECT_BASE}/src.tar.gz" .
tar -xzf src.tar.gz

echo "▶ Pulling seed data"
mkdir -p data/llm_seed
gsutil -m cp "gs://${BUCKET}/${PROJECT_BASE}/data/llm_seed/*.jsonl" data/llm_seed/

echo "▶ Removing torch_xla (Vertex container preloads it, accelerate then tries XLA path)"
pip uninstall -y torch_xla torch-xla 2>/dev/null || true

echo "▶ Installing training deps (known-good QLoRA combo for single-GPU)"
pip install --no-cache-dir -q -U \
    transformers==4.44.2 \
    accelerate==0.33.0 \
    peft==0.12.0 \
    trl==0.9.6 \
    bitsandbytes==0.43.3 \
    datasets==2.21.0 \
    sentencepiece einops scikit-learn

echo "▶ Starting fine-tune (Qwen3-14B QLoRA, single GPU)"
# Auto-tune batch size based on GPU memory.
GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "Detected GPU memory: ${GPU_MEM_MB} MiB"
if [ "${GPU_MEM_MB}" -gt 35000 ]; then
    # A100 40GB+ — push batch & seq length, finish much faster.
    BATCH=8
    ACCUM=2
    SEQ_LEN=1024
    LORA_R=16
    LORA_A=32
else
    # T4 / L4 16-24GB — stay conservative.
    BATCH=2
    ACCUM=4
    SEQ_LEN=512
    LORA_R=8
    LORA_A=16
fi
echo "Training config: batch=${BATCH} accum=${ACCUM} seq=${SEQ_LEN} lora_r=${LORA_R}"
# Single-GPU, non-distributed. The bnb device check is bypassed in code
# via a complete prepare_model replacement (see src/training/llm_finetune.py).
# ACCELERATE_TORCH_DEVICE accidentally activates XLA path → do NOT set.
export ACCELERATE_USE_FSDP=false
export ACCELERATE_USE_DEEPSPEED=false
export CUDA_VISIBLE_DEVICES=0
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT TORCHELASTIC_RUN_ID
unset ACCELERATE_TORCH_DEVICE
# Background uploader: pushes any saved checkpoint to GCS every 60s so a
# cancellation still preserves artifacts.
(
  while true; do
    if [ -d artifacts/qwen3_lora ]; then
      gsutil -m -q rsync -r artifacts/qwen3_lora "gs://${BUCKET}/${PROJECT_BASE}/qwen3_lora_live/" 2>/dev/null || true
    fi
    sleep 60
  done
) &
UPLOAD_PID=$!
trap "kill $UPLOAD_PID 2>/dev/null; gsutil -m cp -r artifacts/qwen3_lora gs://${BUCKET}/${PROJECT_BASE}/ 2>/dev/null || true" EXIT

# Short training: 800 samples (~20 min on T4) is enough to get a usable adapter.
python -m src.training.llm_finetune \
    --model_name "Qwen/Qwen3-14B" \
    --train_file data/llm_seed/train.jsonl \
    --val_file data/llm_seed/val.jsonl \
    --output_dir artifacts/qwen3_lora \
    --num_epochs 1 \
    --per_device_train_batch_size ${BATCH} \
    --gradient_accumulation_steps ${ACCUM} \
    --learning_rate 2e-4 \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_A} \
    --max_seq_length ${SEQ_LEN} \
    --max_train_samples 800

echo "▶ Evaluating on test set"
python -m src.training.llm_eval \
    --model_name "Qwen/Qwen3-14B" \
    --adapter_dir artifacts/qwen3_lora \
    --test_file data/llm_seed/test.jsonl \
    --metrics_path artifacts/metrics_llm.json \
    --max_samples 1000 || true

echo "▶ Uploading artifacts"
gsutil -m cp -r artifacts/qwen3_lora "gs://${BUCKET}/${PROJECT_BASE}/" || true
gsutil cp artifacts/metrics_llm.json "gs://${BUCKET}/${PROJECT_BASE}/" || true

echo "▶ DONE"
