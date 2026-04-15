#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

TOKENIZER_PATH="${TOKENIZER_PATH:-Qwen/Qwen2.5-0.5B}"
CUDA_DEVICES="${CUDA_DEVICES:-1,2,3}"
NPROC="${NPROC:-3}"

TARGET_GB="${TARGET_GB:-50}"
OUT_DATA="${OUT_DATA:-$ROOT_DIR/dataset/pretrain_mix_${TARGET_GB}g_balanced.jsonl}"
STATS_PATH="${STATS_PATH:-$ROOT_DIR/dataset/pretrain_mix_${TARGET_GB}g_balanced_stats.json}"
MIN_CHARS="${MIN_CHARS:-32}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"

MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
BATCH_SIZE="${BATCH_SIZE:-4}"
ACCUM_STEPS="${ACCUM_STEPS:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DTYPE="${DTYPE:-float16}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-5e-4}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
SAVE_WEIGHT="${SAVE_WEIGHT:-pretrain}"
FROM_RESUME="${FROM_RESUME:-1}"

export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

mkdir -p "$ROOT_DIR/dataset" "$ROOT_DIR/out" "$ROOT_DIR/checkpoints"

IFS=',' read -r -a GPU_LIST <<< "$CUDA_DEVICES"
for gpu in "${GPU_LIST[@]}"; do
  if ! nvidia-smi -i "$gpu" --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
    echo "[ERROR] GPU $gpu is not healthy/visible. Please adjust CUDA_DEVICES."
    exit 1
  fi
done

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" NPROC="$NPROC" python - <<'PY'
import os, sys, torch
need = int(os.environ.get("NPROC", "1"))
ok = torch.cuda.is_available()
count = torch.cuda.device_count() if ok else 0
if (not ok) or count < need:
    print(
        f"[ERROR] PyTorch CUDA preflight failed: is_available={ok}, device_count={count}, required={need}."
    )
    print("[HINT] GPU driver/runtime is currently unhealthy. Try rebooting host or removing faulty GPU from service.")
    sys.exit(2)
print(f"[INFO] CUDA preflight passed: {count} visible devices")
PY

if [[ "$FORCE_REBUILD" == "1" || ! -s "$OUT_DATA" ]]; then
  python dataset/prepare_pretrain_50g_balanced.py \
    --out_path "$OUT_DATA" \
    --stats_path "$STATS_PATH" \
    --target_gb "$TARGET_GB" \
    --min_chars "$MIN_CHARS"
else
  echo "[INFO] Reusing existing dataset: $OUT_DATA"
  if [[ -s "$STATS_PATH" ]]; then
    echo "[INFO] Reusing existing stats: $STATS_PATH"
  fi
fi

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" NPROC="$NPROC" python -m torch.distributed.run --nproc_per_node="$NPROC" trainer/train_pretrain.py \
  --data_path "$OUT_DATA" \
  --save_dir "$ROOT_DIR/out" \
  --save_weight "$SAVE_WEIGHT" \
  --checkpoint_dir "$ROOT_DIR/checkpoints" \
  --from_resume "$FROM_RESUME" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --vocab_size -1 \
  --max_seq_len "$MAX_SEQ_LEN" \
  --batch_size "$BATCH_SIZE" \
  --accumulation_steps "$ACCUM_STEPS" \
  --num_workers "$NUM_WORKERS" \
  --dtype "$DTYPE" \
  --epochs "$EPOCHS" \
  --learning_rate "$LR" \
  --log_interval "$LOG_INTERVAL" \
  --save_interval "$SAVE_INTERVAL" 2>&1 | tee "$ROOT_DIR/out/pretrain_${TARGET_GB}g_balanced_$(date +%F_%H-%M-%S).log"
