# QiuMind LLM Pretraining

A lightweight pretraining pipeline for the QiuMind causal language model.

This repository includes:
- model definition in `model/QiuModel.py`
- JSONL pretraining dataset loader in `dataset/lm_dataset.py`
- data mix builders in `dataset/prepare_pretrain_mix.py` and `dataset/prepare_pretrain_50g_balanced.py`
- distributed training entry in `trainer/train_pretrain.py`
- one-click launch scripts in `scripts/`

## 1. Environment

Recommended:
- Linux
- Python 3.10+ (3.10/3.11 tested best for HF + torch stacks)
- CUDA GPU(s) for training

Create and activate env:

```bash
conda create -n qiumind-pretrain python=3.10 -y
conda activate qiumind-pretrain
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Project Layout

```text
checkpoints/                 # resume states and checkpoints
  pretrain_512_resume.pth
  pretrain_512.pth
dataset/
  lm_dataset.py
  prepare_pretrain_50g_balanced.py
  prepare_pretrain_mix.py
model/
  QiuModel.py
scripts/
  run_pretrain_50g_balanced.sh
  run_pretrain_oneclick.sh
trainer/
  train_pretrain.py
  trainer_utils.py
```

## 3. Data Format

Training expects a JSONL file where each line is:

```json
{"text": "your plain text sample"}
```

## 4. Build Dataset (Optional)

### Option A: stable profile mix

```bash
python dataset/prepare_pretrain_mix.py \
  --profile pretrain_stable \
  --out_path dataset/pretrain_mix_pretrain_stable.jsonl \
  --stats_path dataset/pretrain_mix_pretrain_stable_stats.json
```

### Option B: fixed-size 50/50 zh-en dataset

```bash
python dataset/prepare_pretrain_50g_balanced.py \
  --out_path dataset/pretrain_mix_50g_balanced.jsonl \
  --stats_path dataset/pretrain_mix_50g_balanced_stats.json \
  --target_gb 50
```

## 5. Quick Start (One Command)

### Mixed profile one-click

```bash
bash scripts/run_pretrain_oneclick.sh
```

Useful environment overrides:

```bash
CUDA_DEVICES=0,1 NPROC=2 PROFILE=pretrain_stable MAX_SEQ_LEN=1024 \
BATCH_SIZE=4 ACCUM_STEPS=16 DTYPE=float16 EPOCHS=1 \
bash scripts/run_pretrain_oneclick.sh
```

### 50G balanced one-click

```bash
bash scripts/run_pretrain_50g_balanced.sh
```

Example:

```bash
CUDA_DEVICES=0,1 NPROC=2 TARGET_GB=10 FORCE_REBUILD=1 \
MAX_SEQ_LEN=1024 BATCH_SIZE=4 ACCUM_STEPS=16 \
bash scripts/run_pretrain_50g_balanced.sh
```

## 6. Manual Training Command

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 trainer/train_pretrain.py \
  --data_path dataset/pretrain_mix_pretrain_stable.jsonl \
  --save_dir out \
  --checkpoint_dir checkpoints \
  --save_weight pretrain \
  --from_resume 1 \
  --tokenizer_path Qwen/Qwen2.5-0.5B \
  --vocab_size -1 \
  --max_seq_len 1024 \
  --batch_size 4 \
  --accumulation_steps 16 \
  --num_workers 4 \
  --dtype float16 \
  --epochs 1 \
  --learning_rate 5e-4
```

## 7. Resume Training

Set `--from_resume 1` (already enabled in scripts).

The trainer restores state from:
- `checkpoints/<save_weight>_<hidden_size>[_moe]_resume.pth`

Model weights are also saved to:
- `out/<save_weight>_<hidden_size>[_moe].pth`

## 8. Common Args

- `--tokenizer_path`: local tokenizer path or HF model id
- `--vocab_size`: set `-1` to infer from tokenizer
- `--hidden_size`, `--num_hidden_layers`: model scale
- `--max_seq_len`: sequence length
- `--dtype`: `float16` or `bfloat16`
- `--use_moe`: enable MoE (`0` or `1`)
- `--from_weight`: load warm-start weight prefix, use `none` for scratch

## 9. Notes

- First run may take time due to downloading HF datasets/models.
- If you use private mirrors or offline cache, set your HF env vars accordingly.
- `swanlab` is optional and only used when `--use_wandb` is passed.

## 10. Reproducibility Checklist

1. Use the same `tokenizer_path`.
2. Keep `hidden_size`, `num_hidden_layers`, and `max_seq_len` unchanged.
3. Keep `batch_size`, `accumulation_steps`, and `learning_rate` unchanged.
4. Use the same data JSONL and same `from_resume` behavior.
5. Use the same number of GPUs for exact throughput parity.
