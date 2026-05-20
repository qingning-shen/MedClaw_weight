#!/usr/bin/env bash
# Full SFT pipeline: data download → medical distillation → merge → LoRA training.
# Execute from the model_training/ directory on the GPU server.
#
# Prerequisites: run server_setup.sh first.
# Usage: bash run_sft.sh

set -e
cd "$(dirname "$0")"   # always run from model_training/

# Pause function for examining output between steps
pause() {
  echo ""
  read -p "Press Enter to continue... "
  echo ""
}

# Project root (parent of model_training/)
DIR_PROJECT="$(cd .. && pwd)"
echo "=== MedClaw SFT Pipeline ==="
echo "Project dir: $DIR_PROJECT"
echo ""

# ── HuggingFace mirror (speeds up dataset + model downloads in China) ─────────
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
echo "HF mirror : $HF_ENDPOINT"
echo ""

# ── Locate OpenClaw skills (outside project) ─────────────────────────────────
OPENCLAW_DIR="${OPENCLAW_DIR:-$DIR_PROJECT/../OpenClaw-Medical-Skills}"
if [ ! -d "$OPENCLAW_DIR/skills" ]; then
  echo "[pre] OpenClaw-Medical-Skills not found at $OPENCLAW_DIR"
  echo "      Cloning now..."
  git clone https://github.com/FreedomIntelligence/OpenClaw-Medical-Skills.git "$OPENCLAW_DIR"
fi
echo "OpenClaw  : $OPENCLAW_DIR/skills"
echo ""

# ── Step 1: Convert pre-downloaded general datasets ────────────────────────────
echo "[1/4] Converting pre-downloaded datasets (JSON/JSONL in data/raw/)..."
echo "      xlam: 4000 samples | slimorca: 1500 | glaive: 500"
echo ""
echo "  First, download JSON/JSONL files from HuggingFace and place in data/raw/:"
echo "    - Salesforce/xlam-function-calling-60k  → xlam.json[.l]"
echo "    - Open-Orca/SlimOrca                    → slimorca.json[.l]"
echo "    - glaiveai/glaive-function-calling-v2  → glaive.json[.l]"
echo ""
python scripts/convert_local_datasets.py \
  --input_dir data/raw \
  --output_dir data/raw \
  --xlam_n 4000 \
  --slimorca_n 1500 \
  --glaive_n 500
echo ""
pause

# ── Step 2: Distill medical tool-calling data from OpenClaw ──────────────────
echo "[2/4] Distilling medical tool-calling data from OpenClaw skills..."
python scripts/convert_openclaw.py \
  --skills_dir "$OPENCLAW_DIR/skills" \
  --output_file data/distill/medical_tool_calls.jsonl \
  --num_samples 2000
echo ""
pause

# ── Step 3: Merge all datasets ────────────────────────────────────────────────
echo "[3/4] Merging all datasets and creating train/val split..."
python scripts/merge_datasets.py \
  --raw_dir data/raw \
  --distill_dir data/distill \
  --output_dir data/processed
echo ""
pause

# ── Step 4: LoRA SFT ──────────────────────────────────────────────────────────
echo "[4/4] Starting LoRA SFT (8x H20, ~10 min)..."
echo "      Output: output/sft_medclaw/"
echo ""
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift sft \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --tuner_type lora \
  --lora_rank 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --dataset data/processed/train.jsonl \
  --val_dataset data/processed/val.jsonl \
  --max_length 2048 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2.0e-4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --weight_decay 0.01 \
  --output_dir output/sft_medclaw \
  --logging_steps 10 \
  --save_strategy epoch \
  --eval_steps 100 \
  --bf16 true \
  --gradient_checkpointing true \
  --dataloader_num_workers 4 \
  --seed 42 \
  --report_to none

echo ""
echo "=== SFT complete ==="
echo "Checkpoint: output/sft_medclaw/"
echo ""
echo "Next: bash run_rl.sh"
