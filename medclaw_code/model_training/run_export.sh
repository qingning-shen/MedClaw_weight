#!/usr/bin/env bash
# Merge LoRA adapter weights into a single HuggingFace model directory.
# Run after run_rl.sh (or run_sft.sh if skipping RL).
# Execute from the model_training/ directory.
set -e
cd "$(dirname "$0")"

echo "=== MedClaw LoRA Export ==="

# Auto-detect latest checkpoint and determine source type
if [ -d "output/rl_medclaw" ]; then
  # RL checkpoint
  SRC_DIR=$(ls -t output/rl_medclaw/ | head -1)
  SRC_CKPT=$(ls -td output/rl_medclaw/$SRC_DIR/checkpoint-* | head -1)
  SRC_TYPE="RL"
elif [ -d "output/sft_medclaw" ]; then
  # SFT checkpoint
  SRC_DIR=$(ls -t output/sft_medclaw/ | head -1)
  SRC_CKPT=$(ls -td output/sft_medclaw/$SRC_DIR/checkpoint-* | head -1)
  SRC_TYPE="SFT"
else
  echo "ERROR: No checkpoint found. Run run_sft.sh first."
  exit 1
fi

echo "Source: $SRC_TYPE checkpoint ($SRC_CKPT)"

OUT="output/medclaw_final"
echo "Output: $OUT"
echo ""

swift export \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --adapters "$SRC_CKPT" \
  --merge_lora true \
  --output_dir "$OUT" \
  --model_type qwen3 \
  --template qwen3_nothinking

echo ""
echo "=== Export complete ==="
echo "Merged model: $OUT"
du -sh "$OUT"
echo ""
echo "To serve locally (needs GPU):"
echo "  cd ../model_server && bash serve.sh trained"
echo ""
echo "To download to your local machine, run on LOCAL:"
echo "  bash model_training/fetch_model.sh <user>@<server_ip>"
