#!/usr/bin/env bash
# Run GRPO RL fine-tuning on top of the SFT checkpoint.
# Execute from the model_training/ directory.
# Must run run_sft.sh first.
set -e

echo "=== MedClaw GRPO RL Fine-tuning (8x H20, ~5 min) ==="

# Verify SFT output exists
if [ ! -d "output/sft_medclaw" ]; then
  echo "ERROR: SFT checkpoint not found at output/sft_medclaw"
  echo "Run ./run_sft.sh first."
  exit 1
fi

if [ ! -f "data/processed/rl_prompts.jsonl" ]; then
  echo "ERROR: RL prompt file not found. Run ./run_sft.sh first."
  exit 1
fi

# Auto-detect latest SFT checkpoint
SFT_VERSION=$(ls -t output/sft_medclaw/ | head -1)
SFT_CKPT="output/sft_medclaw/$SFT_VERSION/checkpoint-28"
if [ ! -d "$SFT_CKPT" ]; then
  SFT_CKPT=$(ls -td output/sft_medclaw/$SFT_VERSION/checkpoint-* | head -1)
fi
echo "Using SFT checkpoint: $SFT_CKPT"

CUDA_VISIBLE_DEVICES=0 \
swift rlhf \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --adapters "$SFT_CKPT" \
  --model_type qwen3 \
  --template qwen3_nothinking \
  --tuner_type lora \
  --lora_rank 32 \
  --lora_alpha 64 \
  --rlhf_type grpo \
  --num_generations 16 \
  --reward_funcs scripts.reward_fn.tool_call_format_reward \
  --dataset data/processed/rl_prompts.jsonl \
  --max_length 1024 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5.0e-6 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --kl_coef 0.05 \
  --temperature 0.9 \
  --output_dir output/rl_medclaw \
  --logging_steps 5 \
  --save_strategy epoch \
  --bf16 true \
  --gradient_checkpointing true \
  --seed 42 \
  --report_to none

echo "=== RL complete. Final model saved to output/rl_medclaw ==="
echo ""
echo "To export as a standalone model (merge LoRA weights):"
echo "  swift export --model output/rl_medclaw --merge_lora true --output_dir output/medclaw_final"
echo ""
echo "To serve with vLLM:"
echo "  cd ../model_server && bash serve.sh ../model_training/output/medclaw_final"
