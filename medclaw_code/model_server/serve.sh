#!/usr/bin/env bash
# Serve a MedClaw model with vLLM (medclaw-vllm conda env).
#
# Usage:
#   bash serve.sh mock              # baseline Qwen/Qwen3-4B-Instruct from HuggingFace
#   bash serve.sh trained           # LoRA-merged fine-tuned model (model_training/output/medclaw_final)
#   bash serve.sh /absolute/path    # any explicit HF or local path
#
# Optional env vars:
#   PORT          (default 8000)
#   HOST          (default 0.0.0.0)
#   HF_TOKEN      set if model requires authentication``
#   HF_ENDPOINT   HuggingFace mirror, e.g. https://hf-mirror.com  (recommended in China)

set -e

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

# ── Resolve model path ────────────────────────────────────────────────────────
MODE="${1:-mock}"
case "$MODE" in
  mock)
    MODEL_PATH="/root/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Instruct-2507/"
    MODEL_LABEL="mock (Qwen/Qwen3-4B-Instruct)"
    ;;
  trained)
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    MODEL_PATH="$SCRIPT_DIR/../model_training/output/medclaw_final"
    MODEL_LABEL="trained ($MODEL_PATH)"
    if [[ ! -d "$MODEL_PATH" ]]; then
      echo "ERROR: trained model not found at $MODEL_PATH"
      echo "       Run the LoRA merge step first, or use 'mock' to test with the baseline."
      exit 1
    fi
    ;;
  *)
    # Treat as an explicit path / HF repo id
    MODEL_PATH="$MODE"
    MODEL_LABEL="custom ($MODEL_PATH)"
    ;;
esac

# ── HuggingFace mirror (speeds up first-time model download in China) ─────────
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "=== MedClaw vLLM Server ==="
echo "Mode  : $MODE"
echo "Model : $MODEL_LABEL"
echo "Listen: $HOST:$PORT"
echo ""
echo "Agent env vars to set:"
echo "  MODEL_SERVER_URL=http://localhost:$PORT/v1"
echo "  MODEL_NAME=medclaw"
echo ""

# ── Activate the dedicated conda env ─────────────────────────────────────────
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate medclaw-vllm

# --enable-auto-tool-choice + --tool-call-parser hermes:
#   vLLM intercepts Qwen3's Hermes-style XML tool calls and converts them
#   to standard OpenAI tool_calls JSON before returning to the caller.
vllm serve "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name medclaw \
  --max-model-len 4096 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --dtype bfloat16 \
  --trust-remote-code
