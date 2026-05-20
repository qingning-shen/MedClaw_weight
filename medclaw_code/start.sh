#!/usr/bin/env bash
# Start the MedClaw demo stack.
# Run from the project root: bash start.sh [OPTIONS]
#
# Options:
#   --model mock      Use baseline Qwen/Qwen3-4B-Instruct (default when GPU available)
#   --model trained   Use your LoRA-merged fine-tuned model
#   --model <path>    Use any explicit HF repo id or local path
#   --test            Skip vLLM entirely; backend returns mock responses (no GPU needed)
#
# Both --model and --test use the medclaw conda env for the backend/agent.
# The model server runs in the medclaw-vllm conda env (managed by serve.sh).
#
# Conda envs:
#   medclaw       — backend + agent (FastAPI, smolagents, openai)
#   medclaw-vllm  — vLLM model server (torch, vllm)

set -e

BACKEND_PORT=5000
MODEL_MODE="mock"   # default: serve baseline model
TEST_MODE=0

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --test)
      TEST_MODE=1
      shift
      ;;
    --model)
      MODEL_MODE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: bash start.sh [--model mock|trained|<path>] [--test]"
      exit 1
      ;;
  esac
done

# ── Activate backend/agent conda env ─────────────────────────────────────────
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate medclaw

echo "=== MedClaw Demo Stack ==="
echo ""

# ── Model server ──────────────────────────────────────────────────────────────
if [[ $TEST_MODE -eq 1 ]]; then
  echo "[model] Test mode — no model server started."
  echo "        Backend will return mock responses (no GPU required)."
  MODEL_SERVER_URL="http://localhost:8000/v1"   # unused, set for clarity
  MODEL_NAME="medclaw"
  MEDCLAW_TEST_MODE=1
else
  echo "[model] Starting vLLM server (mode: $MODEL_MODE) ..."
  bash model_server/serve.sh "$MODEL_MODE" &
  VLLM_PID=$!
  trap "kill $VLLM_PID 2>/dev/null; echo ''; echo '[model] vLLM stopped.'" EXIT

  echo -n "        Waiting for vLLM to be ready"
  for i in $(seq 1 40); do
    if curl -sf http://localhost:8000/v1/models >/dev/null 2>&1; then
      echo " ready."
      break
    fi
    if [[ $i -eq 40 ]]; then
      echo ""
      echo "ERROR: vLLM did not become ready after 2 minutes."
      echo "       Check GPU availability and model path, then retry."
      exit 1
    fi
    echo -n "."
    sleep 3
  done
  MODEL_SERVER_URL="http://localhost:8000/v1"
  MODEL_NAME="medclaw"
  MEDCLAW_TEST_MODE=0
fi

# ── Backend ───────────────────────────────────────────────────────────────────
echo "[backend] Starting on http://localhost:$BACKEND_PORT ..."
echo ""
echo "  Backend API : http://localhost:$BACKEND_PORT"
echo "  Frontend    : open frontend/index.html in your browser"
echo "  (Windows users: same address http://localhost:$BACKEND_PORT works via WSL2)"
echo ""
echo "Press Ctrl-C to stop."
echo ""

cd backend
PYTHONUTF8=1 \
MEDCLAW_TEST_MODE=$MEDCLAW_TEST_MODE \
MODEL_SERVER_URL="$MODEL_SERVER_URL" \
MODEL_NAME="$MODEL_NAME" \
python -m uvicorn main:app --host 0.0.0.0 --port "$BACKEND_PORT"
