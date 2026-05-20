# model_server

Serves the MedClaw model via vLLM with an OpenAI-compatible API.  
Requires a GPU (cloud instance or WSL2 with CUDA).

Runs in the **`medclaw-vllm`** conda env (separate from the backend).

## Two model modes

| Mode | What it serves | When to use |
|---|---|---|
| `mock` | `Qwen/Qwen3-4B-Instruct` (HF, auto-download) | Baseline testing before training |
| `trained` | `model_training/output/medclaw_final` | After LoRA merge |
| `<path>` | Any explicit HF repo id or local directory | Custom checkpoints |

## Quick start

```bash
# Mock model (baseline, downloads ~8 GB on first run)
bash serve.sh mock

# Fine-tuned model (after LoRA merge step below)
bash serve.sh trained

# Or start everything via the top-level launcher
bash start.sh --model mock
bash start.sh --model trained
bash start.sh --test        # no GPU: backend returns mock responses
```

## LoRA merge (required once after training)

vLLM needs fully merged weights, not a raw LoRA adapter.

```bash
cd model_training
swift export \
  --model output/rl_medclaw \
  --merge_lora true \
  --output_dir output/medclaw_final
```

`output/medclaw_final/` is a standard HuggingFace safetensors directory.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | Port to listen on |
| `HOST` | `0.0.0.0` | Bind address |
| `HF_TOKEN` | — | HuggingFace token if model requires auth |
