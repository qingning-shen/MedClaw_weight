# MedClaw
DDA4210 Final Project (CUHK-Shenzhen Spring 2026)

A small-model biomedical AI agent: fine-tune Qwen3-4B for tool calling, then deploy it in an agentic loop to answer biomedical questions using real medical APIs.

## Architecture

```
[Browser frontend]
        │ HTTP POST /chat
[FastAPI backend]
        │ Python call
[smolagents ToolCallingAgent]  ←→  [8 biomedical skills / tools]
        │ OpenAI-compatible API
[vLLM model server]
        │
[Fine-tuned Qwen3-4B (LoRA SFT + GRPO)]
```

## Repository structure

```
MedClaw/
├── model_training/   # Data pipeline + LoRA SFT + GRPO RL (run on cloud GPU)
├── model_server/     # vLLM serve script (run on GPU instance)
├── agent/            # smolagents agent + 8 biomedical skill tools
├── backend/          # FastAPI /chat endpoint
└── frontend/         # Single-page chat UI (no build step)
```

See the `README.md` in each subdirectory for detailed instructions.

## Quick start (local demo, no GPU)

```bash
# 1. Install backend + agent dependencies
pip install -r backend/requirements.txt -r agent/requirements.txt

# 2. Start the backend in test mode (no vLLM needed)
cd backend
MEDCLAW_TEST_MODE=1 uvicorn main:app --host 0.0.0.0 --port 5000

# 3. Open the frontend
#    Double-click frontend/index.html in your browser
#    or: python -m http.server 8080 (run from frontend/)
```

## Full deployment (with trained model)

```bash
# On the GPU cloud instance:
# 1. Train the model
cd model_training && bash run_sft.sh && bash run_rl.sh

# 2. Serve with vLLM
cd model_server && bash serve.sh ../model_training/output/medclaw_final

# 3. Start backend (pointing at the GPU server)
cd backend
MODEL_SERVER_URL=http://<gpu-ip>:8000/v1 uvicorn main:app --port 5000

# 4. Open frontend/index.html
```

## Training summary

| Data | Size | Source |
|---|---|---|
| General tool calling | 4,000 samples | Salesforce/xlam-function-calling-60k |
| General instruction | 1,500 samples | Open-Orca/SlimOrca |
| Multi-turn tool calling | 500 samples | glaiveai/glaive-function-calling-v2 |
| Medical tool calling | 2,000 samples | Rule-based distillation from OpenClaw-Medical-Skills |

- **SFT**: LoRA rank-16 on Qwen3-4B-Instruct, ~1.5 h on a T4
- **RL**: GRPO with format-validity reward, ~30 min on a T4
- **Total GPU cost**: ~0 on Kaggle free tier (2× T4, 30 h/week)
