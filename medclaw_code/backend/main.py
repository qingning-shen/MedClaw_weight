"""
MedClaw software backend — FastAPI server that routes chat requests to the agent.

Endpoints:
    POST /chat   — run the agent on a user message, return answer + tool steps
    GET  /health — liveness check

Run:
    uvicorn main:app --host 0.0.0.0 --port 5000 --reload

Environment variables (all optional):
    MEDCLAW_TEST_MODE=1        — return mock responses, no vLLM needed
    MODEL_SERVER_URL           — vLLM base URL (default http://localhost:8000/v1)
    MODEL_NAME                 — model name (default medclaw)
    AGENT_MAX_STEPS            — max agent reasoning steps (default 5)
"""

import sys
import os

# Allow the backend to import from the sibling agent/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agent"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from agent import run_agent

app = FastAPI(title="MedClaw API", version="1.0")

# Allow the frontend (served from a local file or different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str


class ToolStep(BaseModel):
    tool: str
    input: dict
    output: str


class ChatResponse(BaseModel):
    answer: str
    steps: list[ToolStep]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok", "test_mode": os.getenv("MEDCLAW_TEST_MODE", "0") == "1"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        result = run_agent(message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    steps = [
        ToolStep(
            tool=s.get("tool", ""),
            input=s.get("input") if isinstance(s.get("input"), dict) else {"raw": str(s.get("input", ""))},
            output=str(s.get("output", "")),
        )
        for s in result.get("steps", [])
    ]

    return ChatResponse(answer=result["answer"], steps=steps)
