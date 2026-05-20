"""
MedClaw agent — assembles a smolagents ToolCallingAgent backed by the vLLM model server.

Usage (called by the backend, not directly):
    from agent.agent import run_agent
    result = run_agent("Find papers about BRCA1")
"""

import os
from smolagents import ToolCallingAgent, OpenAIServerModel

# Monkey-patch OpenAIServerModel to fix vLLM 0.8.0 compatibility
# vLLM 0.8.0 rejects tool_choice="required" but accepts "auto"
_original_call = OpenAIServerModel.__call__

def _patched_call(self, messages, stop_sequences=None, grammar=None, tools_to_call_from=None, **kwargs):
    # Import here to avoid circular imports
    from smolagents.models import get_clean_message_list, get_tool_json_schema
    from smolagents import ChatMessage, MessageRole

    completion_kwargs = self._prepare_completion_kwargs(
        messages=messages,
        stop_sequences=stop_sequences,
        grammar=grammar,
        tools_to_call_from=tools_to_call_from,
        model=self.model_id,
        custom_role_conversions=self.custom_role_conversions,
        convert_images_to_image_urls=True,
        **kwargs,
    )
    # Fix: vLLM 0.8.0 rejects tool_choice="required", use "auto" instead
    if completion_kwargs.get("tool_choice") == "required":
        completion_kwargs["tool_choice"] = "auto"
    response = self.client.chat.completions.create(**completion_kwargs)
    self.last_input_token_count = response.usage.prompt_tokens
    self.last_output_token_count = response.usage.completion_tokens
    first_message = ChatMessage.from_dict(
        response.choices[0].message.model_dump(include={"role", "content", "tool_calls"}),
        raw=response,
    )
    return self.postprocess_message(first_message, tools_to_call_from)

OpenAIServerModel.__call__ = _patched_call

from skills import ALL_SKILLS

# ── Configuration via environment variables ───────────────────────────────────
MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://localhost:8000/v1")
MODEL_NAME       = os.getenv("MODEL_NAME", "medclaw")
MAX_STEPS        = int(os.getenv("AGENT_MAX_STEPS", "5"))
TEST_MODE        = os.getenv("MEDCLAW_TEST_MODE", "0") == "1"

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are MedClaw, a biomedical AI assistant. "
    "When answering questions about diseases, drugs, genes, clinical trials, "
    "or medical calculations, use the available tools to retrieve up-to-date information. "
    "Always cite the tool results in your final answer. "
    "If a question is not biomedical, answer directly without using tools."
)


def _build_agent() -> ToolCallingAgent:
    model = OpenAIServerModel(
        model_id=MODEL_NAME,
        api_base=MODEL_SERVER_URL,
        api_key="EMPTY",   # vLLM does not enforce API keys by default
    )
    agent = ToolCallingAgent(
        tools=ALL_SKILLS,
        model=model,
        max_steps=MAX_STEPS,
        verbosity_level=1,
    )
    # Override default system prompt with MedClaw-specific one
    agent.system_prompt = SYSTEM_PROMPT
    return agent


def _mock_run(message: str) -> dict:
    """Return a fake response for local testing without vLLM (TEST_MODE=1)."""
    return {
        "answer": (
            f"[TEST MODE] Received: '{message}'\n"
            "In production this would call the fine-tuned Qwen3-4B model via vLLM "
            "and use the registered biomedical tools."
        ),
        "steps": [
            {
                "tool": "pubmed_search",
                "input": {"query": message},
                "output": "[mock result — tool not called in test mode]",
            }
        ],
    }


def _extract_steps(agent: ToolCallingAgent) -> list[dict]:
    """Pull intermediate tool calls from the agent's memory after a run."""
    import json as json_module

    steps = []
    try:
        for step in agent.memory.steps:
            # ToolCallingAgent memory steps have tool_calls and observations
            tool_calls = getattr(step, "tool_calls", None)
            observations = getattr(step, "observations", None)
            if tool_calls:
                for tc in tool_calls:
                    tool_name = getattr(tc, "name", str(tc))
                    tool_args = getattr(tc, "arguments", {})
                    # Handle case where arguments is a JSON string instead of dict
                    if isinstance(tool_args, str):
                        try:
                            tool_args = json_module.loads(tool_args)
                        except (json.JSONDecodeError, ValueError):
                            tool_args = {"raw": tool_args}
                    elif not isinstance(tool_args, dict):
                        tool_args = {"raw": str(tool_args)}
                    steps.append({
                        "tool": tool_name,
                        "input": tool_args,
                        "output": str(observations) if observations else "",
                    })
    except Exception:
        pass  # memory API may differ across smolagents versions
    return steps


def run_agent(message: str) -> dict:
    """
    Run the agent on a user message.

    Returns:
        {"answer": str, "steps": list[dict]}
        Steps are the intermediate tool calls made during reasoning.
    """
    if TEST_MODE:
        return _mock_run(message)

    agent = _build_agent()
    answer = agent.run(message)
    steps = _extract_steps(agent)

    return {"answer": str(answer), "steps": steps}
