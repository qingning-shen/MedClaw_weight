"""
GRPO reward function for MedClaw.
Rewards valid tool-call format in model completions.

ms-swift passes the reward function as a Python module path:
    --reward_funcs scripts.reward_fn.tool_call_format_reward
"""

import json
import re

# Names of the tools the model is allowed to call
VALID_TOOLS = {
    "pubmed_search",
    "drug_lookup",
    "search_clinical_trials",
    "query_gene_variant",
    "get_disease_info",
    "check_drug_interaction",
    "lookup_gene_info",
    "medical_calculator",
}


def _parse_tool_call(text: str) -> dict | None:
    """Try to extract a JSON tool call from the model output."""
    # Hermes-style: <tool_call>{...}</tool_call>
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            return None

    # OpenAI-style embedded JSON object
    m = re.search(r'\{"name"\s*:\s*"[^"]+"\s*,\s*"arguments"', text)
    if m:
        # Find matching closing brace
        start = m.start()
        depth = 0
        for i, ch in enumerate(text[start:]):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:start + i + 1])
                    except json.JSONDecodeError:
                        return None
    return None


def tool_call_format_reward(completions: list, **kwargs) -> list[float]:
    """
    Binary reward:
      +1.0 — completion contains a valid, parseable tool call with a known tool name
      +0.5 — completion contains a tool call that parses as JSON but uses an unknown tool name
       0.0 — no valid tool call found
    """
    rewards = []
    for completion in completions:
        # ms-swift passes completions as list of message dicts or plain strings
        if isinstance(completion, list):
            text = " ".join(
                m.get("content", "") for m in completion if isinstance(m, dict)
            )
        elif isinstance(completion, dict):
            text = completion.get("content", "")
        else:
            text = str(completion)

        parsed = _parse_tool_call(text)
        if parsed is None:
            rewards.append(0.0)
            continue

        name = parsed.get("name", "")
        args = parsed.get("arguments", None)

        if name in VALID_TOOLS and args is not None:
            rewards.append(1.0)
        elif name:
            # Parseable but unknown tool — partial credit
            rewards.append(0.5)
        else:
            rewards.append(0.0)

    return rewards


# Standalone test
if __name__ == "__main__":
    test_cases = [
        # Good: valid tool call
        ('<tool_call>{"name": "pubmed_search", "arguments": {"query": "BRCA1"}}</tool_call>', 1.0),
        # Good: OpenAI style
        ('{"name": "drug_lookup", "arguments": {"drug_name": "aspirin"}}', 1.0),
        # Partial: unknown tool name but valid JSON
        ('<tool_call>{"name": "unknown_tool", "arguments": {}}</tool_call>', 0.5),
        # Bad: no tool call
        ("The answer is 42.", 0.0),
        # Bad: malformed JSON
        ("<tool_call>{name: broken json}</tool_call>", 0.0),
    ]

    results = tool_call_format_reward([t[0] for t in test_cases])
    all_pass = True
    for (text, expected), got in zip(test_cases, results):
        status = "PASS" if got == expected else "FAIL"
        if got != expected:
            all_pass = False
        print(f"  [{status}] expected={expected} got={got}  text={text[:60]}")
    print("\nAll tests passed." if all_pass else "\nSome tests failed.")
