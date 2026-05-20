#!/usr/bin/env python3
"""
Download and sample SFT datasets from HuggingFace, converting each to the
ms-swift sharegpt tool-calling format.

Datasets:
  - Salesforce/xlam-function-calling-60k  → 4000 samples  (general tool calling)
  - Open-Orca/SlimOrca                    → 1500 samples  (general instruction)
  - glaiveai/glaive-function-calling-v2   →  500 samples  (multi-turn tool calling)

Usage:
    pip install datasets
    python download_datasets.py --output_dir ../data/raw --seed 42
"""

import json
import os
import random
import argparse
from pathlib import Path

# Use HF mirror if set (speeds up downloads in China)
hf_endpoint = os.environ.get("HF_ENDPOINT", "")
if hf_endpoint:
    os.environ["HF_ENDPOINT"] = hf_endpoint
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # Enable fast transfer

try:
    from datasets import load_dataset
except ImportError:
    raise SystemExit("Run: pip install datasets")


# ---------------------------------------------------------------------------
# Converters — each returns a dict in ms-swift sharegpt format, or None to skip
# ---------------------------------------------------------------------------

def convert_xlam(row: dict) -> dict | None:
    """Convert a single xlam-function-calling-60k row."""
    try:
        tools_raw = json.loads(row["tools"])
        answers_raw = json.loads(row["answers"])
    except (json.JSONDecodeError, TypeError, KeyError):
        return None

    # xlam tools are already in {"name": ..., "description": ..., "parameters": ...} form
    tools = [{"type": "function", "function": t} for t in tools_raw]

    tool_calls = []
    for ans in answers_raw:
        name = ans.get("name") or ans.get("tool_name")
        arguments = ans.get("arguments") or ans.get("parameters") or {}
        if not name:
            return None
        tool_calls.append({
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments, ensure_ascii=False)
            }
        })

    if not tool_calls:
        return None

    return {
        "tools": json.dumps(tools),
        "conversations": [
            {"role": "user", "content": row["query"]},
            {"role": "assistant", "content": "", "tool_calls": tool_calls}
        ]
    }


def convert_slimorca(row: dict) -> dict | None:
    """Convert a SlimOrca row (conversations list with from/value fields)."""
    try:
        convs = row["conversations"]
    except KeyError:
        return None

    role_map = {"system": "system", "human": "user", "gpt": "assistant"}
    converted = []
    for turn in convs:
        role = role_map.get(turn.get("from", ""), None)
        value = turn.get("value", "").strip()
        if role is None or not value:
            return None
        converted.append({"role": role, "content": value})

    if len(converted) < 2:
        return None

    return {"conversations": converted}


def convert_glaive_v2(row: dict) -> dict | None:
    """
    Convert a glaive-function-calling-v2 row.
    Format: system has function defs; chat has USER:/ASSISTANT:/FUNCTION RESPONSE: turns.
    We capture the first tool-call turn only (enough for multi-turn pattern exposure).
    """
    try:
        system_text = row.get("system", "")
        chat_text = row.get("chat", "")
    except AttributeError:
        return None

    # Extract function definitions from system prompt
    # System looks like: "SYSTEM: You are a helpful assistant with access to ...\n\n<func_json>"
    func_json_match = re.search(r"\[.*\]", system_text, re.DOTALL)
    tools_str = None
    if func_json_match:
        try:
            tools_raw = json.loads(func_json_match.group())
            tools = [{"type": "function", "function": t} if "type" not in t else t for t in tools_raw]
            tools_str = json.dumps(tools)
        except (json.JSONDecodeError, TypeError):
            pass

    # Split chat into turns
    # Pattern: USER: ... \nASSISTANT: ... \nFUNCTION RESPONSE: ... \nASSISTANT: ...
    parts = re.split(r"\n(?=USER:|ASSISTANT:|FUNCTION RESPONSE:)", chat_text.strip())
    conversations = []
    for part in parts:
        if part.startswith("USER:"):
            conversations.append({"role": "user", "content": part[5:].strip()})
        elif part.startswith("ASSISTANT:"):
            content = part[10:].strip()
            # Check for <functioncall> tag
            fc_match = re.search(r"<functioncall>(.*?)</functioncall>", content, re.DOTALL)
            if fc_match:
                try:
                    fc_data = json.loads(fc_match.group(1).strip())
                    conversations.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "type": "function",
                            "function": {
                                "name": fc_data.get("name", ""),
                                "arguments": json.dumps(fc_data.get("arguments", {}))
                            }
                        }]
                    })
                    break  # keep only up to first tool call for simplicity
                except (json.JSONDecodeError, AttributeError):
                    conversations.append({"role": "assistant", "content": content})
            else:
                conversations.append({"role": "assistant", "content": content})
        elif part.startswith("FUNCTION RESPONSE:"):
            conversations.append({"role": "tool", "content": part[18:].strip()})

    if len(conversations) < 2:
        return None

    result = {"conversations": conversations}
    if tools_str:
        result["tools"] = tools_str
    return result


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_and_convert(dataset_id: str, split: str, converter, n: int, seed: int, output_path: Path):
    print(f"\nLoading {dataset_id} ({split})...")
    ds = load_dataset(dataset_id, split=split, trust_remote_code=True)
    print(f"  Total rows: {len(ds)}")

    # Shuffle and attempt conversion until we have n valid samples
    indices = list(range(len(ds)))
    random.seed(seed)
    random.shuffle(indices)

    samples = []
    skipped = 0
    for idx in indices:
        if len(samples) >= n:
            break
        row = ds[idx]
        converted = converter(row)
        if converted is not None:
            samples.append(converted)
        else:
            skipped += 1

    print(f"  Converted: {len(samples)}, skipped: {skipped}")

    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

import re  # used in convert_glaive_v2, imported here to keep top-level clean


def main():
    parser = argparse.ArgumentParser(description="Download and convert SFT datasets.")
    parser.add_argument("--output_dir", default="../data/raw")
    parser.add_argument("--seed", type=int, default=42)
    # Per-dataset sample counts
    parser.add_argument("--xlam_n", type=int, default=4000)
    parser.add_argument("--slimorca_n", type=int, default=1500)
    parser.add_argument("--glaive_n", type=int, default=500)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    download_and_convert(
        "Salesforce/xlam-function-calling-60k", "train",
        convert_xlam, args.xlam_n, args.seed,
        out / "xlam_function_calling_60k.json"
    )

    download_and_convert(
        "Open-Orca/SlimOrca", "train",
        convert_slimorca, args.slimorca_n, args.seed,
        out / "oo-labeled_correct.gpt4.sharegpt.jsonl"
    )

    download_and_convert(
        "glaiveai/glaive-function-calling-v2", "train",
        convert_glaive_v2, args.glaive_n, args.seed,
        out / "glaive-function-calling-v2.json"
    )

    print("\nAll datasets downloaded. Run merge_datasets.py next.")


if __name__ == "__main__":
    main()
