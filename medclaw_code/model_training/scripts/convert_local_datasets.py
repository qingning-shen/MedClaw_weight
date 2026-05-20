#!/usr/bin/env python3
"""
Convert pre-downloaded JSON/JSONL datasets to ms-swift sharegpt format.
Run this after manually downloading datasets from HuggingFace.

Usage:
    python convert_local_datasets.py --output_dir data/raw
"""

import json
import random
import argparse
import re
from pathlib import Path


def convert_xlam(row: dict) -> dict | None:
    """Convert a single xlam-function-calling-60k row."""
    try:
        tools_raw = json.loads(row["tools"])
        answers_raw = json.loads(row["answers"])
    except (json.JSONDecodeError, TypeError, KeyError):
        return None

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
    """Convert a glaive-function-calling-v2 row."""
    try:
        system_text = row.get("system", "")
        chat_text = row.get("chat", "")
    except AttributeError:
        return None

    # Extract function definitions from system prompt
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
    parts = re.split(r"\n(?=USER:|ASSISTANT:|FUNCTION RESPONSE:)", chat_text.strip())
    conversations = []
    for part in parts:
        if part.startswith("USER:"):
            conversations.append({"role": "user", "content": part[5:].strip()})
        elif part.startswith("ASSISTANT:"):
            content = part[10:].strip()
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
                    break
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


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_json(path: Path) -> list[dict]:
    """Load a JSON file (array or object with data field)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # Some datasets have {"data": [...]} structure
        if "data" in data:
            return data["data"]
        # Try to find the data field
        for key in ["train", "test", "samples", "examples"]:
            if key in data:
                return data[key]
    return data


def convert_and_save(input_path: Path, output_path: Path, converter, n: int, seed: int):
    """Load, convert, and save dataset."""
    print(f"\nLoading {input_path}...")

    # Detect format
    if input_path.suffix == ".jsonl":
        data = load_jsonl(input_path)
    else:
        data = load_json(input_path)

    print(f"  Total rows: {len(data)}")

    # Shuffle
    random.seed(seed)
    random.shuffle(data)

    # Convert
    samples = []
    skipped = 0
    for row in data:
        if len(samples) >= n:
            break
        converted = converter(row)
        if converted is not None:
            samples.append(converted)
        else:
            skipped += 1

    print(f"  Converted: {len(samples)}, skipped: {skipped}")

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert locally downloaded datasets.")
    parser.add_argument("--input_dir", default="data/raw", help="Directory with downloaded files")
    parser.add_argument("--output_dir", default="data/raw", help="Output directory for converted files")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--xlam_n", type=int, default=4000, help="Max xlam samples")
    parser.add_argument("--slimorca_n", type=int, default=1500, help="Max slimorca samples")
    parser.add_argument("--glaive_n", type=int, default=500, help="Max glaive samples")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find and convert xlam
    xlam_files = list(input_dir.glob("xlam*.json*")) + list(input_dir.glob("*xlam*.json*"))
    if xlam_files:
        convert_and_save(xlam_files[-1], output_dir / "xlam.jsonl", convert_xlam, args.xlam_n, args.seed)
    else:
        print("\n[xlam] No file found. Expected: xlam.json or xlam.jsonl in input_dir")

    # Find and convert slimorca
    slimorca_files = list(input_dir.glob("*oo*.json*")) + list(input_dir.glob("*slimorca*.json*"))
    if slimorca_files:
        convert_and_save(slimorca_files[0], output_dir / "slimorca.jsonl", convert_slimorca, args.slimorca_n, args.seed)
    else:
        print("\n[slimorca] No file found. Expected: slimorca.json or slimorca.jsonl in input_dir")

    # Find and convert glaive
    glaive_files = list(input_dir.glob("glaive*.json*")) + list(input_dir.glob("*glaive*.json*"))
    if glaive_files:
        convert_and_save(glaive_files[-1], output_dir / "glaive.jsonl", convert_glaive_v2, args.glaive_n, args.seed)
    else:
        print("\n[glaive] No file found. Expected: glaive.json or glaive.jsonl in input_dir")

    print("\n=== Conversion complete ===")
    print("Next: python scripts/merge_datasets.py")


if __name__ == "__main__":
    main()
