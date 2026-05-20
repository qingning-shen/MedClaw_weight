#!/usr/bin/env python3
"""
Merge all SFT JSONL files into a single shuffled train/val split.
Also generates a separate RL prompt file (queries only) for GRPO training.

Usage:
    python merge_datasets.py --raw_dir ../data/raw \
                              --distill_dir ../data/distill \
                              --output_dir ../data/processed \
                              --val_ratio 0.05 \
                              --seed 42
"""

import json
import random
import argparse
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return samples


def extract_rl_prompts(samples: list[dict]) -> list[dict]:
    """Extract tool-calling samples as RL prompts (user turn + tools, no answer)."""
    prompts = []
    for s in samples:
        convs = s.get("conversations", [])
        if len(convs) < 2:
            continue
        # Keep only samples that have a tool_calls in the assistant turn
        has_tool_call = any(
            turn.get("role") == "assistant" and turn.get("tool_calls")
            for turn in convs
        )
        if not has_tool_call:
            continue
        # Keep only the user turn(s) as the prompt
        user_turns = [t for t in convs if t.get("role") in ("system", "user")]
        if not user_turns:
            continue
        prompt = {"conversations": user_turns}
        if "tools" in s:
            prompt["tools"] = s["tools"]
        prompts.append(prompt)
    return prompts


def main():
    parser = argparse.ArgumentParser(description="Merge SFT datasets and build RL prompt set.")
    parser.add_argument("--raw_dir", default="../data/raw")
    parser.add_argument("--distill_dir", default="../data/distill")
    parser.add_argument("--output_dir", default="../data/processed")
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--rl_prompts", type=int, default=400,
                        help="Number of RL prompt examples (sampled from medical distilled data)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load each source
    sources = {
        "xlam":     Path(args.raw_dir) / "xlam.jsonl",
        "slimorca": Path(args.raw_dir) / "slimorca.jsonl",
        "glaive":   Path(args.raw_dir) / "glaive.jsonl",
        "medical":  Path(args.distill_dir) / "medical_tool_calls.jsonl",
    }

    all_samples = []
    for name, path in sources.items():
        if not path.exists():
            print(f"  WARNING: {path} not found — skipping {name}")
            continue
        data = load_jsonl(path)
        print(f"  {name}: {len(data)} samples")
        all_samples.extend(data)

    if not all_samples:
        raise SystemExit("No data found. Run download_datasets.py and convert_openclaw.py first.")

    random.shuffle(all_samples)

    # Train / val split
    n_val = max(1, int(len(all_samples) * args.val_ratio))
    val_samples = all_samples[:n_val]
    train_samples = all_samples[n_val:]

    def write_jsonl(data: list[dict], path: Path):
        with open(path, "w", encoding="utf-8") as f:
            for s in data:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    write_jsonl(train_samples, out / "train.jsonl")
    write_jsonl(val_samples,   out / "val.jsonl")
    print(f"\nTrain: {len(train_samples)}  Val: {len(val_samples)}")
    print(f"Saved to {out}/train.jsonl and val.jsonl")

    # Build RL prompt set from medical distilled data only
    medical_path = Path(args.distill_dir) / "medical_tool_calls.jsonl"
    if medical_path.exists():
        medical_data = load_jsonl(medical_path)
        rl_prompts = extract_rl_prompts(medical_data)
        random.shuffle(rl_prompts)
        rl_prompts = rl_prompts[:args.rl_prompts]
        write_jsonl(rl_prompts, out / "rl_prompts.jsonl")
        print(f"RL prompts: {len(rl_prompts)} saved to {out}/rl_prompts.jsonl")
    else:
        print("Medical data not found — RL prompts file skipped.")

    print("\nDone. Next steps:")
    print("  SFT: swift sft --config configs/sft_config.yaml")
    print("  RL:  swift rlhf --config configs/rl_config.yaml")


if __name__ == "__main__":
    main()
