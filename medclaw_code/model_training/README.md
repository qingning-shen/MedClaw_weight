# model_training

LoRA SFT and GRPO RL fine-tuning pipeline for MedClaw.  
Trains Qwen3-4B-Instruct on ~8,000 examples to improve biomedical tool-calling.

## Requirements

- **GPU**: NVIDIA T4 (16 GB) or better — Kaggle free tier works
- **OS**: Linux (the cloud instance)
- **Python**: 3.10+

```bash
pip install -r requirements.txt
```

## Directory layout

```
model_training/
├── scripts/
│   ├── convert_openclaw.py   # Rule-based distillation from OpenClaw SKILL.md files
│   ├── download_datasets.py  # Download + sample xlam / SlimOrca / glaive from HuggingFace
│   ├── merge_datasets.py     # Merge, shuffle, split train/val; build RL prompt file
│   └── reward_fn.py          # GRPO reward: +1.0 for valid tool call, 0.0 otherwise
├── configs/
│   ├── sft_config.yaml       # ms-swift LoRA SFT hyperparameters
│   └── rl_config.yaml        # ms-swift GRPO hyperparameters
├── data/
│   ├── raw/                  # Downloaded JSONL files (gitignored)
│   ├── distill/              # Medical tool-call examples from OpenClaw (gitignored)
│   └── processed/            # Final train.jsonl, val.jsonl, rl_prompts.jsonl (gitignored)
├── output/                   # Training checkpoints (gitignored)
├── run_sft.sh                # Full pipeline: download → distill → merge → SFT
└── run_rl.sh                 # GRPO fine-tuning on top of SFT checkpoint
```

## Step-by-step on the cloud instance

### 1. Clone this repo and install dependencies

```bash
git clone <your-repo-url>
cd MedClaw/model_training
pip install -r requirements.txt
```

### 2. Clone OpenClaw-Medical-Skills

```bash
# In the parent directory of MedClaw/
git clone https://github.com/FreedomIntelligence/OpenClaw-Medical-Skills
# Use sparse checkout to avoid downloading large data files
cd OpenClaw-Medical-Skills
git sparse-checkout init --cone
git sparse-checkout set skills
```

### 3. Run SFT pipeline

```bash
cd MedClaw/model_training
export OPENCLAW_DIR=../../OpenClaw-Medical-Skills/skills
bash run_sft.sh
```

This runs four steps automatically:
1. Downloads xlam (4000), SlimOrca (1500), glaive (500) samples from HuggingFace
2. Distills 2000 medical tool-call examples from OpenClaw SKILL.md files
3. Merges all data → `data/processed/train.jsonl` (~7600 train, ~400 val)
4. Runs `swift sft` with LoRA rank-16 — checkpoint saved to `output/sft_medclaw/`

Estimated time: **~1.5 hours** on a single T4.

### 4. Run GRPO RL fine-tuning

```bash
bash run_rl.sh
```

Runs GRPO for 1 epoch on 400 medical query prompts using format-validity reward.  
Estimated time: **~30 minutes** on a T4.

### 5. Export the final model (merge LoRA weights)

```bash
swift export \
  --model output/rl_medclaw \
  --merge_lora true \
  --output_dir output/medclaw_final
```

Upload `output/medclaw_final/` to the serving instance (or HuggingFace Hub).

## Data format

All JSONL files use the ms-swift sharegpt format with tool calls:

```json
{
  "tools": "[{\"type\":\"function\",\"function\":{...}}]",
  "conversations": [
    {"role": "user", "content": "Find papers about BRCA1."},
    {"role": "assistant", "content": "", "tool_calls": [
      {"type": "function", "function": {"name": "pubmed_search", "arguments": "{\"query\": \"BRCA1\"}"}}
    ]}
  ]
}
```

General instruction samples (SlimOrca) omit the `tools` field and have no `tool_calls`.

## Reward function

`scripts/reward_fn.py` scores completions for GRPO:
- `1.0` — parseable tool call with a valid, known tool name
- `0.5` — parseable but unknown tool name
- `0.0` — no tool call or malformed JSON

Test it locally (no GPU needed):
```bash
python scripts/reward_fn.py
```

## Tips for Kaggle

- Enable GPU (T4 x2) in the notebook settings
- Set `bf16: true` in configs (T4 supports it)
- If OOM, reduce `per_device_train_batch_size` to 1 and double `gradient_accumulation_steps`
- Use `WANDB_DISABLED=true` to avoid wandb prompts
