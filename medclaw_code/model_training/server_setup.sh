#!/usr/bin/env bash
# One-shot setup for MedClaw training on a rented GPU server (RTX 4090).
# Run this once right after you SSH in.
#
# Usage (run on the GPU server):
#   bash server_setup.sh
#
# After setup, run the training pipeline:
#   cd ~/MedClaw/model_training
#   bash run_sft.sh
#   bash run_rl.sh
#   bash run_export.sh

set -e

PROJECT_DIR="${PROJECT_DIR:-$HOME/MedClaw}"
CONDA_ENV="medclaw-train"
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
PIP_HOST="pypi.tuna.tsinghua.edu.cn"
HF_MIRROR="https://hf-mirror.com"

echo "=== MedClaw Server Setup ==="
echo "Project dir : $PROJECT_DIR"
echo "Conda env   : $CONDA_ENV"
echo ""

# ── 1. Miniconda ─────────────────────────────────────────────────────────────
if ! command -v conda &>/dev/null && [ ! -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  echo "[1/6] Installing Miniconda..."
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
  conda init bash
else
  echo "[1/6] Miniconda already present — skipping install."
  CONDA_BASE="${CONDA_PREFIX:-$HOME/miniconda3}"
  # Try common locations
  for p in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda" "/root/miniconda3"; do
    if [ -f "$p/etc/profile.d/conda.sh" ]; then
      CONDA_BASE="$p"; break
    fi
  done
  source "$CONDA_BASE/etc/profile.d/conda.sh"
fi

# ── 2. Conda env ──────────────────────────────────────────────────────────────
if conda env list 2>/dev/null | grep -q "^$CONDA_ENV "; then
  echo "[2/6] Conda env '$CONDA_ENV' already exists — skipping."
else
  echo "[2/6] Creating conda env '$CONDA_ENV' (Python 3.11)..."
  conda create -n "$CONDA_ENV" python=3.11 -y
fi
# shellcheck disable=SC1091
conda activate "$CONDA_ENV"

# ── 3. Configure pip mirror ──────────────────────────────────────────────────
echo "[3/6] Configuring Tsinghua pip mirror..."
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << EOF
[global]
index-url = $PIP_MIRROR
trusted-host = $PIP_HOST
EOF

# ── 4. Install PyTorch with CUDA first, then ms-swift ────────────────────────
echo "[4/6] Installing PyTorch (CUDA) + ms-swift training stack..."
echo "      Detecting CUDA version..."

# Detect system CUDA version from nvidia-smi (driver-reported)
CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1 || echo "12.1")
CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
echo "      System CUDA: $CUDA_VER → using cu${CUDA_MAJOR}${CUDA_MINOR} wheel"

# Map to nearest PyTorch CUDA wheel (cu118, cu121, cu124)
if   [ "$CUDA_MAJOR" -lt 12 ]; then CU_TAG="cu118"
elif [ "$CUDA_MINOR" -ge 4 ];  then CU_TAG="cu124"
else                                 CU_TAG="cu121"
fi

# Install PyTorch via Aliyun mirror (faster in China)
pip install torch==2.6.0+cu124 torchvision==0.21.0 torchaudio==2.6.0+cu124 \
    -f "https://mirrors.aliyun.com/pytorch-wheels/${CU_TAG}" \
    -i "https://mirrors.aliyun.com/pypi/simple/" \
    --trusted-host "mirrors.aliyun.com"

# Verify CUDA is actually usable
python - << 'PYEOF'
import torch, sys
if not torch.cuda.is_available():
    print("WARNING: torch.cuda.is_available() is False — training will fail!")
    print("         Check driver/CUDA installation on this server.")
else:
    print(f"    PyTorch {torch.__version__}  |  GPU: {torch.cuda.get_device_name(0)}  |  VRAM: {torch.cuda.get_device_properties(0).total_memory//1024**3} GB")
PYEOF

# Install ms-swift and remaining deps via Tsinghua
pip install ms-swift -i "$PIP_MIRROR" --trusted-host "$PIP_HOST" -q
pip install -r "$PROJECT_DIR/model_training/requirements.txt" \
    -i "$PIP_MIRROR" --trusted-host "$PIP_HOST" -q \
    --no-deps 2>/dev/null || \
pip install -r "$PROJECT_DIR/model_training/requirements.txt" \
    -i "$PIP_MIRROR" --trusted-host "$PIP_HOST" -q
echo "      Dependencies installed."

# ── 5. Clone OpenClaw-Medical-Skills ─────────────────────────────────────────
OPENCLAW_DIR="$PROJECT_DIR/../OpenClaw-Medical-Skills"
echo "OpenClaw-Medical-Skills dir: $OPENCLAW_DIR"
if [ -d "$OPENCLAW_DIR/skills" ]; then
  echo "[5/6] OpenClaw already cloned — pulling latest..."
  git -C "$OPENCLAW_DIR" pull --ff-only 2>/dev/null || true
else
  echo "[5/6] Cloning OpenClaw-Medical-Skills..."
  # Try GitHub first; fall back to a proxy if blocked
  if git clone --depth=1 https://github.com/FreedomIntelligence/OpenClaw-Medical-Skills.git "$OPENCLAW_DIR" 2>/dev/null; then
    echo "      Cloned from GitHub."
  else
    echo "      GitHub blocked — trying https://ghproxy.cn proxy..."
    git clone --depth=1 "https://ghproxy.cn/https://github.com/FreedomIntelligence/OpenClaw-Medical-Skills.git" "$OPENCLAW_DIR"
  fi
fi
echo "      Skills directory: $OPENCLAW_DIR/skills"

# ── 6. Configure HuggingFace mirror ──────────────────────────────────────────
echo "[6/6] Configuring HuggingFace mirror ($HF_MIRROR)..."
if ! grep -q "HF_ENDPOINT" ~/.bashrc 2>/dev/null; then
  printf '\n# HuggingFace mirror for mainland China\nexport HF_ENDPOINT="%s"\n' "$HF_MIRROR" >> ~/.bashrc
fi
export HF_ENDPOINT="$HF_MIRROR"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "=== Setup complete ==="
echo ""
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
  && echo "" || true
echo "Next steps (copy-paste in order):"
echo ""
echo "  conda activate $CONDA_ENV"
echo "  cd $PROJECT_DIR/model_training"
echo "  screen -S medclaw          # keep alive if SSH drops"
echo "  bash run_sft.sh            # data pipeline + SFT  (~40 min)"
echo "  bash run_rl.sh             # GRPO RL               (~20 min)"
echo "  bash run_export.sh         # merge LoRA → final model"
echo ""
echo "Then on your LOCAL machine:"
echo "  bash model_training/fetch_model.sh <user>@<server_ip>"
