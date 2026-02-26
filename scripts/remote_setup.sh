#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# Remote setup script for Vast.ai GPU training
# Run this AFTER cloning the repo and uploading data files.
# ─────────────────────────────────────────────────────────────────
set -euo pipefail

# Auto-detect repo directory: use the directory containing this script's parent
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="experiments/04_spatiotemporal/st_t13_gaussian/config_cuda.yaml"

echo "═══════════════════════════════════════════════════════════"
echo "  Spatiotemporal DDPM — Remote GPU Setup"
echo "═══════════════════════════════════════════════════════════"

# ── 1. Check CUDA ─────────────────────────────────────────────────
echo ""
echo "[1/5] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── 2. Install dependencies ──────────────────────────────────────
echo "[2/5] Installing Python packages..."
cd "$REPO_DIR"

pip install --quiet --upgrade pip

# Install PyTorch with CUDA if not already present (e.g. vastai/base-image)
python -c "import torch" 2>/dev/null || {
    echo "  PyTorch not found — installing torch + torchvision with CUDA 12.4..."
    pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu124
}

# Install remaining dependencies
pip install --quiet numpy scipy matplotlib tqdm PyYAML Pillow gpytorch

echo "  PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available:  $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  GPU name:        $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")')"

# ── 3. Verify data files ─────────────────────────────────────────
echo ""
echo "[3/5] Checking data files..."
MISSING=0

if [ ! -f "$REPO_DIR/data.pickle" ]; then
    echo "  ✗ MISSING: data.pickle (expected at repo root)"
    MISSING=1
fi

if [ ! -f "$REPO_DIR/data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2.mat" ]; then
    echo "  ✗ MISSING: data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2.mat"
    MISSING=1
fi

if [ ! -f "$REPO_DIR/data/rams_head/boundaries.yaml" ]; then
    echo "  ✗ MISSING: data/rams_head/boundaries.yaml"
    MISSING=1
fi

CKPT="experiments/02_inpaint_algorithm/repaint_gaussian_attn/results/inpaint_gaussian_t250_best_checkpoint.pt"
if [ ! -f "$REPO_DIR/$CKPT" ]; then
    echo "  ✗ MISSING: $CKPT (pretrained spatial checkpoint)"
    MISSING=1
fi

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "  Upload the missing files before training. See VAST_AI_GUIDE.md."
    echo "  Setup continues so you can upload and run later."
else
    echo "  ✓ All data files present."
fi

# ── 4. Dry run ────────────────────────────────────────────────────
echo ""
echo "[4/5] Dry-run config validation..."
cd "$REPO_DIR"
PYTHONPATH=. python experiments/run_experiment.py --dry-run "$CONFIG" 2>&1 | tail -5

# ── 5. Print launch command ──────────────────────────────────────
echo ""
echo "[5/5] Ready to train!"
echo ""
echo "  Launch training with:"
echo "    cd $REPO_DIR"
echo "    PYTHONPATH=. python experiments/run_experiment.py $CONFIG"
echo ""
echo "  Or run in background (keeps going if SSH disconnects):"
echo "    cd $REPO_DIR"
echo "    nohup bash -c 'PYTHONPATH=. python experiments/run_experiment.py $CONFIG' > training.log 2>&1 &"
echo "    tail -f training.log"
echo ""
echo "  To try a larger batch size (if 16GB allows):"
echo "    Edit config_cuda.yaml: batch_size: 6, gradient_accumulation_steps: 1"
echo ""
echo "═══════════════════════════════════════════════════════════"
