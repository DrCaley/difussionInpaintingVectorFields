#!/bin/bash
# Deploy project to vast.ai remote server
set -e

REMOTE="root@158.51.110.52"
SSH_OPTS="-p 28793 -o StrictHostKeyChecking=accept-new"
PROJECT_DIR="/Users/caleyjb/Library/Mobile Documents/com~apple~CloudDocs/JeffsStuff/PLU/Research/diffusionInpaintingVectorFields"
REMOTE_DIR="/root/project"

echo "=== Step 1: Transfer code ==="
rsync -az -e "ssh $SSH_OPTS" \
  --exclude='env/' \
  --exclude='__pycache__/' \
  --exclude='.DS_Store' \
  --exclude='*.pyc' \
  --exclude='ddpm/Trained_Models/' \
  --exclude='training_output/' \
  --exclude='plots/' \
  --exclude='paper/' \
  --exclude='garbage.txt/' \
  --exclude='.git/' \
  --exclude='noising_process/' \
  --exclude='tmp_*.py' \
  --exclude='*.ipynb' \
  "$PROJECT_DIR/" "$REMOTE:$REMOTE_DIR/"
echo "Code synced."

echo "=== Step 2: Install dependencies on remote ==="
ssh -T $SSH_OPTS $REMOTE "cd $REMOTE_DIR && pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && pip install -q pyyaml tqdm matplotlib scipy 2>&1 | tail -5"
echo "Dependencies installed."

echo "=== Step 3: Verify remote setup ==="
ssh -T $SSH_OPTS $REMOTE "cd $REMOTE_DIR && python3 -c \"
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
import yaml, tqdm, scipy
print('All imports OK')
\""

echo "=== Step 4: Run GP precompute ==="
ssh -T $SSH_OPTS $REMOTE "cd $REMOTE_DIR && PYTHONPATH=. python3 scripts/precompute_gp.py 2>&1"
echo "GP precompute done."

echo "=== Step 5: Smoke test (3 epochs) ==="
ssh -T $SSH_OPTS $REMOTE "cd $REMOTE_DIR && PYTHONPATH=. python3 experiments/run_experiment.py --smoke experiments/09_gp_context/gp_context_eps/config.yaml 2>&1 | tail -30"
echo "Smoke test done."

echo ""
echo "=== READY TO TRAIN ==="
echo "Run full training with:"
echo "  ssh $SSH_OPTS $REMOTE 'cd $REMOTE_DIR && nohup PYTHONPATH=. python3 experiments/run_experiment.py experiments/09_gp_context/gp_context_eps/config.yaml > train.log 2>&1 &'"
