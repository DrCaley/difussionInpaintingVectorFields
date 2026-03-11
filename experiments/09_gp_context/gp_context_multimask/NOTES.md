# GP-Context Multi-Mask Experiment Notes

## What's being tested
The v2 model was trained with a FIXED row-22 mask (~2.3% known). This v3
experiment retrains from scratch with diverse sparse Gaussian masks:
- 0.1% known (~4 pixels)
- 1% known (~41 pixels)
- 5% known (~207 pixels)
- 10% known (~414 pixels)

Each training sample randomly receives one of these 4 mask configurations
(with pre-computed GP posterior matching that mask). This should teach
the model to handle varied observation densities.

## Controlled variables
Same architecture (MyUNet_Attn, 8ch, 23.2M params), same hyperparameters
(lr=0.0003, cosine+warmup, EMA, grad clip=1.0), same data augmentation.

## Log

### [Date TBD] — GP Precompute
- Run `scripts/precompute_gp_multimask.py` on remote (96 cores)
- Expected: ~50 min for 4 masks × 11K samples
- Output: `data/rams_head/gp_multimask.pt` (~5.6 GB)

### [Date TBD] — Training Launch
- Fresh training, 1000 epoch budget (conservative ceiling — NOT a target)
- Watch for convergence, compare test loss plateau with v2 (~0.049)

### 2026-03-05: Training converged — best at epoch 126
- **Test loss plateaued; best checkpoint saved at epoch 126**
- Downloaded locally: `experiments/09_gp_context/gp_context_multimask/results/inpaint_gaussian_t250_best_checkpoint.pt` (354 MB)
- **STATUS: CONVERGED.** The 1000 epoch budget is a conservative ceiling;
  the model plateaued well before that. Do not treat this as "mid-training."
- Note: EMA weights have missing `time_embed_table.weight` buffer — must load
  with `model_state_dict` key (not EMA).

### 2026-03-05: Inference comparisons (10 samples, 0.5% known)

**Vanilla / RePaint / Resample** (`scripts/eval_gp_context_multimask.py`):
- GP baseline: mean MSE 0.000685
- Vanilla: 3.15× GP (0/10 wins)
- RePaint: 3.26× GP (0/10 wins)
- Resample: 1.51× GP (5/10 wins)

**S6 GP-warm / V-CNN** (`tmp_compare_6s_vcnn_gp.py`):
- S6 GP-warm: 0.996× GP mean ratio, median 0.883×, 7/10 wins vs GP
- V-CNN: 0.349× GP, 10/10 wins vs GP
- S6 beats V-CNN: 1/10
- V-CNN dominates at 0.5% sparse random observations
