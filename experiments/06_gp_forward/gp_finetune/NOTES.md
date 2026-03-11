# GP-Forward Fine-tune Experiment Notes

## 2026-02-28 — Initial setup
- Created experiment to address training-inference distribution mismatch
- The FiLM+Attn model (03/film_attn_divfree) was trained on noise(GT) but
  at inference sees noise(GP), achieving only 1-2% MSE improvement over GP
- GP-forward training noises the GP posterior instead of GT during training
- Fine-tuning from epoch-57 best checkpoint (best_test_loss=0.00579)
- Using lower LR (0.0001 vs 0.0003) since we're fine-tuning
- 200 epochs should be sufficient — model already knows ocean flow structure
