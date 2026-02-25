# Mid-Size UNet Experiment Notes

## Motivation

| Model | Params | Channels | Attention | Dropout | Best Test Loss | Params/Sample |
|-------|--------|----------|-----------|---------|----------------|---------------|
| Big (MyUNet_Attn) | 23.2M | [64,128,256,256] | 16×32+8×16+bottleneck | None | 0.0093 @ ep57 | 2,523 |
| **Mid (MyUNet_Attn_Mid)** | **~13.5M** | **[48,96,192,192]** | **8×16+bottleneck** | **0.1** | **TBD** | **~1,470** |
| Slim (MyUNet_Attn_Slim) | 6.1M | [32,64,128,128] | bottleneck only | 0.1 | 0.016 @ ep28 | 666 |

- Slim was too small: peaked at 0.016, which is 72% worse than big model
- Big model overfits: no dropout, attention everywhere, 2523 params/sample
- Mid-size aims for the sweet spot: enough capacity to approach big model quality, with regularization to prevent overfitting

## Architecture Details

- Channels: [48, 96, 192, 192] — 1.5× slim, 0.75× big
- Attention at level 4 (8×16 = 128 positions) + bottleneck (4×8 = 32 positions)
- Dropout p=0.1 in every ResBlock
- batch_size=32 (vs 64 for slim, 16 for big) — fits in MPS memory
- All other training improvements: AdamW, cosine LR, EMA with warmup

## Run 1 — 2026-02-21

**Config**: batch_size=32, lr=0.001, cosine LR, warmup=10 epochs, AdamW, EMA warmup=2000 steps

*Training in progress...*
