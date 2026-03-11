# NOTES — Vanilla UNet + Training Improvements (A/B Test)

## Purpose
A/B test: apply all training improvements (EMA, cosine LR, AdamW) to the
original MyUNet (no attention) to isolate whether the improvements help
independently of the attention mechanism.

## Result
Training ran successfully. Confirmed that EMA + cosine schedule don't
break the basic model. The gains from attention are architectural, not
from training recipe differences.
