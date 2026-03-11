# helmholtz_split_matched_noise_continued — NOTES

## What's being tested
Warm restart of the matched noise model (best@ep328, test=0.0371).
The original 400-epoch run hit LR starvation — still improving at epoch 328
(LR=0.000045) when the cosine schedule decayed too aggressively.

## Setup
- `init_from_weights` loads best weights from original run (fresh optimizer)
- Fresh cosine schedule: 200 epochs, peak LR=0.0003 (slightly lower than
  original 0.0005 since we're fine-tuning), 5-epoch warmup
- Same loss, noise, architecture as original

## Key question
How much more can the model improve with additional LR budget?
Original eval: 1.038x V-CNN (10-sample avg at 0.5% coverage).

## Observations
