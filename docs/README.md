# How to Train and Run Models

This is the quick-start guide. For the full config reference, see
[TRAINING_AND_INFERENCE.md](TRAINING_AND_INFERENCE.md).

---

## Training a Model

### 1. Write a small config file

You only need to specify what's *different* from the defaults. Put it in
an experiment folder:

```yaml
# experiments/12_helmholtz_dual_head/my_experiment/config.yaml
model_name: my_experiment
unet_type: film
prediction_target: x0
```

That's it — three lines. Everything else (learning rate, batch size, noise
schedule, etc.) comes from the base template automatically.

### 2. Check that it makes sense

```bash
PYTHONPATH=. python experiments/run_experiment.py --dry-run \
    experiments/12_helmholtz_dual_head/my_experiment/config.yaml
```

This prints the full resolved config and checks that your components are
compatible (e.g., div-free noise needs a unified standardizer). Nothing
gets trained yet.

### 3. Do a quick smoke test

```bash
PYTHONPATH=. python experiments/run_experiment.py --smoke \
    experiments/12_helmholtz_dual_head/my_experiment/config.yaml
```

Runs 3 epochs to make sure everything wires up without errors.

### 4. Train for real

```bash
PYTHONPATH=. python experiments/run_experiment.py \
    experiments/12_helmholtz_dual_head/my_experiment/config.yaml
```

Training output goes to `experiments/.../my_experiment/results/`:
- `resolved_config.yaml` — the full config that was used
- `training_log_*.csv` — loss values per epoch
- `*_best_weights.pt` — best model weights (for inference)
- `*_best_checkpoint.pt` — full checkpoint (for resuming)

### 5. Resume if interrupted

Add two lines to your config:

```yaml
retrain_mode: true
model_to_retrain: experiments/.../results/inpaint_*_best_checkpoint.pt
```

Then run the same training command again. It picks up where it left off.

---

## Running Inference

### Figure out what kind of model you have

Look at the `resolved_config.yaml` that was saved alongside your weights.
The two things that matter most are:

- **`prediction_target`** — did the model learn to predict `x0` (the clean
  image) or `eps` (the noise)?
- **`unet_type`** — what architecture was used?

### Pick the right inference algorithm

**If the model predicts x0** (most of our models):
→ Use `x0_full_reverse_inpaint()`

**If the model predicts eps with a conditioned UNet** (concat or film):
→ Use `mask_aware_inpaint()`

**If the model predicts eps with an unconditional UNet** (standard):
→ Use `repaint_standard()`

Using the wrong one will give you garbage. When in doubt, check the
resolved config.

### Run an eval script

Most eval scripts follow the same pattern:

```bash
PYTHONPATH=. python scripts/eval_subframes_baseline.py \
    --weights path/to/model_best_weights.pt \
    --coverage 0.5 1.0 2.0 5.0 \
    --n-samples 5
```

**What the arguments mean:**

- `--weights` — path to your trained model
- `--coverage` — what percentage of the domain is observed (e.g., `2.0`
  means 2% of grid cells have measurements — the rest is inpainted)
- `--n-samples` — how many test cases to evaluate
- `--t-starts` — where in the diffusion chain to start denoising (lower =
  faster but less flexible; `25` or `50` is usually fine)
- `--n-ensemble` — average this many independent runs for a smoother result
  (`1` for quick tests, `10` for paper numbers)

Results are saved as `.pt` files first, then plots are generated from those.
This way if plotting crashes you don't lose the expensive GPU work.

---

## Common Configurations

**Simple FiLM model (our default):**
```yaml
model_name: my_film_model
unet_type: film
prediction_target: x0
```

**Helmholtz dual-head with multi-resolution conditioning:**
```yaml
model_name: my_helmholtz_model
unet_type: helmholtz_split_film_multires
prediction_target: x0
use_bathymetry: true
```

**With topology penalties (vorticity + divergence loss):**
```yaml
model_name: my_topo_model
loss_function: helmholtz_supervised
helmholtz_loss:
  lambda_decomp: 0.1
  lambda_orth: 0.01
topology:
  lambda_vort: 0.01
  lambda_div: 0.005
```

**Gaussian noise baseline (for comparison):**
```yaml
model_name: gaussian_baseline
noise_function: gaussian
unet_type: concat
prediction_target: eps
```

---

## Things to Know

- **High mask coverage is the point.** We're testing 70–95%+ missing data.
  That's not a bug — it's the whole research question.

- **Models converge well before 1000 epochs.** If the loss has flattened,
  the model is done. Don't worry about "only 200/1000 epochs."

- **`mask_xt` must match between training and inference.** FiLM models are
  always trained with `mask_xt: true`. If you use a different setting at
  inference time, the results will be wrong.

- **V-CNN is the baseline.** Every evaluation compares against the
  Voronoi-CNN (Fukami et al. 2021). Report results as ratios like
  "0.85× V-CNN MSE."

- **Noise function determines standardization automatically.** Div-free
  noise needs unified z-score normalization. Gaussian noise uses
  per-component. The launcher handles this — you don't need to set it
  manually.
