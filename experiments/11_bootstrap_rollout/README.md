# Bootstrap rollout fine-tuning

This group tests whether a Voronoi-forward DDPM can be adapted to multistep
diffusion by fine-tuning on its own rollout distribution.

## Research question

Can a model trained on noised-Voronoi inputs recover multistep performance if we
continue training it on self-generated intermediate states instead of only raw
Voronoi starts?

## Controlled variables

- Same base checkpoint: `experiments/10_topology_metrics/voronoi_topo_training/`
- Same unconditional `standard_attn` architecture
- Same topology-aware loss
- Same Voronoi-forward data generation and random sparse masks

## Varied variables

- Rollout depth mixture used during fine-tuning
- Fine-tuning LR / duration

## Experiments

`voronoi_multistep_bootstrap/`

- Start from the converged Voronoi-forward checkpoint
- Fine-tune on a mixture of rollout depths $k \in \{0,1,2\}$
- Goal: improve multistep S6-style inference without discarding strong single-step behavior

`voronoi_detached_stage3/`

- Start from the converged Voronoi-forward topology-aware weights
- Add a learned stage token for stages $S1,S2,S3$
- Train on a 3-stage detached rollout with known-value repaste between stages
- Apply topology-aware loss at every stage with stage-weighted aggregation
