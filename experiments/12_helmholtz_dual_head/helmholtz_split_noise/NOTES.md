# Helmholtz Split Noise Schedule (Proposal 6)

## Idea

Two independent DDPM noise schedules for orthogonal Helmholtz subspaces:
- **Solenoidal** (div-free): standard schedule — preserves flow structure longer
- **Irrotational** (curl-free): 2× faster schedule — reaches pure noise sooner

Mathematically rigorous because the subspaces are orthogonal in L², so
independent DDPM processes compose to a valid joint process and the ELBO
factorises.

## Controlled Variables (vs helmholtz_baseline)

Everything identical except:
- `helmholtz_split_noise: true` (vs standard Gaussian noise)
- `irr_speed: 2.0` (irrotational schedule max_beta = 2× solenoidal)

## Training Log

*(To be updated as training proceeds)*

