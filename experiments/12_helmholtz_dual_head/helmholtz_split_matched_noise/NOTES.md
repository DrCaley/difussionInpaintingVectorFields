# Helmholtz Split Decoder + Matched Noise

## Hypothesis
Operator-matched Helmholtz noise — where ε = curl_fwd(ψ_noise) + grad_cd(φ_noise)
using the same discrete operators as the UNet's ψ and φ heads — should improve
decomposition quality compared to standard Gaussian noise because:

1. Both heads see noise that matches their operator's range/null-space structure
2. The ψ head denoises purely solenoidal noise components, φ head denoises purely
   irrotational noise components — clean separation at all timesteps
3. No cross-contamination between subspaces in the noise

## Controlled Variables
- Architecture: helmholtz_split (Option C split decoder) — same as helmholtz_split_decoder
- Loss: helmholtz_supervised (lambda_decomp=0.1, lambda_orth=0.01)
- All hyperparameters identical to helmholtz_split_decoder

## Varied Variable
- noise_function: helmholtz_matched (vs gaussian in helmholtz_split_decoder)

## Log
