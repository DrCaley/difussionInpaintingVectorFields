"""Exponential Moving Average (EMA) for model parameters.

Standard DDPM practice (Ho et al. 2020): maintain a shadow copy of model
weights as an exponentially-weighted running average.  EMA weights are
smoother and typically produce better inference quality than raw optimizer
weights.

Usage in training loop:
    ema = EMA(model, decay=0.9999)
    for epoch in ...:
        ...
        optimizer.step()
        ema.update()       # update shadow weights
    ema.apply()            # copy EMA weights → model for inference/eval
    ema.restore()          # restore original weights to resume training
"""

import copy
import torch


class EMA:
    """Exponential Moving Average of model parameters.

    Args:
        model: nn.Module whose parameters to track.
        decay: EMA decay rate.  0.9999 is the DDPM standard.
               Higher = smoother but slower to adapt.
        warmup_steps: Linearly ramp up decay from 0 to target over this
                      many update() calls.  Prevents the EMA from being
                      dominated by random initial weights.
    """

    def __init__(self, model, decay=0.9999, warmup_steps=0):
        self.model = model
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step_count = 0

        # Shadow copy of parameters (detached, on same device)
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def _get_decay(self):
        """Get effective decay, linearly warming up if configured."""
        if self.warmup_steps <= 0:
            return self.decay
        # Ramp from 0 → decay over warmup_steps
        return min(self.decay, self.step_count / self.warmup_steps * self.decay)

    @torch.no_grad()
    def update(self):
        """Update shadow parameters with current model parameters."""
        decay = self._get_decay()
        self.step_count += 1
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].lerp_(param.data, 1.0 - decay)

    def apply(self):
        """Copy EMA (shadow) weights → model.  Call before eval/inference."""
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original (optimizer) weights → model.  Call after eval."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        """Serialize EMA state for checkpointing."""
        return {
            "shadow": {k: v.cpu() for k, v in self.shadow.items()},
            "step_count": self.step_count,
            "decay": self.decay,
        }

    def load_state_dict(self, state):
        """Restore EMA state from checkpoint."""
        self.step_count = state.get("step_count", 0)
        self.decay = state.get("decay", self.decay)
        for name, tensor in state.get("shadow", {}).items():
            if name in self.shadow:
                self.shadow[name].copy_(tensor.to(self.shadow[name].device))
