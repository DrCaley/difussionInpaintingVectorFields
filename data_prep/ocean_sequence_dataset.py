"""Temporal sequence dataset for spatiotemporal diffusion training.

Extracts sliding-window sequences of T consecutive hourly frames from
the ocean velocity dataset, respecting chunk boundaries so that no
sequence spans a temporal gap.

Data layout
───────────
The training data in ``data.pickle`` is shape (94, 44, 2, N) where N
is the concatenation of 70-timestep chunks (from ``spliting_data_sets.py``).
Chunk boundaries occur every 70 timesteps; consecutive chunks are
separated by 60-hour gaps in the original timeseries.

A valid sequence of length T must lie within a single chunk:
    chunk_start + offset  …  chunk_start + offset + T - 1
    where 0 ≤ offset ≤ chunk_size - T.

With chunk_size=70 and T=13 there are 58 valid offsets per chunk.
For 131 chunks → 131 × 58 = 7,598 training sequences.

Output format
─────────────
Each ``__getitem__`` returns ``(x0, t, noise)`` where:
    x0    : (T*C, H, W) = (26, 64, 128) — T frames stacked channel-wise
    t     : int — diffusion timestep (shared across all T frames)
    noise : (T*C, H, W) — noise from the active NoiseStrategy

The T*C channel layout is [u₀, v₀, u₁, v₁, …, u_{T-1}, v_{T-1}].
This is directly compatible with ``MyUNet_ST`` and ``GaussianDDPM``.
"""

import torch
from torch import Tensor
from torch.utils.data import Dataset
from typing import Optional


class OceanSequenceDataset(Dataset):
    """Sliding-window sequences of T consecutive velocity frames.

    Args:
        data_tensor: Raw data tensor of shape (94, 44, 2, N).
        n_steps: Number of diffusion timesteps (for random t sampling).
        noise_strategy: NoiseStrategy instance for generating ε.
        transform: Composed transform (resize + standardize), same as
            used by ``OceanImageDataset``.
        T: Number of consecutive frames per sequence (default 13).
        chunk_size: Number of consecutive timesteps per data chunk
            (default 70, matching ``spliting_data_sets.py``).
        boundaries: Path to boundaries YAML (unused here, kept for API
            consistency with ``OceanImageDataset``).
    """

    def __init__(
        self,
        data_tensor: Tensor,
        n_steps: int,
        noise_strategy,
        transform=None,
        T: int = 13,
        chunk_size: int = 70,
        boundaries: Optional[str] = None,
    ):
        assert data_tensor.ndim == 4 and data_tensor.shape[2] == 2, (
            f"Expected shape (94, 44, 2, N), got {data_tensor.shape}"
        )
        self.T = T
        self.n_steps = n_steps
        self.noise_strategy = noise_strategy
        self.transform = transform
        self.raw_tensor = data_tensor

        N = data_tensor.shape[3]
        n_chunks = N // chunk_size

        # ── Build valid sequence start indices ───────────────────────
        self.sequences = []
        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * chunk_size
            for offset in range(chunk_size - T + 1):
                self.sequences.append(chunk_start + offset)

        # Handle possible partial last chunk
        remainder = N - n_chunks * chunk_size
        if remainder >= T:
            chunk_start = n_chunks * chunk_size
            for offset in range(remainder - T + 1):
                self.sequences.append(chunk_start + offset)

        # ── Pre-process all individual frames ────────────────────────
        # Cache transformed frames in a single contiguous tensor for
        # fast __getitem__ slicing.
        sample = self._load_frame(0)
        C, H, W = sample.shape  # (2, 64, 128) after transform
        self.C, self.H, self.W = C, H, W

        self.all_frames = torch.zeros(N, C, H, W)
        for n in range(N):
            self.all_frames[n] = self._load_frame(n)

        print(
            f"[OceanSequenceDataset] {len(self.sequences)} sequences of T={T} "
            f"from {n_chunks} chunks of {chunk_size} ({N} total frames)"
        )

    def _load_frame(self, n: int) -> Tensor:
        """Load, NaN-handle, and transform a single timestep."""
        u = self.raw_tensor[..., n][..., 0].T  # (44, 94)
        v = self.raw_tensor[..., n][..., 1].T  # (44, 94)

        mask = (~(u.isnan() | v.isnan())).float()
        u = torch.nan_to_num(u, nan=0.0)
        v = torch.nan_to_num(v, nan=0.0)

        frame = torch.stack((u, v, mask), dim=0)  # (3, 44, 94)

        if self.transform:
            frame = self.transform(frame)  # (2, 64, 128)

        return frame

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx):
        start = self.sequences[idx]

        # Stack T frames channel-wise: (T, C, H, W) → (T*C, H, W)
        seq = self.all_frames[start : start + self.T]  # (T, C, H, W)
        x0 = seq.reshape(self.T * self.C, self.H, self.W)  # (T*C, H, W)

        t = torch.randint(0, self.n_steps, (1,)).item()
        noise = self.noise_strategy(
            x0.unsqueeze(0), torch.tensor([t])
        ).squeeze(0)

        return x0, t, noise
