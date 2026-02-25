"""Random mask generation for Palette-style mask-aware training.

Generates diverse masks during training so the model learns to inpaint
arbitrary missing regions.  Mask convention: 1 = missing, 0 = known.

Each generator produces a mask where a SHAPE represents the KNOWN region
(like a robot path, sensor track, or observation window), and everything
else is MISSING.  This matches the test scenario where ~80% of pixels
are unobserved and the model must inpaint them.

Mask types (selected randomly each call):
  - Random-walk path:  brush walk = known, rest missing    (~60-90% missing)
  - Straight-line:     1-3 known lines, rest missing       (~90-99% missing)
  - Sparse Gaussian:   sparse known points, rest missing   (~90-99.99% missing)
  - Robot BFS path:    greedy walk = known, rest missing   (~60-90% missing)
  - No mask / all-known (unconditional denoising)          (0% missing)

Probabilities are tuned so the model sees a wide variety of coverage
levels, with emphasis on high-missing masks matching the real test case.
"""

import random
import math
import torch
import numpy as np


# ── probability table ────────────────────────────────────────────────
#   type                prob    % MISSING (what model must inpaint)
#   no mask             5%      0%
#   random walk path    15%     ~60-90%
#   straight line       30%     ~90-99%
#   sparse gaussian     20%     ~90-99.99%
#   robot-style BFS     30%     ~60-90%  (matches test scenario)

_THRESHOLDS = [
    0.05,   # no mask
    0.20,   # random walk path (known trail)
    0.50,   # straight line (known tracks)
    0.70,   # sparse gaussian (known points)
    1.00,   # robot-style BFS walk (known explored area)
]


def generate_training_mask(h: int, w: int, land_mask: torch.Tensor = None) -> torch.Tensor:
    """Generate a random binary mask for training.

    Each mask type generates the KNOWN region as a shape (path, rectangle,
    points, etc.) and marks everything else as MISSING.  This matches the
    test scenario where only a small explored region is known.

    Args:
        h, w: spatial dimensions (64, 128 after resize)
        land_mask: (1, H, W) or (H, W) binary tensor, 1 = valid ocean pixel.
                   If provided, mask is intersected with land_mask so we never
                   ask the model to inpaint land pixels.

    Returns:
        mask: (1, H, W) float tensor.  1 = missing, 0 = known.
    """
    r = random.random()

    if r < _THRESHOLDS[0]:
        mask = torch.zeros(1, h, w)
    elif r < _THRESHOLDS[1]:
        mask = _random_walk_mask(h, w)
    elif r < _THRESHOLDS[2]:
        mask = _straight_line_mask(h, w)
    elif r < _THRESHOLDS[3]:
        mask = _sparse_gaussian_mask(h, w)
    else:
        mask = _robot_bfs_mask(h, w, land_mask)

    # Intersect with land mask if provided
    if land_mask is not None:
        lm = land_mask.view(1, h, w) if land_mask.dim() == 2 else land_mask[:1]
        mask = mask * lm  # only mark ocean pixels as missing

    return mask


# ── mask generators ──────────────────────────────────────────────────
# All generators return masks where 1=missing, 0=known.
# The "shape" (rectangle/path/line/points) is the KNOWN region.

def _rectangle_mask(h: int, w: int) -> torch.Tensor:
    """1-3 small known rectangle windows; rest is missing (~85-98% missing).

    Known region is 2-15% of the image — small observation windows.
    """
    mask = torch.ones(1, h, w)  # start all-missing
    n_rects = random.randint(1, 3)
    for _ in range(n_rects):
        rh = random.randint(h // 16, h // 4)
        rw = random.randint(w // 16, w // 4)
        y0 = random.randint(0, h - rh)
        x0 = random.randint(0, w - rw)
        mask[0, y0:y0 + rh, x0:x0 + rw] = 0.0  # known window
    return mask


def _random_walk_mask(h: int, w: int, brush_size: int = 3) -> torch.Tensor:
    """Random walk with a brush = known path; rest is missing (~60-90% missing).

    The walk represents an explored trail (like a robot path).
    Only 5-18% of pixels are known (the path), 82-95% are missing.
    """
    known = np.zeros((h, w), dtype=np.float32)
    target_known = int(h * w * random.uniform(0.05, 0.18))
    y, x = random.randint(0, h - 1), random.randint(0, w - 1)
    filled = 0
    brush = random.randint(max(1, brush_size - 1), brush_size + 2)

    while filled < target_known:
        y0 = max(0, y - brush // 2)
        y1 = min(h, y + brush // 2 + 1)
        x0 = max(0, x - brush // 2)
        x1 = min(w, x + brush // 2 + 1)
        new_pixels = int((known[y0:y1, x0:x1] == 0).sum())
        known[y0:y1, x0:x1] = 1.0
        filled += new_pixels

        dy, dx = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1),
                                 (-1, -1), (-1, 1), (1, -1), (1, 1)])
        step = random.randint(1, 4)
        y = max(0, min(h - 1, y + dy * step))
        x = max(0, min(w - 1, x + dx * step))

    # known=1 → mask=0 (known), known=0 → mask=1 (missing)
    mask = 1.0 - known
    return torch.from_numpy(mask).unsqueeze(0)


def _straight_line_mask(h: int, w: int) -> torch.Tensor:
    """1-3 known straight lines (sensor tracks); rest is missing (~90-99% missing)."""
    known = torch.zeros(1, h, w)
    n_lines = random.randint(1, 3)

    for _ in range(n_lines):
        line_width = random.randint(1, 4)
        radius = line_width // 2

        # Random start point
        sy, sx = random.randint(0, h - 1), random.randint(0, w - 1)

        # Random angle → end point at boundary
        angle = random.uniform(0, 2 * math.pi)
        dx_dir = math.cos(angle)
        dy_dir = math.sin(angle)

        candidates = []
        if dx_dir > 0:
            candidates.append((w - 1 - sx) / dx_dir)
        elif dx_dir < 0:
            candidates.append(-sx / dx_dir)
        if dy_dir > 0:
            candidates.append((h - 1 - sy) / dy_dir)
        elif dy_dir < 0:
            candidates.append(-sy / dy_dir)
        candidates = [t for t in candidates if t > 0]
        if not candidates:
            continue
        t = min(candidates)
        ex = int(round(sx + t * dx_dir))
        ey = int(round(sy + t * dy_dir))
        ex = max(0, min(w - 1, ex))
        ey = max(0, min(h - 1, ey))

        # Draw thick line as KNOWN
        length = max(abs(ex - sx), abs(ey - sy)) + 1
        if length <= 1:
            continue
        ys = np.linspace(sy, ey, length)
        xs = np.linspace(sx, ex, length)
        for yy, xx in zip(ys, xs):
            cy, cx = int(round(yy)), int(round(xx))
            for ddy in range(-radius, radius + 1):
                for ddx in range(-radius, radius + 1):
                    ny, nx = cy + ddy, cx + ddx
                    if 0 <= ny < h and 0 <= nx < w:
                        known[0, ny, nx] = 1.0

    # Invert: known → 0, unknown → 1
    mask = 1.0 - known
    return mask


def _sparse_gaussian_mask(h: int, w: int) -> torch.Tensor:
    """Sparse random known measurement points; rest is missing (~90-99.99% missing).

    Represents scattered sensor measurements across the domain.
    Log-uniform known fraction: 0.01% to 10% of pixels are known.
    """
    log_rate = random.uniform(math.log(0.0001), math.log(0.10))
    rate = math.exp(log_rate)
    known = (torch.rand(1, h, w) < rate).float()
    mask = 1.0 - known  # known points → 0, rest → 1 (missing)
    return mask


def _pixel_sampling_mask(h: int, w: int) -> torch.Tensor:
    """Random known pixel samples; rest is missing (~85-97% missing).

    Represents random sub-sampling of the field.
    3-15% of pixels are known observations.
    """
    known_rate = random.uniform(0.03, 0.15)
    known = (torch.rand(1, h, w) < known_rate).float()
    mask = 1.0 - known  # known → 0, missing → 1
    return mask


def _robot_bfs_mask(h: int, w: int, land_mask: torch.Tensor = None) -> torch.Tensor:
    """Greedy robot-path exploration = known; rest is missing (~60-90% missing).

    Matches the real test scenario: robot explores ~20% of valid cells,
    the remaining ~80% are missing and must be inpainted.
    """
    # Build valid mask (ocean only)
    if land_mask is not None:
        valid = land_mask.squeeze().numpy().astype(np.float32)
    else:
        valid = np.ones((h, w), dtype=np.float32)

    mask = np.ones((h, w), dtype=np.float32)  # start all-missing
    valid_coords = np.argwhere(valid > 0.5)
    if len(valid_coords) == 0:
        return torch.from_numpy(mask).unsqueeze(0)

    # Robot explores 15-25% of valid cells → ~75-85% missing
    n_valid = len(valid_coords)
    explore_fraction = random.uniform(0.15, 0.25)
    target = int(n_valid * explore_fraction)

    # Start from random valid point
    sy, sx = valid_coords[random.randint(0, len(valid_coords) - 1)]
    visited = np.zeros((h, w), dtype=bool)
    visited[sy, sx] = True
    mask[sy, sx] = 0.0  # explored = known
    explored = 1

    # Greedy random walk
    y, x = int(sy), int(sx)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while explored < target:
        random.shuffle(directions)
        moved = False
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and valid[ny, nx] > 0.5 and not visited[ny, nx]:
                visited[ny, nx] = True
                mask[ny, nx] = 0.0
                y, x = ny, nx
                explored += 1
                moved = True
                break

        if not moved:
            unvisited = np.argwhere((valid > 0.5) & (~visited))
            if len(unvisited) == 0:
                break
            jy, jx = unvisited[random.randint(0, len(unvisited) - 1)]
            y, x = int(jy), int(jx)
            visited[y, x] = True
            mask[y, x] = 0.0
            explored += 1

    return torch.from_numpy(mask).unsqueeze(0)
