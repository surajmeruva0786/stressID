"""C4 - richer video features from the cached face crops.

Video is the strongest single modality on the leakage-free subset (§15.3), yet
the baseline description of it is almost nothing: a global pixel mean/std, a
64-element diff-energy vector and three percentiles. That discards where on the
face motion happens and what its texture looks like -- exactly the information
action units would carry.

The planning doc calls for OpenFace AUs, which are not available here and would
need the raw video re-decoded at full resolution (~1 h). These features are the
best obtainable from the existing 32x32 grayscale crop cache:

  * regional temporal dynamics  - motion energy / variability on a 4x4 grid, so
    brow, eye and mouth regions are described separately rather than pooled
  * LBP texture histograms      - per-region uniform local binary patterns,
    a standard facial-expression descriptor, robust to illumination
  * global temporal profile     - frame-to-frame motion trajectory summary

    feats = extract_recording(video, mask, n_windows)   # [max_windows, D]
"""
from __future__ import annotations

import numpy as np
from skimage.feature import local_binary_pattern

GRID = 4                      # 4x4 regions over the 32x32 crop -> 8x8 px each
LBP_P, LBP_R = 8, 1           # 8 neighbours, radius 1
N_LBP_BINS = LBP_P + 2        # 'uniform' method -> P+2 bins

_REGION_STATS = 5             # mean, std, motion_mean, motion_max, temporal_std
N_REGION_FEATURES = GRID * GRID * _REGION_STATS
N_LBP_FEATURES = GRID * GRID * N_LBP_BINS
N_GLOBAL = 8
N_FEATURES = N_REGION_FEATURES + N_LBP_FEATURES + N_GLOBAL


def _regions(x: np.ndarray) -> np.ndarray:
    """[F, S, S] -> [F, GRID*GRID, cell, cell] region view."""
    f, s, _ = x.shape
    c = s // GRID
    x = x[:, : c * GRID, : c * GRID]
    return (x.reshape(f, GRID, c, GRID, c)
             .transpose(0, 1, 3, 2, 4)
             .reshape(f, GRID * GRID, c, c))


def window_features(frames: np.ndarray) -> np.ndarray:
    """`frames` [F, S, S] float in [0,1] -> feature vector [N_FEATURES]."""
    f = frames.astype(np.float32)
    if len(f) < 2:
        return np.zeros(N_FEATURES, np.float32)

    reg = _regions(f)                              # [F, R, c, c]
    reg_mean = reg.mean(axis=(2, 3))               # [F, R]
    motion = np.abs(np.diff(reg_mean, axis=0))     # [F-1, R]

    region_feats = np.concatenate([
        reg_mean.mean(0),                          # appearance level
        reg_mean.std(0),                           # appearance spread
        motion.mean(0),                            # mean motion energy
        motion.max(0),                             # peak motion
        reg.std(axis=(2, 3)).mean(0),              # within-region texture spread
    ])

    # LBP on the temporally-averaged face: texture of the resting expression
    avg = f.mean(0)
    rng = avg.max() - avg.min()
    avg8 = (((avg - avg.min()) / rng) * 255).astype(np.uint8) if rng > 1e-6 \
        else np.zeros_like(avg, np.uint8)
    lbp = local_binary_pattern(avg8, LBP_P, LBP_R, method="uniform")
    lbp_reg = _regions(lbp[None])[0]                # [R, c, c]
    lbp_feats = np.concatenate([
        np.histogram(r, bins=N_LBP_BINS, range=(0, N_LBP_BINS), density=True)[0]
        for r in lbp_reg])

    frame_mean = f.mean(axis=(1, 2))
    gmotion = np.abs(np.diff(frame_mean))
    global_feats = np.array([
        frame_mean.mean(), frame_mean.std(),
        gmotion.mean(), gmotion.std(), gmotion.max(),
        float((gmotion > gmotion.mean()).mean()),   # fraction of active frames
        f.std(), np.percentile(f, 90) - np.percentile(f, 10),
    ])

    out = np.concatenate([region_feats, lbp_feats, global_feats])
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def extract_recording(video: np.ndarray, mask: np.ndarray,
                      n_windows: int, max_windows: int) -> np.ndarray:
    """`video` [W, F, S, S] -> [max_windows, N_FEATURES]; absent windows stay 0."""
    out = np.zeros((max_windows, N_FEATURES), np.float32)
    for i in range(min(n_windows, max_windows)):
        if mask[i, 2] <= 0:                         # no video in this window
            continue
        out[i] = window_features(video[i].astype(np.float32))
    return out
