"""Window-level views of each recording, for models that classify windows.

Why this exists
---------------
Every model in `sota.py` sees one row per recording: 700 rows against ~1500
columns. That ratio is the binding constraint on this dataset — it is why the
1.2 M-parameter transformer in `model.py` memorised, and why the recording-level
classical models plateau.

A StressID recording is not one observation. It is a 60–90 s task cut into 16
overlapping 10 s windows, and the stress label applies to the whole task. Fitting
at window level turns 700 training examples into ~11 000, then averages the
window probabilities back up to a recording decision. The label is noisier per
window (not every second of a Stroop task is stressful) but the sample count
grows 16×, and for the boosted-tree learners that trade is heavily favourable.

This module only builds the tensors. The estimator that consumes them lives in
`sota_models.py`, and it is wired into the same nested-CV harness so window
models compete with recording models on identical inner OOF folds.

    W, mask = build_windows(cfg, man)     # W [N, 16, D], mask [N, 16] bool
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sst

from .config import Config, DATA_DIR
from .sota_features import _winsorize, _videofeat_windows

WINDOW_VERSION = 1


def _audio_window(mel: np.ndarray) -> np.ndarray:
    """[F, M] log-mel frames of one window -> compact descriptor.

    Deliberately smaller than the recording-level audio block: at window level
    there are 16x more rows, so width costs 16x more, and the per-band
    percentiles that help over a whole task are largely redundant over 10 s.
    """
    n_mels = mel.shape[-1]
    if len(mel) < 3:
        return np.zeros(n_mels * 2 + 8, np.float32)
    m = mel.astype(np.float32)
    d = np.diff(m, axis=0)
    e = np.exp(m) + 1e-8
    idx = np.arange(n_mels, dtype=np.float64)
    tot = e.sum(1) + 1e-12
    cent = (e * idx).sum(1) / tot
    spread = np.sqrt((e * (idx - cent[:, None]) ** 2).sum(1) / tot)
    flux = np.sqrt((np.diff(e, axis=0) ** 2).sum(1))
    energy = m.mean(1)
    extra = np.array([cent.mean(), cent.std(), spread.mean(), flux.mean(),
                      energy.mean(), energy.std(), m.max(), np.abs(d).mean()])
    return np.nan_to_num(np.concatenate([m.mean(0), m.std(0), extra]),
                         nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _physio_window(sig: np.ndarray) -> np.ndarray:
    """[T, C] resampled waveform of one window -> 9 descriptors per channel."""
    out = []
    for c in range(sig.shape[-1]):
        x = np.nan_to_num(sig[:, c].astype(np.float64))
        d = np.diff(x) if len(x) > 1 else np.zeros(1)
        p10, p50, p90 = np.percentile(x, [10, 50, 90])
        out.append([x.mean(), x.std(), sst.skew(x), sst.kurtosis(x),
                    p10, p50, p90, np.abs(d).mean(), (x ** 2).mean()])
    return np.nan_to_num(np.array(out, np.float64).ravel(),
                         nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_windows(cfg: Config, man: pd.DataFrame, verbose: bool = True
                  ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """-> (W [N, max_windows, D] float32, valid [N, max_windows] bool, names).

    A window is `valid` when the recording actually reaches that far; padding
    windows past the end of a short task are excluded so they cannot dilute the
    recording-level average. Missing *modalities* inside a valid window stay as
    zeros — that absence is real and is information.
    """
    out_dir = DATA_DIR / f"sotawin_{cfg.data_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    Ws, Vs = [], []
    for i, key in enumerate(man["key"]):
        p = out_dir / f"{key}_v{WINDOW_VERSION}.npz"
        if p.exists():
            with np.load(p) as z:
                Ws.append(z["w"]); Vs.append(z["v"])
        else:
            with np.load(cfg.cache_dir / f"{key}.npz") as z:
                n = int(z["n_windows"])
                mask, physio, audio = z["mask"], z["physio"], z["audio"]
                vfw = _videofeat_windows(cfg, key, z)
            with np.load(cfg.physfeat_dir / f"{key}.npz") as z:
                pfw = z["feat"]

            rows, valid = [], np.zeros(cfg.max_windows, bool)
            for w in range(cfg.max_windows):
                if w >= n:
                    rows.append(None)
                    continue
                valid[w] = True
                rows.append(np.concatenate([
                    pfw[w],
                    _physio_window(physio[w]) if mask[w, 0] > 0 else np.zeros(27, np.float32),
                    _audio_window(audio[w]) if mask[w, 1] > 0 else np.zeros(88, np.float32),
                    vfw[w] if mask[w, 2] > 0 else np.zeros(vfw.shape[-1], np.float32),
                    mask[w].astype(np.float32),
                    np.array([w / max(1, n - 1)], np.float32),   # relative position
                ]))
            width = len(next(r for r in rows if r is not None))
            W = np.stack([r if r is not None else np.zeros(width, np.float32)
                          for r in rows])
            np.savez_compressed(p, w=W, v=valid)
            Ws.append(W); Vs.append(valid)
        if verbose and (i + 1) % 200 == 0:
            print(f"[sotawin] {i + 1}/{len(man)}", flush=True)

    W = np.nan_to_num(np.stack(Ws).astype(np.float32))
    V = np.stack(Vs)
    # winsorise on the flattened valid rows so one broken window cannot set the
    # scale for the whole column (same rationale as the recording-level blocks)
    flat = W[V]
    lo, hi = np.percentile(flat, [0.5, 99.5], axis=0)
    W = np.clip(W, lo, hi) * V[..., None]
    names = [f"f{j}" for j in range(W.shape[-1])]
    return W, V, names


def subject_center(W: np.ndarray, V: np.ndarray, man: pd.DataFrame,
                   mode: str = "z") -> np.ndarray:
    """Per-participant centring of the window tensor.

    Same rationale and the same caveat as the recording-level `rel`/`z` views:
    transductive, label-free, admissible only on the subject-shared track.
    """
    out = W.copy()
    for s in man["subject"].unique():
        m = (man["subject"] == s).values
        rows = W[m][V[m]]
        if len(rows) == 0:
            continue
        mu = rows.mean(0)
        sd = rows.std(0) + 1e-6 if mode == "z" else 1.0
        out[m] = (W[m] - mu) / sd
    return out * V[..., None]
