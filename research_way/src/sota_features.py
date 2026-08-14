"""Rich per-recording feature blocks for the SOTA campaign.

Scope note (read this before reusing these numbers)
---------------------------------------------------
This module powers the *paper-comparable* track: subject-shared random splits,
the protocol the StressID origin paper uses (random 80/20 + SMOTE). That
protocol lets the same subject appear in train and test, which inflates scores
(quantified in ../../LEAKY_PROTOCOL.md). We keep it here **only** so our numbers
are directly comparable to published ones; the leakage-free GroupKFold track in
`classical.py` is unchanged and remains the honest estimate.

Because subject identity is shared across the split by construction in this
track, subject-referenced normalisation is admissible and is used explicitly:

  * ``subject-relative``  -- subtract that subject's own low-stress baseline
    (mean over their Relax/Breathing recordings). No labels are read.
  * ``subject-z``         -- z-score each feature within a subject over all of
    that subject's recordings. Transductive, no labels read.

Both are declared in every report so no reader can mistake them for free lunch.

Blocks produced per recording (all fixed-width, NaN-free, float32):

  physfeat   neurokit2 HRV/EDA/RSP window features -> 10 aggregates  (29 x 10)
  physraw    time/frequency descriptors on the 64 Hz cached waveform (3 chans)
  audio      log-mel band statistics + deltas + spectral shape
  videostat  coarse pixel statistics (the original baseline block)
  videofeat  regional dynamics + LBP texture -> 3 aggregates         (248 x 3)
  avail      3 modality-availability bits (a task proxy - reported separately)

Everything is cached to ``data/sotafeat_<tag>/<key>.npz`` so downstream sweeps
cost seconds, not minutes. Bump ``FEATURE_VERSION`` to invalidate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal as sps
from scipy import stats as sst

from .config import Config, DATA_DIR
from .video_features import extract_recording as extract_video_features

FEATURE_VERSION = 3

BASELINE_TASKS = ("Relax", "Breathing")   # low-stress reference recordings
PHYSIO_CHANNELS = ("ECG", "EDA", "RR")


# --------------------------------------------------------------- aggregators

def _agg_windows(f: np.ndarray) -> np.ndarray:
    """[W, D] window features -> [D*10] recording summary.

    Level (mean/median), spread (std/IQR), extremes (min/max), trend (linear
    slope, last-minus-first) and edges (first/last). The trend terms matter:
    stress responses ramp over a task, so a recording whose EDA climbs is not
    the same as one that starts high and decays, yet mean/std cannot tell them
    apart.
    """
    if len(f) == 0:
        f = np.zeros((1, f.shape[-1]), np.float32)
    w = np.arange(len(f), dtype=np.float64)
    if len(f) > 1:
        wc = w - w.mean()
        slope = (wc[:, None] * (f - f.mean(0))).sum(0) / (wc ** 2).sum()
    else:
        slope = np.zeros(f.shape[-1])
    q75, q25 = np.percentile(f, [75, 25], axis=0)
    return np.concatenate([
        f.mean(0), np.median(f, axis=0), f.std(0), q75 - q25,
        f.min(0), f.max(0), slope, f[-1] - f[0], f[0], f[-1],
    ]).astype(np.float32)


N_AGG = 10


def _channel_descriptors(x: np.ndarray, fs: float) -> np.ndarray:
    """1-D signal -> 18 time/frequency descriptors."""
    if len(x) < 8 or not np.isfinite(x).any():
        return np.zeros(18, np.float32)
    x = np.nan_to_num(x.astype(np.float64))
    d = np.diff(x)
    p10, p25, p50, p75, p90 = np.percentile(x, [10, 25, 50, 75, 90])
    # Welch PSD in five bands spanning 0-8 Hz: covers RSP (0-0.5), EDA phasic
    # (0.05-0.5), HR band (0.5-2) and QRS energy (2-8) at the 64 Hz cache rate.
    nper = min(256, len(x))
    freqs, psd = sps.welch(x, fs=fs, nperseg=nper)
    edges = [0.0, 0.15, 0.4, 1.0, 3.0, 8.0]
    total = psd.sum() + 1e-12
    bands = [psd[(freqs >= a) & (freqs < b)].sum() / total for a, b in zip(edges, edges[1:])]
    return np.nan_to_num(np.array([
        x.mean(), x.std(), sst.skew(x), sst.kurtosis(x),
        p10, p25, p50, p75, p90, p90 - p10,
        np.abs(d).mean(), d.std(),
        float(((x[:-1] * x[1:]) < 0).mean()),          # zero-crossing rate
        float((x ** 2).mean()),                        # energy
        *bands[:4],
    ], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _physraw_block(physio: np.ndarray, mask: np.ndarray, n: int, fs: float) -> np.ndarray:
    """[W, T, C] cached waveform -> descriptors per channel, concatenated."""
    w = mask[:n, 0] > 0
    if not w.any():
        return np.zeros(18 * physio.shape[-1] * 2, np.float32)
    v = physio[:n][w]                                   # [w, T, C]
    flat = v.reshape(-1, v.shape[-1])
    per_chan = [_channel_descriptors(flat[:, c], fs) for c in range(v.shape[-1])]
    # window-to-window variability of the channel means: slow drift over a task
    wmean = v.mean(axis=1)                              # [w, C]
    drift = []
    for c in range(v.shape[-1]):
        s = wmean[:, c]
        tr = np.polyfit(np.arange(len(s)), s, 1)[0] if len(s) > 1 else 0.0
        drift.append(_channel_descriptors(s, 1.0)[:17])
        drift.append(np.array([tr], np.float32))
    return np.concatenate([np.concatenate(per_chan), np.concatenate(drift)]).astype(np.float32)


def _audio_block(audio: np.ndarray, mask: np.ndarray, n: int) -> np.ndarray:
    """[W, F, M] log-mel -> band statistics, deltas and spectral shape.

    Width is fixed whether or not audio exists (58% of recordings have none);
    absent audio yields the zero vector, which downstream standardisation maps
    to the train mean.
    """
    n_mels = audio.shape[-1]
    width = n_mels * 7 + 14
    w = mask[:n, 1] > 0
    if not w.any():
        return np.zeros(width, np.float32)
    m = audio[:n][w].astype(np.float32).reshape(-1, n_mels)     # [T, M]
    if len(m) < 3:
        return np.zeros(width, np.float32)
    d1 = np.diff(m, axis=0)
    p10, p50, p90 = np.percentile(m, [10, 50, 90], axis=0)
    band = np.concatenate([m.mean(0), m.std(0), p10, p50, p90,
                           np.abs(d1).mean(0), d1.std(0)])

    # spectral shape over time: centroid / spread / flux / rolloff / flatness,
    # each summarised by mean and std across frames. These are the descriptors
    # that separate strained from relaxed speech.
    e = np.exp(m) + 1e-8
    idx = np.arange(n_mels, dtype=np.float64)
    tot = e.sum(1) + 1e-12
    cent = (e * idx).sum(1) / tot
    spread = np.sqrt((e * (idx - cent[:, None]) ** 2).sum(1) / tot)
    flux = np.concatenate([[0.0], np.sqrt(((np.diff(e, axis=0)) ** 2).sum(1))])
    csum = np.cumsum(e, axis=1) / tot[:, None]
    roll = np.argmax(csum >= 0.85, axis=1).astype(np.float64)
    flat = np.exp(np.log(e).mean(1)) / (e.mean(1) + 1e-12)
    energy = m.mean(1)
    shape = np.concatenate([[s.mean(), s.std()] for s in
                            (cent, spread, flux, roll, flat, energy, m.max(1))])
    return np.nan_to_num(np.concatenate([band, shape]),
                         nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _videostat_block(video: np.ndarray, mask: np.ndarray, n: int) -> np.ndarray:
    """Coarse pixel statistics -- the block the original baselines used."""
    w = mask[:n, 2] > 0
    px = video.shape[-2] * video.shape[-1]
    width = 2 + 64 + 3
    if not w.any():
        return np.zeros(width, np.float32)
    v = video[:n][w].astype(np.float32).reshape(-1, px)
    vm = v.mean(0)
    diff = np.abs(np.diff(v, axis=0)).mean(0)[:64] if len(v) > 1 else np.zeros(64, np.float32)
    return np.nan_to_num(np.concatenate([
        [vm.mean(), vm.std()], diff, np.percentile(v, [10, 50, 90], axis=0).mean(1),
    ]), nan=0.0).astype(np.float32)


# ------------------------------------------------------------------ per-file

def _videofeat_windows(cfg: Config, key: str, z) -> np.ndarray:
    """Cached C4 video descriptors, computed on demand."""
    d = DATA_DIR / f"videofeat_{cfg.data_tag}"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{key}.npz"
    if p.exists():
        with np.load(p) as c:
            return c["feat"]
    f = extract_video_features(z["video"], z["mask"], int(z["n_windows"]), cfg.max_windows)
    np.savez_compressed(p, feat=f)
    return f


def _valid(f: np.ndarray) -> np.ndarray:
    v = f[(f != 0).any(axis=1)]
    return v if len(v) else np.zeros((1, f.shape[1]), np.float32)


def extract_recording(cfg: Config, key: str) -> dict[str, np.ndarray]:
    with np.load(cfg.cache_dir / f"{key}.npz") as z:
        n = int(z["n_windows"])
        mask, physio, audio, video = z["mask"], z["physio"], z["audio"], z["video"]
        vfw = _videofeat_windows(cfg, key, z)
    with np.load(cfg.physfeat_dir / f"{key}.npz") as z:
        pfw = z["feat"]

    vf = _valid(vfw)
    return {
        "physfeat": _agg_windows(_valid(pfw)),
        "physraw": _physraw_block(physio, mask, n, cfg.physio_fs),
        "audio": _audio_block(audio, mask, n),
        "videostat": _videostat_block(video, mask, n),
        "videofeat": np.concatenate([vf.mean(0), vf.std(0), vf.max(0)]).astype(np.float32),
    }


BLOCKS = ("physfeat", "physraw", "audio", "videostat", "videofeat", "avail")


def build(cfg: Config, man: pd.DataFrame, verbose: bool = True) -> dict[str, np.ndarray]:
    """{block -> [N, D]} for every row of `man`, disk-cached per recording."""
    out_dir = DATA_DIR / f"sotafeat_{cfg.data_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    acc: dict[str, list] = {}
    for i, key in enumerate(man["key"]):
        p = out_dir / f"{key}_v{FEATURE_VERSION}.npz"
        if p.exists():
            with np.load(p) as z:
                d = {k: z[k] for k in z.files}
        else:
            d = extract_recording(cfg, key)
            np.savez_compressed(p, **d)
        for k, v in d.items():
            acc.setdefault(k, []).append(v)
        if verbose and (i + 1) % 100 == 0:
            print(f"[sotafeat] {i + 1}/{len(man)}", flush=True)

    feats = {k: np.nan_to_num(np.stack(v).astype(np.float64)) for k, v in acc.items()}
    feats["avail"] = man[["has_physio", "has_audio", "has_video"]].values.astype(np.float64)
    return feats


# ------------------------------------------------- subject-referenced views

def subject_relative(X: np.ndarray, man: pd.DataFrame) -> np.ndarray:
    """x - (that subject's mean over their Relax/Breathing recordings).

    Baseline correction against the participant's own resting state. No labels
    are consulted. Subjects with no baseline recording fall back to their own
    overall mean, which is still within-subject.
    """
    out = X.copy()
    is_base = man["task"].isin(BASELINE_TASKS).values
    for s in man["subject"].unique():
        m = (man["subject"] == s).values
        b = X[m & is_base]
        out[m] = X[m] - (b.mean(0) if len(b) else X[m].mean(0))
    return out


def subject_z(X: np.ndarray, man: pd.DataFrame) -> np.ndarray:
    """(x - subject mean) / subject std, over that subject's own recordings.

    Removes between-person offsets and gains, so the model sees "how unusual is
    this recording *for this person*". Transductive (test rows contribute to
    their own subject's statistics) and therefore only valid in the
    subject-shared track. Declared as such in every report.
    """
    out = X.copy()
    for s in man["subject"].unique():
        m = (man["subject"] == s).values
        v = X[m]
        out[m] = (v - v.mean(0)) / (v.std(0) + 1e-6)
    return out


VIEWS = {"raw": lambda X, man: X, "rel": subject_relative, "z": subject_z}
