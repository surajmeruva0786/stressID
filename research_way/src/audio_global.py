"""Voice-quality features from the raw waveform — the audio analogue of
`physio_global`.

The gap this closes
-------------------
The audio block so far is derived entirely from a log-mel spectrogram: band
energies, deltas and spectral shape. That describes *what the spectrum looks
like* but not *how the voice is being produced*, and the stress literature is
built almost entirely on the latter. Under stress the larynx tenses: F0 rises,
cycle-to-cycle period variation (jitter) and amplitude variation (shimmer)
change, the harmonics-to-noise ratio falls, and energy shifts toward higher
frequencies. None of that is recoverable from 40 mel bands at 25 ms.

openSMILE/eGeMAPS and Praat are the usual tools and neither is installed here,
so these are implemented directly from the waveform:

  F0        autocorrelation pitch track, 60-400 Hz -> mean, std, range, slope,
            and the fraction of frames that are voiced
  jitter    mean absolute period-to-period change, normalised by mean period
  shimmer   the same for cycle amplitude
  HNR       from the height of the normalised autocorrelation peak
  spectral  centroid, spread, rolloff, flatness, slope, the alpha ratio and the
            Hammarberg index -- the standard "energy moved upward" descriptors
  loudness  RMS level, its variability, and the voiced-segment RATE

Audio exists for only 378 of 700 recordings (the speech tasks), and absent
audio yields the zero vector, which downstream standardisation maps to the
train mean.

Duration safety
---------------
Every feature is a mean, a ratio, a fraction or a per-second rate. Not one is a
count. `physio_global` v1 shipped counts and they turned out to be the strongest
"physiological" predictor in the whole block purely because StressID's
high-stress tasks all run 59 s while its low-stress tasks run 117-177 s. That
mistake is not repeated here.

    python -m src.audio_global          # build the cache for the full corpus
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

from .config import Config, DATA_DIR, full_config
from . import signals
from .preprocess import _paths

warnings.filterwarnings("ignore")

AUDIO_VERSION = 1

F0_MIN, F0_MAX = 60.0, 400.0
FRAME_MS, HOP_MS = 40.0, 10.0
VOICED_R = 0.45          # normalised autocorrelation peak above which a frame is voiced

FEATURE_NAMES = [
    "a_f0_mean", "a_f0_std", "a_f0_p10", "a_f0_p90", "a_f0_range", "a_f0_slope",
    "a_voiced_frac", "a_voiced_rate",
    "a_jitter", "a_shimmer", "a_hnr_mean", "a_hnr_std",
    "a_rms_mean", "a_rms_std", "a_rms_range", "a_rms_slope",
    "a_centroid_mean", "a_centroid_std", "a_spread_mean",
    "a_rolloff_mean", "a_flatness_mean", "a_slope_mean",
    "a_alpha_ratio", "a_hammarberg", "a_zcr_mean",
]
N_FEATURES = len(FEATURE_NAMES)


def _frames(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def _slope_per_s(v: np.ndarray, hop_s: float) -> float:
    if len(v) < 3:
        return 0.0
    t = np.arange(len(v)) * hop_s
    tc = t - t.mean()
    d = (tc ** 2).sum()
    return float((tc * (v - v.mean())).sum() / d) if d > 0 else 0.0


def _autocorr_pitch(F: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame (F0 in Hz, normalised autocorrelation peak). 0 where unvoiced.

    FFT autocorrelation over the whole frame matrix at once -- a per-frame
    Python loop over ~6000 frames per recording would dominate the build.
    """
    win = F.shape[1]
    w = np.hanning(win)[None, :]
    Fw = (F - F.mean(1, keepdims=True)) * w
    n_fft = 1 << int(np.ceil(np.log2(2 * win)))
    S = np.fft.rfft(Fw, n=n_fft, axis=1)
    ac = np.fft.irfft(S * np.conj(S), n=n_fft, axis=1)[:, :win]
    r0 = ac[:, :1].copy()
    r0[r0 <= 0] = 1e-12
    acn = ac / r0

    lo, hi = int(sr / F0_MAX), min(int(sr / F0_MIN), win - 1)
    if hi <= lo + 1:
        return np.zeros(len(F)), np.zeros(len(F))
    seg = acn[:, lo:hi]
    k = seg.argmax(1)
    peak = seg[np.arange(len(seg)), k]
    lag = k + lo
    f0 = np.where(peak > VOICED_R, sr / np.maximum(lag, 1), 0.0)
    return f0, np.clip(peak, 0.0, 0.999)


def _spectral(F: np.ndarray, sr: int) -> dict:
    win = F.shape[1]
    w = np.hanning(win)[None, :]
    S = np.abs(np.fft.rfft(F * w, axis=1)) + 1e-10
    freqs = np.fft.rfftfreq(win, 1.0 / sr)
    p = S ** 2
    tot = p.sum(1) + 1e-12
    cent = (p * freqs).sum(1) / tot
    spread = np.sqrt((p * (freqs[None, :] - cent[:, None]) ** 2).sum(1) / tot)
    csum = np.cumsum(p, axis=1) / tot[:, None]
    roll = freqs[np.argmax(csum >= 0.85, axis=1)]
    flat = np.exp(np.log(S).mean(1)) / (S.mean(1) + 1e-12)
    # spectral slope: least-squares fit of log magnitude against frequency
    fc = freqs - freqs.mean()
    slope = (fc[None, :] * (np.log(S) - np.log(S).mean(1, keepdims=True))).sum(1) \
        / max((fc ** 2).sum(), 1e-12)
    # alpha ratio: energy 50-1000 Hz vs 1-5 kHz. Hammarberg: peak 0-2 kHz vs 2-5 kHz.
    b = lambda lo, hi: p[:, (freqs >= lo) & (freqs < hi)].sum(1) + 1e-12
    alpha = 10 * np.log10(b(1000, 5000) / b(50, 1000))
    m = lambda lo, hi: p[:, (freqs >= lo) & (freqs < hi)].max(1) + 1e-12
    hamm = 10 * np.log10(m(0, 2000) / m(2000, 5000))
    return {"cent": cent, "spread": spread, "roll": roll, "flat": flat,
            "slope": slope, "alpha": alpha, "hamm": hamm}


def extract(x: np.ndarray, sr: int) -> np.ndarray:
    if len(x) < sr // 4:
        return np.zeros(N_FEATURES, np.float32)
    win, hop = int(sr * FRAME_MS / 1000), int(sr * HOP_MS / 1000)
    F = _frames(x, win, hop)
    hop_s = hop / sr

    f0, peak = _autocorr_pitch(F, sr)
    v = f0 > 0
    rms = np.sqrt((F ** 2).mean(1) + 1e-12)
    zcr = (np.diff(np.signbit(F), axis=1).sum(1) / F.shape[1]).astype(float)

    if v.sum() >= 3:
        f0v = f0[v]
        periods = 1.0 / f0v
        jitter = float(np.abs(np.diff(periods)).mean() / (periods.mean() + 1e-12))
        av = rms[v]
        shimmer = float(np.abs(np.diff(av)).mean() / (av.mean() + 1e-12))
        pk = peak[v]
        hnr = 10 * np.log10(np.clip(pk, 1e-6, 0.999) / (1 - np.clip(pk, 1e-6, 0.999)))
        f0_stats = [f0v.mean(), f0v.std(), *np.percentile(f0v, [10, 90]),
                    float(np.percentile(f0v, 90) - np.percentile(f0v, 10)),
                    _slope_per_s(f0v, hop_s)]
        hnr_stats = [float(hnr.mean()), float(hnr.std())]
    else:
        jitter = shimmer = 0.0
        f0_stats = [0.0] * 6
        hnr_stats = [0.0, 0.0]

    # voiced-segment RATE per second, not a count: onsets divided by duration
    dur_s = len(x) / sr
    onsets = int(np.sum(np.diff(v.astype(int)) == 1))
    voiced_rate = onsets / dur_s if dur_s > 0 else 0.0

    sp = _spectral(F, sr)
    out = np.array([
        *f0_stats, float(v.mean()), float(voiced_rate),
        jitter, shimmer, *hnr_stats,
        float(rms.mean()), float(rms.std()),
        float(np.percentile(rms, 90) - np.percentile(rms, 10)),
        _slope_per_s(rms, hop_s),
        float(sp["cent"].mean()), float(sp["cent"].std()), float(sp["spread"].mean()),
        float(sp["roll"].mean()), float(sp["flat"].mean()), float(sp["slope"].mean()),
        float(sp["alpha"].mean()), float(sp["hamm"].mean()), float(zcr.mean()),
    ], dtype=np.float64)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build(cfg: Config, man: pd.DataFrame, verbose: bool = True) -> np.ndarray:
    d = DATA_DIR / f"audioglobal_{cfg.data_tag}"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, r in enumerate(man.itertuples()):
        p = d / f"{r.key}_v{AUDIO_VERSION}.npz"
        if p.exists():
            with np.load(p) as z:
                rows.append(z["feat"])
        else:
            f = np.zeros(N_FEATURES, np.float32)
            if bool(r.has_audio):
                try:
                    x = signals.read_wav(_paths(r.subject, r.task)["audio"],
                                         cfg.audio_sr)
                    f = extract(x, cfg.audio_sr)
                except Exception:
                    pass
            np.savez_compressed(p, feat=f)
            rows.append(f)
        if verbose and (i + 1) % 100 == 0:
            print(f"[audioglobal] {i + 1}/{len(man)}", flush=True)
    return np.nan_to_num(np.stack(rows).astype(np.float64))


def main() -> None:
    cfg = full_config()
    man = pd.read_csv(cfg.manifest_path)
    X = build(cfg, man)
    has = man["has_audio"].values.astype(bool)
    print(f"[audioglobal] {X.shape}; {has.sum()} recordings carry audio")
    for name, col in zip(FEATURE_NAMES, X.T):
        c = col[has]
        print(f"  {name:<18} mean={c.mean():10.3f}  std={c.std():9.3f}")


if __name__ == "__main__":
    sys.exit(main())
