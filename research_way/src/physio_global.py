"""Whole-recording physiological features — the ones a 10 s window cannot carry.

The gap this closes
-------------------
`physio_features.py` computes HRV per 10 s window, and its own docstring names
the limitation: at that length, time-domain HRV is noisy and **frequency-domain
HRV is not computable at all** — LF (0.04–0.15 Hz) needs ≥60 s to resolve even
one cycle. So the campaign has been running without LF/HF, which is the single
most cited autonomic stress marker in the literature: sympathetic activation
raises LF and suppresses HF, and the ratio is the standard readout.

StressID tasks run 60–90 s. That is enough for LF/HF at the *recording* level
even though it is hopeless at the window level. This module therefore works on
the whole recording at the original 500 Hz, and emits what only that view can
give:

  ECG   frequency-domain HRV (LF, HF, LF/HF, total power, normalised units),
        plus time-domain HRV computed over the full RR series rather than over
        ~12 beats, plus the HR trend across the task
  EDA   tonic/phasic decomposition over the whole task: SCR *rate*,
        amplitude statistics, mean phasic activity, tonic level and slope
  RSP   breathing rate and its variability over the task, amplitude statistics,
        and the inhale/exhale duty cycle

The duration trap
-----------------
Everything here is expressed as a rate, a mean or a per-second slope, and never
as a count. That is not stylistic. StressID task durations are near-deterministic
per task — every high-stress task runs 59 s while Relax, Breathing and Video run
118–177 s — so recording length *is* the task label, and any count that grows
with length smuggles it in. Version 1 of this module did exactly that: its
strongest feature by far was `g_n_beats` (Cohen's d 0.74, correlation 0.937 with
duration), which is the clock wearing a stethoscope.

Measured, so the size of the trap is on record: a random forest given **duration
alone** — one scalar, no physiology — scores **0.5923 macro F1**, against a
0.344 majority baseline and the full multimodal pipeline's 0.7484.

Cached to `data/physglobal_<tag>/<key>.npz`. Roughly 1–2 s per recording.

    python -m src.physio_global          # build the cache for the full corpus
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
from scipy import signal as sps

from .config import Config, DATA_DIR, full_config
from . import signals
from .preprocess import _paths

warnings.filterwarnings("ignore")

try:
    import neurokit2 as nk
    HAVE_NK = True
except Exception:                                    # pragma: no cover
    HAVE_NK = False

GLOBAL_VERSION = 2   # v1 leaked recording duration; see the note on ECG_G

# DURATION MUST NOT LEAK IN. StressID task durations are near-deterministic per
# task -- every high-stress task runs 59 s while Relax/Breathing/Video run
# 118-177 s -- so recording length is a task label in disguise. Measured: a
# random forest on duration ALONE, with no physiology whatsoever, scores 0.5923
# macro F1 against a 0.344 majority baseline. Any count that scales with
# duration is therefore banned from this block, however physiological it looks:
#
#   g_n_beats   correlated 0.937 with duration, Cohen's d 0.74 -- the single
#               strongest "physiological" discriminator, and it was measuring
#               the clock
#   g_n_breaths same
#   g_scr_count same; kept only as g_scr_rate_min (count per minute)
#   g_phasic_auc  was a running sum; now a per-second mean
#
# Slopes are expressed per second rather than per sample for the same reason: a
# per-sample slope shrinks with recording length for an identical total change.
ECG_G = ["g_hr_mean", "g_hr_std", "g_hr_slope", "g_sdnn", "g_rmssd", "g_pnn50",
         "g_cvnn", "g_rr_range",
         "g_lf", "g_hf", "g_lf_hf", "g_total_power", "g_lf_nu", "g_hf_nu"]
EDA_G = ["g_scl_mean", "g_scl_std", "g_scl_slope", "g_scl_range",
         "g_scr_rate_min", "g_scr_amp_mean", "g_scr_amp_max",
         "g_phasic_mean_abs", "g_phasic_std"]
RSP_G = ["g_rsp_rate", "g_rsp_rate_std", "g_rsp_rate_slope", "g_rrv_rmssd",
         "g_rsp_amp_mean", "g_rsp_amp_std", "g_rsp_duty"]
FEATURE_NAMES = ECG_G + EDA_G + RSP_G
N_FEATURES = len(FEATURE_NAMES)


def _slope(x: np.ndarray, fs: float = 1.0) -> float:
    """Least-squares slope in units per SECOND.

    Per-sample slope would be duration-dependent: the same total drift across a
    59 s task and a 177 s task yields a threefold different per-sample slope,
    which reintroduces the duration shortcut this module is careful to exclude.
    """
    if len(x) < 3:
        return 0.0
    t = np.arange(len(x), dtype=np.float64) / fs
    tc = t - t.mean()
    d = (tc ** 2).sum()
    return float((tc * (x - x.mean())).sum() / d) if d > 0 else 0.0


def _hrv_frequency(rr_ms: np.ndarray) -> list[float]:
    """LF / HF / ratio / total power from an RR interval series.

    The RR series is unevenly sampled by construction, so it is resampled onto a
    uniform 4 Hz grid before the periodogram — the standard approach. Returns
    zeros when the recording is too short to resolve LF, rather than a number
    that would be meaningless.
    """
    if len(rr_ms) < 12:
        return [0.0] * 6
    t = np.cumsum(rr_ms) / 1000.0
    dur = t[-1] - t[0]
    if dur < 30.0:                    # cannot resolve 0.04 Hz in under ~25 s
        return [0.0] * 6
    fs = 4.0
    grid = np.arange(t[0], t[-1], 1.0 / fs)
    if len(grid) < 32:
        return [0.0] * 6
    series = np.interp(grid, t, rr_ms)
    series = series - series.mean()
    nper = min(len(series), int(fs * 60))
    freqs, psd = sps.welch(series, fs=fs, nperseg=nper)
    band = lambda lo, hi: float(np.trapezoid(psd[(freqs >= lo) & (freqs < hi)],
                                             freqs[(freqs >= lo) & (freqs < hi)])) \
        if ((freqs >= lo) & (freqs < hi)).sum() > 1 else 0.0
    lf, hf = band(0.04, 0.15), band(0.15, 0.40)
    vlf = band(0.003, 0.04)
    total = vlf + lf + hf
    ratio = lf / hf if hf > 1e-12 else 0.0
    lf_nu = 100.0 * lf / (lf + hf) if (lf + hf) > 1e-12 else 0.0
    hf_nu = 100.0 * hf / (lf + hf) if (lf + hf) > 1e-12 else 0.0
    return [lf, hf, ratio, total, lf_nu, hf_nu]


def _ecg_block(ecg: np.ndarray, fs: int) -> np.ndarray:
    try:
        clean = nk.ecg_clean(ecg, sampling_rate=fs)
        _, info = nk.ecg_peaks(clean, sampling_rate=fs)
        peaks = np.asarray(info["ECG_R_Peaks"], dtype=float)
        if len(peaks) < 6:
            return np.zeros(len(ECG_G))
        rr = np.diff(peaks) / fs * 1000.0
        rr = rr[(rr > 300) & (rr < 2000)]            # physiological plausibility
        if len(rr) < 5:
            return np.zeros(len(ECG_G))
        hr = 60000.0 / rr
        d = np.diff(rr)
        # hr is indexed per beat; convert to per second using the mean RR so
        # the trend is comparable across recordings of different lengths
        hr_fs = 1000.0 / rr.mean() if rr.mean() > 0 else 1.0
        time_dom = [hr.mean(), hr.std(), _slope(hr, hr_fs), rr.std(),
                    float(np.sqrt((d ** 2).mean())),
                    float((np.abs(d) > 50).mean() * 100.0),
                    float(rr.std() / rr.mean()) if rr.mean() > 0 else 0.0,
                    float(rr.max() - rr.min())]
        return np.array(time_dom + _hrv_frequency(rr))
    except Exception:
        return np.zeros(len(ECG_G))


def _eda_block(eda: np.ndarray, fs: int) -> np.ndarray:
    try:
        sig, info = nk.eda_process(eda, sampling_rate=fs)
        tonic = np.asarray(sig["EDA_Tonic"])
        phasic = np.asarray(sig["EDA_Phasic"])
        amps = np.asarray(info.get("SCR_Amplitude", []), dtype=float)
        amps = amps[np.isfinite(amps)]
        dur_min = len(eda) / fs / 60.0
        n_scr = float(len(amps))
        return np.array([
            tonic.mean(), tonic.std(), _slope(tonic, fs),
            float(tonic.max() - tonic.min()),
            n_scr / dur_min if dur_min > 0 else 0.0,
            float(amps.mean()) if len(amps) else 0.0,
            float(amps.max()) if len(amps) else 0.0,
            float(np.abs(phasic).mean()), float(phasic.std()),
        ])
    except Exception:
        return np.zeros(len(EDA_G))


def _rsp_block(rsp: np.ndarray, fs: int) -> np.ndarray:
    try:
        sig, info = nk.rsp_process(rsp, sampling_rate=fs)
        rate = np.asarray(sig["RSP_Rate"])
        rate = rate[np.isfinite(rate)]
        amp = np.asarray(sig["RSP_Amplitude"])
        amp = amp[np.isfinite(amp)]
        peaks = np.asarray(info.get("RSP_Peaks", []), dtype=float)
        troughs = np.asarray(info.get("RSP_Troughs", []), dtype=float)
        if len(rate) < 3:
            return np.zeros(len(RSP_G))
        bb = np.diff(peaks) / fs * 1000.0 if len(peaks) > 2 else np.array([0.0])
        rrv = float(np.sqrt((np.diff(bb) ** 2).mean())) if len(bb) > 2 else 0.0
        # duty cycle: fraction of the breath cycle spent inhaling
        duty = 0.0
        if len(peaks) > 1 and len(troughs) > 1:
            n = min(len(peaks), len(troughs)) - 1
            if n > 0:
                inh = np.abs(peaks[:n] - troughs[:n])
                cyc = np.abs(np.diff(peaks)[:n])
                ok = cyc > 0
                duty = float((inh[ok] / cyc[ok]).mean()) if ok.any() else 0.0
        return np.array([rate.mean(), rate.std(), _slope(rate, fs), rrv,
                         amp.mean() if len(amp) else 0.0,
                         amp.std() if len(amp) else 0.0, duty])
    except Exception:
        return np.zeros(len(RSP_G))


def extract(sig_raw: np.ndarray, fs: int) -> np.ndarray:
    if not HAVE_NK:
        return np.zeros(N_FEATURES, np.float32)
    out = np.concatenate([_ecg_block(sig_raw[:, 0], fs),
                          _eda_block(sig_raw[:, 1], fs),
                          _rsp_block(sig_raw[:, 2], fs)])
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build(cfg: Config, man: pd.DataFrame, verbose: bool = True) -> np.ndarray:
    """[N, N_FEATURES] for every recording, disk-cached per recording."""
    d = DATA_DIR / f"physglobal_{cfg.data_tag}"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, r in enumerate(man.itertuples()):
        p = d / f"{r.key}_v{GLOBAL_VERSION}.npz"
        if p.exists():
            with np.load(p) as z:
                rows.append(z["feat"])
        else:
            try:
                sig = signals.read_physio(_paths(r.subject, r.task)["physio"],
                                          cfg.physio_fs_raw, cfg.physio_fs_raw,
                                          cfg.physio_channels)
                f = extract(sig, cfg.physio_fs_raw)
            except Exception:
                f = np.zeros(N_FEATURES, np.float32)
            np.savez_compressed(p, feat=f)
            rows.append(f)
        if verbose and (i + 1) % 50 == 0:
            print(f"[physglobal] {i + 1}/{len(man)}", flush=True)
    return np.nan_to_num(np.stack(rows).astype(np.float64))


def main() -> None:
    cfg = full_config()
    man = pd.read_csv(cfg.manifest_path)
    X = build(cfg, man)
    nz = (X != 0).any(0)
    print(f"[physglobal] {X.shape}, non-constant columns {int(nz.sum())}/{X.shape[1]}")
    for name, col in zip(FEATURE_NAMES, X.T):
        print(f"  {name:<18} mean={col.mean():10.3f}  nonzero={float((col!=0).mean()):.2f}")


if __name__ == "__main__":
    sys.exit(main())
