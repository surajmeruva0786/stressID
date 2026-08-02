"""A1 - domain physiological features per window (neurokit2), replacing the
raw-signal physio encoder.

Motivation (RESEARCH_PROGRESS.md §12.1): the origin paper's handcrafted physio
features reach 0.73 weighted F1 where our learned CNN+BiLSTM encoder gets
0.54-0.57, and the training curves show the encoder memorises rather than
generalises. 448 training recordings is not enough to rediscover HRV from raw
ECG, so we hand the model the domain features directly and let it spend its
capacity on the temporal/fusion problem instead.

Features are computed on the ORIGINAL 500 Hz signal (not the 64 Hz cache) --
R-peak localisation at 64 Hz quantises RR intervals to 15.6 ms, which destroys
RMSSD/pNN50. Extraction is per window on the same window grid as the existing
cache, so the sequence model is unchanged.

    feats, names = extract_recording(ecg, eda, rsp, fs, bounds)   # [W, D]

CAVEAT: 10 s windows hold only ~10-14 beats. Time-domain HRV (SDNN, RMSSD,
pNN20/50) is noisy at that length and frequency-domain HRV (LF/HF) is not
computable at all -- it needs >=60 s. We therefore emit time-domain HRV only,
and the per-window noise is expected to be absorbed by the temporal aggregator.
"""
from __future__ import annotations

import warnings

import numpy as np

# neurokit2 is chatty about short signals; the fallbacks below handle those.
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import neurokit2 as nk
    HAVE_NK = True
except Exception:                                   # pragma: no cover
    HAVE_NK = False


ECG_FEATURES = ["hr_mean", "hr_std", "sdnn", "rmssd", "pnn20", "pnn50",
                "rr_mean", "rr_min", "rr_max", "n_beats"]
EDA_FEATURES = ["scl_mean", "scl_std", "scl_slope", "scl_range",
                "scr_n_peaks", "scr_amp_mean", "scr_amp_max", "phasic_std"]
RSP_FEATURES = ["rsp_rate", "rsp_rate_std", "rsp_amp_mean", "rsp_amp_std",
                "rrv_rmssd", "rsp_n_breaths", "rsp_rate_spec"]
RAW_FEATURES = ["ecg_std", "eda_mean", "eda_slope", "rsp_std"]

FEATURE_NAMES = ECG_FEATURES + EDA_FEATURES + RSP_FEATURES + RAW_FEATURES
N_FEATURES = len(FEATURE_NAMES)


def _safe(fn, n_out: int) -> np.ndarray:
    """Run a feature block; return zeros of the right width if it blows up.

    Short/flat/clipped windows are common in StressID (padded tails, dropped
    sensors), and neurokit2 raises on several of them. A zero vector is the
    correct 'no information' encoding here because features are standardised
    downstream on TRAIN statistics.
    """
    try:
        v = np.asarray(fn(), dtype=np.float64)
        if v.shape != (n_out,):
            return np.zeros(n_out)
        return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        return np.zeros(n_out)


def _hrv_time(rr_ms: np.ndarray) -> np.ndarray:
    """Time-domain HRV from RR intervals (ms). Cheaper than nk.hrv_time and
    tolerant of the short series a 10 s window produces."""
    if len(rr_ms) < 3:
        return np.zeros(len(ECG_FEATURES))
    d = np.diff(rr_ms)
    hr = 60000.0 / np.clip(rr_ms, 1e-6, None)
    return np.array([
        hr.mean(), hr.std(),
        rr_ms.std(),                                     # SDNN
        np.sqrt((d ** 2).mean()),                        # RMSSD
        (np.abs(d) > 20).mean() * 100.0,                 # pNN20
        (np.abs(d) > 50).mean() * 100.0,                 # pNN50
        rr_ms.mean(), rr_ms.min(), rr_ms.max(),
        float(len(rr_ms) + 1),
    ])


def _ecg_block(ecg: np.ndarray, fs: int) -> np.ndarray:
    def run():
        clean = nk.ecg_clean(ecg, sampling_rate=fs)
        _, info = nk.ecg_peaks(clean, sampling_rate=fs)
        peaks = np.asarray(info["ECG_R_Peaks"], dtype=float)
        if len(peaks) < 4:
            return np.zeros(len(ECG_FEATURES))
        rr_ms = np.diff(peaks) / fs * 1000.0
        # physiologically implausible intervals -> artefacts, drop them
        rr_ms = rr_ms[(rr_ms > 300) & (rr_ms < 2000)]
        return _hrv_time(rr_ms)
    return _safe(run, len(ECG_FEATURES))


def _eda_block(eda: np.ndarray, fs: int) -> np.ndarray:
    def run():
        clean = nk.eda_clean(eda, sampling_rate=fs)
        comp = nk.eda_phasic(clean, sampling_rate=fs)
        tonic = comp["EDA_Tonic"].values
        phasic = comp["EDA_Phasic"].values

        t = np.arange(len(tonic), dtype=float) / fs
        slope = np.polyfit(t, tonic, 1)[0] if len(tonic) > 2 else 0.0

        n_pk, amp_mean, amp_max = 0.0, 0.0, 0.0
        try:
            _, pinfo = nk.eda_peaks(phasic, sampling_rate=fs)
            amps = np.asarray(pinfo.get("SCR_Amplitude", []), dtype=float)
            amps = amps[np.isfinite(amps)]
            if len(amps):
                n_pk, amp_mean, amp_max = float(len(amps)), amps.mean(), amps.max()
        except Exception:
            pass

        return np.array([tonic.mean(), tonic.std(), slope,
                         tonic.max() - tonic.min(),
                         n_pk, amp_mean, amp_max, phasic.std()])
    return _safe(run, len(EDA_FEATURES))


def _rsp_block(rsp: np.ndarray, fs: int) -> np.ndarray:
    def run():
        clean = nk.rsp_clean(rsp, sampling_rate=fs)
        _, info = nk.rsp_peaks(clean, sampling_rate=fs)
        peaks = np.asarray(info.get("RSP_Peaks", []), dtype=float)
        if len(peaks) < 3:
            return np.zeros(len(RSP_FEATURES))

        bb_ms = np.diff(peaks) / fs * 1000.0            # breath-to-breath
        rate = 60000.0 / np.clip(bb_ms, 1e-6, None)
        amps = clean[np.asarray(peaks, dtype=int)]
        rrv = np.sqrt((np.diff(bb_ms) ** 2).mean()) if len(bb_ms) > 1 else 0.0
        return np.array([rate.mean(), rate.std(), amps.mean(), amps.std(),
                         rrv, float(len(peaks))])
    return _safe(run, len(RSP_FEATURES))


def _raw_block(ecg: np.ndarray, eda: np.ndarray, rsp: np.ndarray, fs: int) -> np.ndarray:
    """Cheap descriptors that survive even when peak detection fails entirely,
    so a window is never wholly uninformative."""
    def run():
        t = np.arange(len(eda), dtype=float) / fs
        slope = np.polyfit(t, eda, 1)[0] if len(eda) > 2 else 0.0
        return np.array([ecg.std(), eda.mean(), slope, rsp.std()])
    return _safe(run, len(RAW_FEATURES))


def extract_window(ecg: np.ndarray, eda: np.ndarray, rsp: np.ndarray,
                   fs: int) -> np.ndarray:
    """Feature vector [N_FEATURES] for one window of 500 Hz signal."""
    if not HAVE_NK:
        raise RuntimeError("neurokit2 is required for A1 physio features")
    return np.concatenate([
        _ecg_block(ecg, fs),
        _eda_block(eda, fs),
        _rsp_block(rsp, fs),
        _raw_block(ecg, eda, rsp, fs),
    ]).astype(np.float32)


def extract_recording(sig: np.ndarray, fs: int, bounds: list[tuple[int, int]],
                      max_windows: int) -> np.ndarray:
    """Per-window features for one recording. -> [max_windows, N_FEATURES]

    `sig` is [T, 3] = (ECG, EDA, RR) at `fs`; `bounds` are sample indices into
    it, one pair per window, already aligned to the cache's window grid.
    """
    out = np.zeros((max_windows, N_FEATURES), np.float32)
    for i, (a, b) in enumerate(bounds[:max_windows]):
        w = sig[a:b]
        if len(w) < fs:                                  # < 1 s -> unusable
            continue
        out[i] = extract_window(w[:, 0], w[:, 1], w[:, 2], fs)
    return out
