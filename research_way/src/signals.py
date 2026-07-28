"""Low-level signal readers / front ends (no librosa, no OpenFace required).

physio : 500 Hz ECG/EDA/RR .txt  -> resampled float32 [T, 3]
audio  : 16 kHz mono .wav        -> log-mel spectrogram [frames, n_mels]
video  : 5 fps .mp4              -> grayscale face crops  [frames, S, S]

The audio and video front ends are deliberately dependency-light stand-ins for
the wav2vec2 / OpenFace-AU front ends named in the planning doc. Both are
isolated here so they can be swapped without touching the model code:
  * `mel_spectrogram` -> replace with a wav2vec2 feature extractor
  * `video_face_frames` -> replace with per-frame OpenFace AU vectors
"""
from __future__ import annotations

import wave
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import resample_poly, stft

# --------------------------------------------------------------- physio


def read_physio(path: Path, fs_raw: int, fs_out: int, channels=("ECG", "EDA", "RR")) -> np.ndarray:
    """Read a StressID physiological .txt and resample to `fs_out`. -> [T, C]"""
    arr = np.genfromtxt(path, delimiter=",", names=True)
    sig = np.stack([np.asarray(arr[c], dtype=np.float64) for c in channels], axis=-1)
    sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)

    g = np.gcd(fs_out, fs_raw)
    sig = resample_poly(sig, fs_out // g, fs_raw // g, axis=0)
    return sig.astype(np.float32)


# ---------------------------------------------------------------- audio


def read_wav(path: Path, expect_sr: int) -> np.ndarray:
    """Mono float32 waveform in [-1, 1]. Uses stdlib `wave` (no librosa)."""
    with wave.open(str(path), "rb") as w:
        sr, n_ch, sw, n = (w.getframerate(), w.getnchannels(),
                           w.getsampwidth(), w.getnframes())
        raw = w.readframes(n)
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[sw]
    x = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    x = (x - 128.0) / 128.0 if sw == 1 else x / float(np.iinfo(dtype).max)
    if n_ch > 1:
        x = x.reshape(-1, n_ch).mean(axis=1)
    if sr != expect_sr:
        g = np.gcd(expect_sr, sr)
        x = resample_poly(x, expect_sr // g, sr // g)
    return x.astype(np.float32)


def _mel_filterbank(n_mels: int, n_fft: int, sr: int) -> np.ndarray:
    """Slaney-style triangular mel filterbank -> [n_mels, n_fft//2 + 1]."""
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    f_min, f_max = 20.0, sr / 2.0
    pts = mel2hz(np.linspace(hz2mel(f_min), hz2mel(f_max), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        lo, ctr, hi = bins[i], bins[i + 1], bins[i + 2]
        if ctr == lo:
            ctr = lo + 1
        if hi == ctr:
            hi = ctr + 1
        hi = min(hi, fb.shape[1] - 1)
        for k in range(lo, ctr):
            fb[i, k] = (k - lo) / max(ctr - lo, 1)
        for k in range(ctr, hi):
            fb[i, k] = (hi - k) / max(hi - ctr, 1)
    return fb


_FB_CACHE: dict = {}


def mel_spectrogram(x: np.ndarray, sr: int, n_fft: int, hop: int, n_mels: int) -> np.ndarray:
    """log-mel spectrogram -> [frames, n_mels], float32."""
    key = (n_mels, n_fft, sr)
    if key not in _FB_CACHE:
        _FB_CACHE[key] = _mel_filterbank(n_mels, n_fft, sr)
    fb = _FB_CACHE[key]

    _, _, Z = stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop,
                   boundary=None, padded=False)
    power = (np.abs(Z) ** 2).astype(np.float32)          # [freq, frames]
    mel = fb @ power
    return np.log(mel + 1e-8).T.astype(np.float32)       # [frames, n_mels]


# ---------------------------------------------------------------- video

_CASCADE = None


def _cascade() -> cv2.CascadeClassifier:
    global _CASCADE
    if _CASCADE is None:
        _CASCADE = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _CASCADE


def video_face_frames(path: Path, fps_target: float, size: int) -> tuple[np.ndarray, float]:
    """Decode an mp4, crop the face, return (frames [N, size, size] float32 in
    [0,1], effective_fps). The face box is detected once on a mid-clip frame and
    reused (the subject is seated and static), which keeps decoding cheap."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(int(round(src_fps / fps_target)), 1)

    # --- locate the face on a mid-clip frame
    box = None
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(n_total // 2, 0))
    ok, probe = cap.read()
    if ok:
        gray = cv2.cvtColor(probe, cv2.COLOR_BGR2GRAY)
        faces = _cascade().detectMultiScale(gray, 1.15, 5, minSize=(80, 80))
        if len(faces):
            box = max(faces, key=lambda b: b[2] * b[3])
    if box is None:                                   # centre-crop fallback
        h, w = (probe.shape[:2] if ok else (720, 1280))
        s = int(min(h, w) * 0.6)
        box = (w // 2 - s // 2, h // 2 - s // 2, s, s)
    x, y, bw, bh = [int(v) for v in box]
    pad = int(0.15 * max(bw, bh))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frames, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            h, w = frame.shape[:2]
            crop = frame[max(y - pad, 0): min(y + bh + pad, h),
                         max(x - pad, 0): min(x + bw + pad, w)]
            if crop.size == 0:
                crop = frame
            g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
            frames.append(g.astype(np.float32) / 255.0)
        idx += 1
    cap.release()
    if not frames:
        return np.zeros((0, size, size), np.float32), fps_target
    return np.stack(frames), src_fps / step
