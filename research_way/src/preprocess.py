"""Stage 1 - temporal windowing + caching.

Turns every (subject, task) recording in the manifest into one .npz holding an
aligned sequence of fixed-length windows for all three modalities plus a
per-window availability mask:

    physio [W, Tp, 3]   audio [W, Fa, n_mels]   video [W, Fv, S, S]
    mask   [W, 3]  (1 = modality present in this window)
    n_windows

Physio is z-scored PER SUBJECT (stats pooled over that subject's recordings),
not per recording: this removes between-subject offsets (supporting the
subject-invariance objective) while preserving the between-task differences
that actually carry the stress signal.
"""
from __future__ import annotations

import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .config import Config, DATA_DIR, DATASET_ROOT
from . import signals


def _paths(subject: str, task: str) -> dict:
    return {
        "physio": DATASET_ROOT / "Physiological" / subject / f"{subject}_{task}.txt",
        "audio": DATASET_ROOT / "Audio" / subject / f"{subject}_{task}.wav",
        "video": DATASET_ROOT / "Videos" / subject / f"{subject}_{task}.mp4",
    }


def _window_bounds(n_samples: int, win: int, hop: int, max_w: int) -> list[tuple[int, int]]:
    if n_samples < win:                      # short recording -> single padded window
        return [(0, n_samples)]
    starts = list(range(0, n_samples - win + 1, hop))
    return [(s, s + win) for s in starts[:max_w]]


def _fit_seq(x: np.ndarray, n_target: int) -> np.ndarray:
    """Pad (edge) or truncate the leading axis to exactly n_target."""
    if x.shape[0] == n_target:
        return x
    if x.shape[0] > n_target:
        return x[:n_target]
    pad = np.repeat(x[-1:] if len(x) else np.zeros((1,) + x.shape[1:], x.dtype),
                    n_target - x.shape[0], axis=0)
    return np.concatenate([x, pad], axis=0)


# ------------------------------------------------------- per-subject physio stats


def compute_physio_stats(man: pd.DataFrame, cfg: Config) -> dict[str, np.ndarray]:
    stats = {}
    for subj, grp in man.groupby("subject"):
        chunks = []
        for _, r in grp.iterrows():
            if not r["has_physio"]:
                continue
            sig = signals.read_physio(_paths(subj, r["task"])["physio"],
                                      cfg.physio_fs_raw, cfg.physio_fs,
                                      cfg.physio_channels)
            chunks.append(sig)
        if not chunks:
            continue
        allsig = np.concatenate(chunks, axis=0)
        med = np.median(allsig, axis=0)
        iqr = np.subtract(*np.percentile(allsig, [75, 25], axis=0))
        stats[subj] = np.stack([med, np.maximum(iqr, 1e-6)])   # [2, C]
    return stats


# ------------------------------------------------------------------ per recording


def process_recording(row: pd.Series, cfg: Config, phys_stats: dict) -> dict:
    subj, task = row["subject"], row["task"]
    p = _paths(subj, task)
    W = cfg.max_windows
    Tp = cfg.window_samples_physio

    physio = np.zeros((W, Tp, len(cfg.physio_channels)), np.float32)
    audio = np.zeros((W, cfg.audio_frames, cfg.n_mels), np.float16)
    video = np.zeros((W, cfg.video_frames, cfg.video_face_size,
                      cfg.video_face_size), np.float16)
    mask = np.zeros((W, 3), np.float32)

    # ---- physio drives the window grid (it is the only always-present modality)
    sig = signals.read_physio(p["physio"], cfg.physio_fs_raw, cfg.physio_fs,
                              cfg.physio_channels)
    med, iqr = phys_stats[subj]
    sig = (sig - med) / iqr
    sig = np.clip(sig, -10.0, 10.0)

    bounds = _window_bounds(len(sig), Tp, int(cfg.hop_sec * cfg.physio_fs), W)
    n_win = len(bounds)
    for i, (a, b) in enumerate(bounds):
        physio[i] = _fit_seq(sig[a:b], Tp)
        mask[i, 0] = 1.0

    # window boundaries in SECONDS -> shared clock for the other modalities
    times = [(a / cfg.physio_fs, b / cfg.physio_fs) for a, b in bounds]

    # ---- audio
    if row["has_audio"]:
        try:
            wav = signals.read_wav(p["audio"], cfg.audio_sr)
            for i, (t0, t1) in enumerate(times):
                seg = wav[int(t0 * cfg.audio_sr): int(t1 * cfg.audio_sr)]
                if len(seg) < cfg.audio_sr:            # < 1 s of audio -> unusable
                    continue
                mel = signals.mel_spectrogram(seg, cfg.audio_sr, cfg.mel_win,
                                              cfg.mel_hop, cfg.n_mels)
                audio[i] = _fit_seq(mel, cfg.audio_frames).astype(np.float16)
                mask[i, 1] = 1.0
        except Exception as e:                          # keep the recording, drop audio
            print(f"  ! audio failed {subj}_{task}: {e}", file=sys.stderr)

    # ---- video
    if row["has_video"]:
        try:
            frames, fps = signals.video_face_frames(p["video"], cfg.video_fps_target,
                                                    cfg.video_face_size)
            for i, (t0, t1) in enumerate(times):
                a, b = int(t0 * fps), int(t1 * fps)
                seg = frames[a:b]
                if len(seg) < max(cfg.video_frames // 3, 2):
                    continue
                video[i] = _fit_seq(seg, cfg.video_frames).astype(np.float16)
                mask[i, 2] = 1.0
        except Exception as e:
            print(f"  ! video failed {subj}_{task}: {e}", file=sys.stderr)

    return {"physio": physio, "audio": audio, "video": video,
            "mask": mask, "n_windows": np.int64(n_win)}


def _subject_job(payload: tuple) -> list[tuple]:
    """Process one subject end to end: its physio stats, then its recordings.

    A subject is the natural unit of work because the normalisation stats are
    pooled over that subject's own recordings -- doing it this way reads each
    physio file once instead of twice, and makes the whole stage parallel.
    """
    cfg, rows, todo_keys = payload
    man = pd.DataFrame(rows)
    stats = compute_physio_stats(man, cfg)

    results = []
    for _, row in man.iterrows():
        if row["key"] not in todo_keys:
            continue
        try:
            d = process_recording(row, cfg, stats)
            np.savez_compressed(cfg.cache_dir / f"{row['key']}.npz", **d)
            cov = d["mask"][: int(d["n_windows"])].mean(axis=0)
            results.append((row["key"], int(d["n_windows"]), cov.tolist(), None))
        except Exception:
            results.append((row["key"], 0, [0.0, 0.0, 0.0], traceback.format_exc()))
    return results


def main(cfg: Config | None = None, force: bool = False,
         workers: int | None = None) -> None:
    cfg = cfg or Config()
    man = pd.read_csv(cfg.manifest_path)
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    todo = {r["key"] for _, r in man.iterrows()
            if force or not (cfg.cache_dir / f"{r['key']}.npz").exists()}
    print(f"[preprocess] {len(todo)}/{len(man)} recordings to process "
          f"-> {cfg.cache_dir}")
    if not todo:
        return

    jobs = []
    for _, grp in man.groupby("subject"):
        keys = todo & set(grp["key"])
        if keys:
            jobs.append((cfg, grp.to_dict("records"), keys))

    workers = workers if workers is not None else min(6, os.cpu_count() or 1)
    workers = max(1, min(workers, len(jobs)))
    print(f"[preprocess] {len(jobs)} subjects | {workers} worker(s)")

    done = failed = 0

    def _report(batch: list[tuple]) -> None:
        nonlocal done, failed
        for key, n_win, cov, err in batch:
            done += 1
            if err is not None:
                failed += 1
                print(f"[{done:3d}/{len(todo)}] {key} FAILED\n{err}", file=sys.stderr)
            else:
                print(f"[{done:3d}/{len(todo)}] {key:<22} win={n_win:2d} "
                      f"cov(p/a/v)={cov[0]:.2f}/{cov[1]:.2f}/{cov[2]:.2f}", flush=True)

    if workers == 1:
        for job in jobs:
            _report(_subject_job(job))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for batch in ex.map(_subject_job, jobs):
                _report(batch)

    print(f"[preprocess] cache -> {cfg.cache_dir} | {done - failed} ok, {failed} failed")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
