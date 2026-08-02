"""A1 - build the per-window domain-physio-feature cache.

Writes `data/physfeat_{tag}/{key}.npz` holding `feat [max_windows, N_FEATURES]`
for every recording in the manifest. Features come from the ORIGINAL 500 Hz
signal but land on exactly the window grid the main cache uses, so the two are
index-aligned and the sequence model is unchanged.

    python -m src.physfeat_cache            # small config
    python run_full.py --stage physfeat     # full corpus

Cheap compared to the main preprocess stage (~5 min single-core for 700
recordings, no video decode), and cached per recording, so re-running only
picks up what is missing.
"""
from __future__ import annotations

import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .config import Config
from . import signals
from .physio_features import extract_recording, FEATURE_NAMES, N_FEATURES
from .preprocess import _paths, _window_bounds


def window_bounds_500(cfg: Config, n_samples_at_fs: int) -> list[tuple[int, int]]:
    """Cache window grid (computed at cfg.physio_fs) expressed in raw-rate samples.

    Must mirror `preprocess.process_recording` exactly, or feature window i would
    not describe the same seconds as cache window i.
    """
    b = _window_bounds(n_samples_at_fs, cfg.window_samples_physio,
                       int(cfg.hop_sec * cfg.physio_fs), cfg.max_windows)
    r = cfg.physio_fs_raw / cfg.physio_fs
    return [(int(a * r), int(b_ * r)) for a, b_ in b]


def process_recording(row: pd.Series, cfg: Config) -> np.ndarray:
    p = _paths(row["subject"], row["task"])["physio"]
    # read twice: once at cache rate to reproduce the grid, once raw for features
    sig_fs = signals.read_physio(p, cfg.physio_fs_raw, cfg.physio_fs, cfg.physio_channels)
    sig_raw = signals.read_physio(p, cfg.physio_fs_raw, cfg.physio_fs_raw, cfg.physio_channels)
    bounds = window_bounds_500(cfg, len(sig_fs))
    return extract_recording(sig_raw, cfg.physio_fs_raw, bounds, cfg.max_windows)


def _subject_job(payload: tuple) -> list[tuple]:
    cfg, rows, todo = payload
    out = []
    for row in pd.DataFrame(rows).to_dict("records"):
        if row["key"] not in todo:
            continue
        try:
            feat = process_recording(pd.Series(row), cfg)
            np.savez_compressed(cfg.physfeat_dir / f"{row['key']}.npz", feat=feat)
            nz = int((feat != 0).any(axis=1).sum())
            out.append((row["key"], nz, None))
        except Exception:
            out.append((row["key"], 0, traceback.format_exc()))
    return out


def main(cfg: Config | None = None, force: bool = False,
         workers: int | None = None) -> None:
    cfg = cfg or Config()
    man = pd.read_csv(cfg.manifest_path)
    cfg.physfeat_dir.mkdir(parents=True, exist_ok=True)

    todo = {r["key"] for _, r in man.iterrows()
            if force or not (cfg.physfeat_dir / f"{r['key']}.npz").exists()}
    print(f"[physfeat] {len(todo)}/{len(man)} recordings -> {cfg.physfeat_dir}")
    print(f"[physfeat] {N_FEATURES} features: {', '.join(FEATURE_NAMES)}")
    if not todo:
        return

    jobs = [(cfg, g.to_dict("records"), todo & set(g["key"]))
            for _, g in man.groupby("subject") if todo & set(g["key"])]
    workers = workers if workers is not None else min(6, os.cpu_count() or 1)
    workers = max(1, min(workers, len(jobs)))
    print(f"[physfeat] {len(jobs)} subjects | {workers} worker(s)")

    done = failed = 0

    def _report(batch):
        nonlocal done, failed
        for key, nz, err in batch:
            done += 1
            if err is not None:
                failed += 1
                print(f"[{done:3d}/{len(todo)}] {key} FAILED\n{err}", file=sys.stderr)
            elif done % 50 == 0 or done == len(todo):
                print(f"[{done:3d}/{len(todo)}] {key:<22} nonzero_windows={nz}", flush=True)

    if workers == 1:
        for j in jobs:
            _report(_subject_job(j))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for batch in ex.map(_subject_job, jobs):
                _report(batch)

    print(f"[physfeat] done | {done - failed} ok, {failed} failed")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
