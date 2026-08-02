"""Torch dataset over the cached window sequences.

One item = one (subject, task) recording = a sequence of windows with a
per-window x per-modality availability mask. Padded windows are masked out
everywhere, so the model never sees them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .config import Config, DATA_DIR


class StressIDWindows(Dataset):
    def __init__(self, manifest: pd.DataFrame, cfg: Config,
                 subjects: list[str] | None = None,
                 norm: dict | None = None,
                 subject_vocab: dict | None = None):
        self.cfg = cfg
        man = manifest if subjects is None else \
            manifest[manifest["subject"].isin(subjects)]
        self.man = man.reset_index(drop=True)
        self.norm = norm or {}
        self.subject_vocab = subject_vocab or {
            s: i for i, s in enumerate(sorted(manifest["subject"].unique()))}
        self._cache: dict[str, dict] = {}

    def __len__(self) -> int:
        return len(self.man)

    def _load(self, key: str) -> dict:
        if key not in self._cache:
            with np.load(self.cfg.cache_dir / f"{key}.npz") as z:
                d = {k: z[k] for k in z.files}
            if self.cfg.physio_mode == "features":       # A1
                with np.load(self.cfg.physfeat_dir / f"{key}.npz") as z:
                    d["physio_feat"] = z["feat"]
            self._cache[key] = d
        return self._cache[key]

    def __getitem__(self, i: int) -> dict:
        r = self.man.iloc[i]
        d = self._load(r["key"])
        n = int(d["n_windows"])

        win_mask = np.zeros(self.cfg.max_windows, np.float32)
        win_mask[:n] = 1.0
        mod_mask = d["mask"].astype(np.float32) * win_mask[:, None]

        physio = d["physio"].astype(np.float32)
        audio = d["audio"].astype(np.float32)
        video = d["video"].astype(np.float32)

        # dataset-level standardisation for the two spectral/pixel modalities
        if "audio_mean" in self.norm:
            audio = (audio - self.norm["audio_mean"]) / self.norm["audio_std"]
        if "video_mean" in self.norm:
            video = (video - self.norm["video_mean"]) / self.norm["video_std"]
        audio *= mod_mask[:, 1][:, None, None]
        video *= mod_mask[:, 2][:, None, None, None]

        return {
            "physio": torch.from_numpy(physio),
            "audio": torch.from_numpy(audio),
            "video": torch.from_numpy(video),
            "mod_mask": torch.from_numpy(mod_mask),
            "win_mask": torch.from_numpy(win_mask),
            "task_id": torch.tensor(int(r["task_id"]), dtype=torch.long),
            "subject_id": torch.tensor(self.subject_vocab[r["subject"]], dtype=torch.long),
            "y_binary": torch.tensor(float(r["binary"]), dtype=torch.float32),
            "y_affect3": torch.tensor(int(r["affect3"]), dtype=torch.long),
            "y_score": torch.tensor(float(r["stress_score"]) / 10.0, dtype=torch.float32),
            "key": r["key"],
        }


def compute_norm(manifest: pd.DataFrame, cfg: Config, subjects: list[str]) -> dict:
    """Audio/video standardisation stats from TRAIN subjects only (no leakage)."""
    a_sum = a_sq = a_n = 0.0
    v_sum = v_sq = v_n = 0.0
    for key in manifest[manifest["subject"].isin(subjects)]["key"]:
        with np.load(cfg.cache_dir / f"{key}.npz") as z:
            m, a, v = z["mask"], z["audio"].astype(np.float32), z["video"].astype(np.float32)
        ia, iv = m[:, 1] > 0, m[:, 2] > 0
        if ia.any():
            x = a[ia]
            a_sum += x.sum(); a_sq += (x ** 2).sum(); a_n += x.size
        if iv.any():
            x = v[iv]
            v_sum += x.sum(); v_sq += (x ** 2).sum(); v_n += x.size

    def _ms(s, sq, n):
        if n == 0:
            return 0.0, 1.0
        mean = s / n
        return float(mean), float(max(np.sqrt(max(sq / n - mean ** 2, 1e-8)), 1e-3))

    am, astd = _ms(a_sum, a_sq, a_n)
    vm, vstd = _ms(v_sum, v_sq, v_n)
    return {"audio_mean": am, "audio_std": astd, "video_mean": vm, "video_std": vstd}


def collate(batch: list[dict]) -> dict:
    out = {}
    for k in batch[0]:
        if k == "key":
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out


def make_loader(ds: Dataset, cfg: Config, shuffle: bool) -> DataLoader:
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                      collate_fn=collate, num_workers=0, drop_last=False)


def load_manifest(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or Config()
    return pd.read_csv(cfg.manifest_path)
