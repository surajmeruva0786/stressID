"""Stage 0a - build the SMALL subset manifest.

Selects a handful of subjects and tasks out of the full StressID corpus and
writes one manifest row per (subject, task) recording, including which
modalities physically exist on disk. Everything downstream reads this manifest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, DATASET_ROOT, DATA_DIR


def _load_labels() -> pd.DataFrame:
    lab = pd.read_csv(DATASET_ROOT / "labels.csv")
    lab = lab.rename(columns={"subject/task": "key",
                              "binary-stress": "binary",
                              "affect3-class": "affect3"})
    lab["subject"] = lab["key"].str.split("_").str[0]
    lab["task"] = lab["key"].str.split("_").str[1]
    return lab


def _load_stress_scores() -> pd.DataFrame:
    """self_assessments.csv is `;`-separated with tasks as rows, subjects as cols."""
    sa = pd.read_csv(DATASET_ROOT / "self_assessments.csv", sep=";")
    sa = sa.rename(columns={sa.columns[0]: "item"})
    rows = []
    for _, r in sa.iterrows():
        item = str(r["item"])
        if not item.endswith("_stress"):
            continue
        task = item[: -len("_stress")]
        for subj in sa.columns[1:]:
            val = pd.to_numeric(r[subj], errors="coerce")
            rows.append({"subject": subj, "task": task, "stress_score": val})
    return pd.DataFrame(rows)


def _modality_paths(subject: str, task: str) -> dict:
    return {
        "physio": DATASET_ROOT / "Physiological" / subject / f"{subject}_{task}.txt",
        "audio": DATASET_ROOT / "Audio" / subject / f"{subject}_{task}.wav",
        "video": DATASET_ROOT / "Videos" / subject / f"{subject}_{task}.mp4",
    }


def build_manifest(cfg: Config) -> pd.DataFrame:
    lab = _load_labels()
    scores = _load_stress_scores()
    lab = lab.merge(scores, on=["subject", "task"], how="left")

    lab = lab[lab["task"].isin(cfg.tasks)].copy()

    # availability flags
    for m in ("physio", "audio", "video"):
        lab[f"has_{m}"] = [
            _modality_paths(s, t)[m].exists() for s, t in zip(lab["subject"], lab["task"])
        ]

    # ---- subject selection -------------------------------------------------
    per_subj = lab.groupby("subject").agg(
        n_tasks=("task", "nunique"),
        physio=("has_physio", "sum"),
        audio=("has_audio", "sum"),
        video=("has_video", "sum"),
        pos_rate=("binary", "mean"),
    )
    complete = per_subj[per_subj["n_tasks"] == len(cfg.tasks)]
    if cfg.require_all_modalities:
        # audio only exists for speech tasks, so require it on those tasks only
        n_audio_tasks = sum(t not in ("Breathing", "Relax", "Video1", "Video2")
                            for t in cfg.tasks)
        complete = complete[(complete["physio"] == len(cfg.tasks))
                            & (complete["video"] == len(cfg.tasks))
                            & (complete["audio"] >= n_audio_tasks)]

    # Deterministic, label-stratified pick: sort by how balanced the subject's
    # own binary labels are, then take a seeded random draw from the top pool.
    rng = np.random.default_rng(cfg.seed)
    pool = complete.copy()
    pool["balance"] = (pool["pos_rate"] - 0.5).abs()
    pool = pool.sort_values(["balance", "pos_rate"])
    take_from = pool.index.tolist()[: max(cfg.n_subjects * 2, cfg.n_subjects)]
    chosen = sorted(rng.choice(take_from, size=min(cfg.n_subjects, len(take_from)),
                               replace=False).tolist())

    man = lab[lab["subject"].isin(chosen)].copy()
    man = man.sort_values(["subject", "task"]).reset_index(drop=True)
    man["stress_score"] = man["stress_score"].fillna(man["stress_score"].mean())
    man["task_id"] = pd.Categorical(man["task"], categories=list(cfg.tasks)).codes
    return man[["key", "subject", "task", "task_id", "binary", "affect3",
                "stress_score", "has_physio", "has_audio", "has_video"]]


def main(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or Config()
    man = build_manifest(cfg)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "manifest_small.csv"
    man.to_csv(out, index=False)

    print(f"[subset] {len(man)} recordings | {man['subject'].nunique()} subjects "
          f"| {man['task'].nunique()} tasks -> {out}")
    print(f"[subset] binary balance: {man['binary'].mean():.3f}  "
          f"affect3: {man['affect3'].value_counts().to_dict()}")
    print(f"[subset] modality coverage: "
          f"physio={man['has_physio'].mean():.2f} "
          f"audio={man['has_audio'].mean():.2f} "
          f"video={man['has_video'].mean():.2f}")
    return man


if __name__ == "__main__":
    main()
