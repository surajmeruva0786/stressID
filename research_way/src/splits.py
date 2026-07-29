"""Stage 0 - leakage-free evaluation protocol (Contribution #1).

Subject-level GroupKFold: no subject ever appears in both train and test. The
splits are computed once, written to disk, and reused by every experiment
(baselines, our model, ablations) so all numbers are apples-to-apples.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .config import Config, DATA_DIR


def make_splits(man: pd.DataFrame, cfg: Config) -> list[dict]:
    subjects = np.array(sorted(man["subject"].unique()))
    # group at subject level; StratifiedGroupKFold would balance labels too but
    # with 16 subjects plain GroupKFold keeps folds interpretable and equal-sized.
    gkf = GroupKFold(n_splits=cfg.n_folds)
    folds = []
    for k, (tr, te) in enumerate(gkf.split(subjects, groups=subjects)):
        train_subj = subjects[tr].tolist()
        test_subj = subjects[te].tolist()
        # carve a validation set out of TRAIN subjects (still no leakage)
        rng = np.random.default_rng(cfg.seed + k)
        n_val = max(1, len(train_subj) // 5)
        val_subj = sorted(rng.choice(train_subj, size=n_val, replace=False).tolist())
        train_subj = sorted(s for s in train_subj if s not in val_subj)
        folds.append({"fold": k, "train": train_subj, "val": val_subj, "test": test_subj})
    return folds


def assert_no_leakage(folds: list[dict]) -> None:
    for f in folds:
        tr, va, te = set(f["train"]), set(f["val"]), set(f["test"])
        assert not (tr & te), f"fold {f['fold']}: train/test subject overlap"
        assert not (tr & va), f"fold {f['fold']}: train/val subject overlap"
        assert not (va & te), f"fold {f['fold']}: val/test subject overlap"
    all_test = [s for f in folds for s in f["test"]]
    assert len(all_test) == len(set(all_test)), "a subject is tested twice"


def main(cfg: Config | None = None) -> list[dict]:
    cfg = cfg or Config()
    man = pd.read_csv(cfg.manifest_path)
    folds = make_splits(man, cfg)
    assert_no_leakage(folds)

    out = cfg.splits_path
    out.write_text(json.dumps(folds, indent=2), encoding="utf-8")
    for f in folds:
        te = man[man["subject"].isin(f["test"])]
        print(f"[splits] fold {f['fold']}: train={len(f['train'])} val={len(f['val'])} "
              f"test={len(f['test'])} subj | test n={len(te)} "
              f"pos={te['binary'].mean():.2f}")
    print(f"[splits] leakage check passed -> {out}")
    return folds


def load_splits() -> list[dict]:
    return json.loads((DATA_DIR / "splits_small.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
