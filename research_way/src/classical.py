"""Classical / hybrid pipeline on the leakage-free subset.

Why this exists: on the 364 all-modality recordings under subject GroupKFold,
seven classical configurations beat the 1.2 M-parameter transformer
(video+SVC 0.544 vs. the model's 0.485). Deep representation learning on 448
training recordings is the wrong tool, so the search moved here.

Two design rules, both non-negotiable given what this project is about:

1.  **Nested CV.** Model/hyper-parameter selection runs on an INNER GroupKFold
    over the training subjects only. Picking a winner on the outer test fold and
    reporting that number is test-set fishing -- it would manufacture a "SOTA"
    the same way random splits manufactured the published one.

2.  **Subject-relative features are opt-in and reported separately.** They
    express each recording relative to that subject's own low-stress baseline
    (Relax / Breathing). Those baseline tasks have no audio, so they are NOT in
    the 364-recording evaluation subset -- no evaluated sample is touched, and
    no label is ever read. This is calibration, not leakage, but it changes the
    problem to a personalised setting and is labelled as such.

    python -m src.classical --run-name c1_baseline
    python -m src.classical --run-name c2_subjrel --subject-relative
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import (RandomForestClassifier, HistGradientBoostingClassifier,
                              ExtraTreesClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import Config, full_config
from .baselines import extract_static_features
from .video_features import extract_recording as extract_video_features
from .splits import load_splits
from .report import write_report

BASELINE_TASKS = ("Relax", "Breathing")     # low-stress, no audio -> outside the 364
K_GRID = (1, 2, 3, 5, 8)                    # candidate ensemble sizes, chosen by inner CV


# ------------------------------------------------------------------- features

def build_features(cfg: Config, man: pd.DataFrame) -> dict[str, np.ndarray]:
    """{block name -> [N, D]} for every recording in the manifest."""
    feats = extract_static_features(cfg, man)          # physio(raw)/audio/video

    pf = []
    for key in man["key"]:
        with np.load(cfg.physfeat_dir / f"{key}.npz") as z:
            f = z["feat"]
        v = f[(f != 0).any(axis=1)]
        if len(v) == 0:
            v = np.zeros((1, f.shape[1]), np.float32)
        # window-sequence summary: level, spread and extremes over the recording
        pf.append(np.concatenate([v.mean(0), v.std(0), v.min(0), v.max(0)]))
    feats["physfeat"] = np.nan_to_num(np.stack(pf))

    # C4: richer video descriptors (regional dynamics + LBP texture), cached
    vf_dir = cfg.physfeat_dir.parent / f"videofeat_{cfg.data_tag}"
    vf_dir.mkdir(parents=True, exist_ok=True)
    vf = []
    for key in man["key"]:
        cache = vf_dir / f"{key}.npz"
        if cache.exists():
            with np.load(cache) as z:
                f = z["feat"]
        else:
            with np.load(cfg.cache_dir / f"{key}.npz") as z:
                f = extract_video_features(z["video"], z["mask"],
                                           int(z["n_windows"]), cfg.max_windows)
            np.savez_compressed(cache, feat=f)
        v = f[(f != 0).any(axis=1)]
        if len(v) == 0:
            v = np.zeros((1, f.shape[1]), np.float32)
        vf.append(np.concatenate([v.mean(0), v.std(0)]))
    feats["videofeat"] = np.nan_to_num(np.stack(vf))

    return {k: np.nan_to_num(v.astype(np.float64)) for k, v in feats.items()}


def apply_subject_relative(X: np.ndarray, man: pd.DataFrame) -> np.ndarray:
    """Express each recording relative to its subject's own low-stress baseline.

    Baseline = mean feature vector over that subject's Relax/Breathing
    recordings. Those tasks carry no audio so they never enter the 364-recording
    evaluation subset; no label is consulted. Subjects lacking a baseline fall
    back to their own overall mean, which is still within-subject.
    """
    out = X.copy()
    is_base = man["task"].isin(BASELINE_TASKS).values
    for subj in man["subject"].unique():
        m = (man["subject"] == subj).values
        b = X[m & is_base]
        ref = b.mean(0) if len(b) else X[m].mean(0)
        out[m] = X[m] - ref
    return out


# --------------------------------------------------------------------- models

def model_zoo() -> dict:
    return {
        "logreg": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=5000, C=1.0,
                                                 class_weight="balanced")),
        "logreg_c01": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=5000, C=0.1,
                                                 class_weight="balanced")),
        "svc_rbf": lambda: make_pipeline(
            StandardScaler(), SVC(C=1.0, class_weight="balanced",
                                  probability=True, random_state=0)),
        "svc_lin": lambda: make_pipeline(
            StandardScaler(), SVC(kernel="linear", C=0.1, class_weight="balanced",
                                  probability=True, random_state=0)),
        "rf": lambda: RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                             class_weight="balanced",
                                             random_state=0, n_jobs=-1),
        "extratrees": lambda: ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2,
                                                   class_weight="balanced",
                                                   random_state=0, n_jobs=-1),
        "hgb": lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06,
                                                      max_leaf_nodes=15, random_state=0),
    }


FEATURE_SETS = {
    "videofeat": ["videofeat"],
    "videofeat+video": ["videofeat", "video"],
    "videofeat+audio": ["videofeat", "audio"],
    "videofeat+physfeat+audio": ["videofeat", "physfeat", "audio"],
    "video": ["video"],
    "audio": ["audio"],
    "physfeat": ["physfeat"],
    "physfeat+video": ["physfeat", "video"],
    "audio+video": ["audio", "video"],
    "physfeat+audio+video": ["physfeat", "audio", "video"],
    "all": ["physio", "physfeat", "audio", "video"],
    "all+videofeat": ["physio", "physfeat", "audio", "video", "videofeat"],
}


def _score(y_true, y_pred) -> dict:
    return {"macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "balanced_acc": balanced_accuracy_score(y_true, y_pred),
            "accuracy": accuracy_score(y_true, y_pred)}


C1_FEATURE_SETS = ("video", "audio", "physfeat", "physfeat+video", "audio+video",
                   "physfeat+audio+video", "all")   # frozen c1 protocol


def _fresh_folds(subj: np.ndarray, n_folds: int, seed: int) -> list[dict]:
    """New subject partitions, independent of data/splits_*.json.

    The five stored outer folds have been reused by every iteration, so the
    campaign maximum over them is optimistically biased (§15.6). Confirming on
    partitions never touched during the search is the only way to get a number
    that is not a search artefact.
    """
    rng = np.random.default_rng(seed)
    subjects = np.unique(subj)
    rng.shuffle(subjects)
    chunks = np.array_split(subjects, n_folds)
    return [{"fold": i, "test": list(c)} for i, c in enumerate(chunks)]


def run(cfg: Config, run_name: str, subject_relative: bool = False,
        ensemble_top_k: int = 3, notes: str = "", fixed_k: bool = False,
        splits_seed: int | None = None, c1_only: bool = False) -> dict:
    """`fixed_k=True` pins the ensemble size to `ensemble_top_k` a priori (the
    c1 protocol). `fixed_k=False` lets inner CV choose it per fold (c3), which
    §15.6 showed costs ~0.043 -- so comparisons must hold this constant."""
    t0 = time.time()
    man = pd.read_csv(cfg.manifest_path)
    feats = build_features(cfg, man)
    if subject_relative:
        feats = {k: apply_subject_relative(v, man) for k, v in feats.items()}

    comp = ((man.has_physio == 1) & (man.has_audio == 1) & (man.has_video == 1)).values
    y = man["binary"].values
    subj = man["subject"].values
    folds = (_fresh_folds(subj, cfg.n_folds, splits_seed)
             if splits_seed is not None else load_splits(cfg))
    zoo = model_zoo()
    global FEATURE_SETS
    _all_sets = FEATURE_SETS
    if c1_only:
        FEATURE_SETS = {k: v for k, v in FEATURE_SETS.items() if k in C1_FEATURE_SETS}

    outer_rows, inner_rows = [], []
    per_fold_choice = []

    for fo in folds:
        te = np.isin(subj, fo["test"]) & comp
        tr = (~np.isin(subj, fo["test"])) & comp
        if te.sum() == 0 or tr.sum() == 0:
            continue

        # ---- INNER selection: GroupKFold over TRAIN subjects only.
        # Build out-of-fold inner probabilities for every config, then use them
        # to choose BOTH the configs and the ensemble size k. Choosing k by
        # comparing outer scores afterwards is test-set fishing -- it is exactly
        # how 0.544 was manufactured in §15.3.
        tr_subj = subj[tr]
        n_inner = min(3, len(np.unique(tr_subj)))
        inner = GroupKFold(n_splits=n_inner)
        ytr = y[tr]
        oof = {}
        for fs_name, blocks in FEATURE_SETS.items():
            X = np.concatenate([feats[b] for b in blocks], axis=1)
            Xtr = X[tr]
            for m_name, mk in zoo.items():
                p_oof = np.full(len(ytr), np.nan)
                for itr, ite in inner.split(Xtr, ytr, groups=tr_subj):
                    try:
                        c = mk(); c.fit(Xtr[itr], ytr[itr])
                        p_oof[ite] = (c.predict_proba(Xtr[ite])[:, 1]
                                      if hasattr(c, "predict_proba")
                                      else c.decision_function(Xtr[ite]))
                    except Exception:
                        p_oof[ite] = 0.5
                p_oof = np.nan_to_num(p_oof, nan=0.5)
                oof[(fs_name, m_name)] = p_oof
                s = f1_score(ytr, (p_oof >= 0.5).astype(int),
                             average="macro", zero_division=0)
                inner_rows.append({"fold": fo["fold"], "features": fs_name,
                                   "model": m_name, "inner_macro_f1": s})

        scores = {k: f1_score(ytr, (v >= 0.5).astype(int), average="macro",
                              zero_division=0) for k, v in oof.items()}
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

        # pick ensemble size on the SAME inner OOF predictions (train-only)
        k_scores = {}
        for k in K_GRID:
            if k > len(ranked):
                continue
            pe = np.mean([oof[cfg] for cfg, _ in ranked[:k]], axis=0)
            k_scores[k] = f1_score(ytr, (pe >= 0.5).astype(int),
                                   average="macro", zero_division=0)
        best_k = ensemble_top_k if fixed_k else max(k_scores, key=k_scores.get)
        top = ranked[:best_k]
        per_fold_choice.append({
            "fold": fo["fold"], "chosen_k": best_k,
            "inner_score_at_k": k_scores[best_k],
            "k_grid_scores": " ".join(f"{k}:{v:.3f}" for k, v in sorted(k_scores.items())),
            "picked": " | ".join(f"{fs}/{mn}" for (fs, mn), _ in top)})

        # ---- OUTER evaluation: refit the inner winners on the full train fold
        probs = []
        for (fs_name, m_name), _ in top:
            X = np.concatenate([feats[b] for b in FEATURE_SETS[fs_name]], axis=1)
            c = zoo[m_name]()
            c.fit(X[tr], y[tr])
            p = (c.predict_proba(X[te])[:, 1] if hasattr(c, "predict_proba")
                 else c.decision_function(X[te]))
            probs.append(p)
        pred_ens = (np.mean(probs, axis=0) >= 0.5).astype(int)

        # single best inner model, reported for reference only -- NOT selectable
        (fs1, m1), _ = ranked[0]
        X1 = np.concatenate([feats[b] for b in FEATURE_SETS[fs1]], axis=1)
        c1 = zoo[m1](); c1.fit(X1[tr], y[tr])
        pred_single = c1.predict(X1[te])

        outer_rows.append({"fold": fo["fold"], "n_test": int(te.sum()),
                           "chosen_k": best_k,
                           **{f"ens_{k}": v for k, v in _score(y[te], pred_ens).items()},
                           **{f"single_{k}": v for k, v in _score(y[te], pred_single).items()}})

    outer = pd.DataFrame(outer_rows)
    headline = {
        "complete364_macro_f1": float(outer["ens_macro_f1"].mean()),
        "complete364_macro_f1_single_ref": float(outer["single_macro_f1"].mean()),
        "mean_chosen_k": float(outer["chosen_k"].mean()),
        "fixed_k": bool(fixed_k),
        "splits_seed": splits_seed if splits_seed is not None else "stored",
        "complete364_weighted_f1": float(outer["ens_weighted_f1"].mean()),
        "complete364_balanced_acc": float(outer["ens_balanced_acc"].mean()),
        "complete364_accuracy": float(outer["ens_accuracy"].mean()),
        "n_folds": int(len(outer)),
        "subject_relative": bool(subject_relative),
    }

    FEATURE_SETS = _all_sets
    inner_df = pd.DataFrame(inner_rows)
    top_inner = (inner_df.groupby(["features", "model"])["inner_macro_f1"]
                 .mean().reset_index()
                 .sort_values("inner_macro_f1", ascending=False).head(15))

    write_report(
        run_name=run_name,
        headline=headline,
        config={"subject_relative": subject_relative,
                "ensemble_top_k": ensemble_top_k, "fixed_k": fixed_k,
                "feature_sets": list(FEATURE_SETS),
                "models": list(zoo),
                "n_folds": cfg.n_folds, "data_tag": cfg.data_tag,
                "selection": "nested GroupKFold (inner selection on train subjects only)"},
        notes=notes,
        tables={"Per-fold outer results": outer.to_dict("records"),
                "Inner-CV model ranking (mean over folds)": top_inner.to_dict("records"),
                "Per-fold selection": per_fold_choice},
        duration_s=time.time() - t0,
    )
    return headline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--subject-relative", action="store_true")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--splits-seed", type=int, default=None,
                    help="generate FRESH subject partitions instead of the stored splits")
    ap.add_argument("--c1-only", action="store_true",
                    help="restrict to the frozen c1 feature-set list")
    ap.add_argument("--fixed-k", action="store_true",
                    help="pin ensemble size a priori (c1 protocol) instead of inner-CV selection")
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    h = run(full_config(), a.run_name, a.subject_relative, a.top_k, a.notes,
            a.fixed_k, a.splits_seed, a.c1_only)
    print(h)


if __name__ == "__main__":
    main()
