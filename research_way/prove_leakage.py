"""Standalone, reproducible proof that subject leakage inflates StressID scores.

Run:  python prove_leakage.py

Four experiments, each answering a different objection:

  E-A  Protocol swap        Same features, same model, only the split rule changes.
                            Shows the symptom: scores rise under random splits.
  E-B  Negative control     `availability_only` carries no subject identity, so it
                            must NOT inflate. If it did, E-A would be measuring
                            some other artifact of the split (fold size, class
                            balance) rather than leakage.
  E-C  Identity probe       Can a classifier recognise WHO a recording came from,
                            using physiology alone? Establishes the mechanism:
                            physiology is a fingerprint.
  E-D  Identity oracle      Predict the label using NO signal at all - only "which
                            subject is this, and what did they usually report in
                            training?" This is the leak in its purest form. It is
                            available under a random split and impossible under
                            GroupKFold, where the subject is unseen by construction.

Outputs a CSV + JSON under reports/leakage_proof/ and prints a readable report.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.baselines import extract_static_features
from src.config import RESEARCH_ROOT, full_config
from src.dataset import load_manifest
from src.splits import load_splits

OUT_DIR = RESEARCH_ROOT / "reports" / "leakage_proof"
MODELS = {
    "logreg": lambda: make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
    "rf": lambda: RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=0, n_jobs=-1),
}


def _score(y_true, y_pred) -> dict:
    return {
        "acc": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


# ---------------------------------------------------------------- E-A / E-B
def protocol_swap(cfg, man, feats, y, subj) -> pd.DataFrame:
    """Identical features and model; only the split rule differs."""
    folds = load_splits(cfg)
    grouped = [(np.where(~np.isin(subj, f["test"]))[0],
                np.where(np.isin(subj, f["test"]))[0]) for f in folds]
    kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    random_folds = list(kf.split(np.arange(len(man))))   # LEAKY on purpose

    # sanity: confirm the two protocols really do differ in subject overlap
    overlap = []
    for name, splits in (("groupkfold", grouped), ("random_kfold_LEAKY", random_folds)):
        shared = [len(set(subj[tr]) & set(subj[te])) for tr, te in splits]
        overlap.append({"protocol": name, "mean_shared_subjects_train_test": float(np.mean(shared))})

    configs = {
        "physio": ["physio"], "audio": ["audio"], "video": ["video"],
        "feature_fusion": ["physio", "audio", "video"],
        "availability_only": ["availability"],      # E-B negative control
    }
    rows = []
    for protocol, splits in (("groupkfold", grouped), ("random_kfold_LEAKY", random_folds)):
        for clf_name, ctor in MODELS.items():
            for cfg_name, mods in configs.items():
                X = np.concatenate([feats[m] for m in mods], axis=1)
                per_fold = []
                for tr, te in splits:
                    clf = ctor()
                    clf.fit(X[tr], y[tr])
                    per_fold.append(_score(y[te], clf.predict(X[te])))
                row = {"protocol": protocol, "clf": clf_name, "features": cfg_name}
                for k in ("acc", "f1_macro", "f1_weighted"):
                    row[k] = float(np.mean([d[k] for d in per_fold]))
                    row[k + "_std"] = float(np.std([d[k] for d in per_fold]))
                rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(overlap)


# ---------------------------------------------------------------- E-C
def identity_probe(feats, subj) -> dict:
    """Can we tell WHO this is from physiology alone? Chance = 1 / n_subjects."""
    out = {}
    codes, uniq = pd.factorize(subj)
    # keep subjects with >= 5 recordings so stratified CV is well defined
    keep = np.isin(codes, [c for c, n in Counter(codes).items() if n >= 5])
    for mod in ("physio", "audio", "video"):
        X, yy = feats[mod][keep], codes[keep]
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        acc = cross_val_score(clf, X, yy, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                              scoring="accuracy", n_jobs=-1).mean()
        out[mod] = {"subject_id_accuracy": float(acc),
                    "chance": float(1.0 / len(np.unique(yy))),
                    "times_chance": float(acc * len(np.unique(yy)))}
    out["_n_subjects"] = int(len(np.unique(codes[keep])))
    out["_n_recordings"] = int(keep.sum())
    return out


# ---------------------------------------------------------------- E-D
def identity_oracle(cfg, man, y, subj) -> dict:
    """Predict the label from subject identity alone - zero signal content.

    For each test recording under a RANDOM split, look up that subject's other
    recordings that landed in train and predict their majority label. Under
    GroupKFold this is undefined by construction (subject never in train), which
    is the whole point: the random protocol hands the model a lookup table.
    """
    kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    scores, coverage = [], []
    global_major = int(Counter(y).most_common(1)[0][0])
    for tr, te in kf.split(np.arange(len(man))):
        table = {}
        for s in np.unique(subj[tr]):
            lbls = y[tr][subj[tr] == s]
            table[s] = int(Counter(lbls).most_common(1)[0][0])
        pred = np.array([table.get(s, global_major) for s in subj[te]])
        scores.append(_score(y[te], pred))
        coverage.append(float(np.mean([s in table for s in subj[te]])))

    # how self-consistent is a subject's labelling? upper bound on what the leak buys
    per_subj = [Counter(y[subj == s]).most_common(1)[0][1] / (subj == s).sum()
                for s in np.unique(subj)]
    return {
        "random_split_oracle": {k: float(np.mean([d[k] for d in scores]))
                                for k in ("acc", "f1_macro", "f1_weighted")},
        "test_subjects_seen_in_train": float(np.mean(coverage)),
        "groupkfold_oracle": "undefined - test subjects never appear in train",
        "mean_within_subject_label_consistency": float(np.mean(per_subj)),
    }


def main() -> None:
    cfg = full_config()          # 700 recordings / 64 subjects, not the 96-rec subset
    man = load_manifest(cfg)
    print(f"[proof] manifest: {len(man)} recordings, {man['subject'].nunique()} subjects")

    feats = extract_static_features(cfg, man)
    feats["availability"] = man[["has_physio", "has_audio", "has_video"]].values.astype(float)
    y = man["binary"].values
    subj = man["subject"].values

    df, overlap = protocol_swap(cfg, man, feats, y, subj)
    probe = identity_probe(feats, subj)
    oracle = identity_oracle(cfg, man, y, subj)

    # inflation table
    g = df[df.protocol == "groupkfold"].set_index(["clf", "features"])
    r = df[df.protocol == "random_kfold_LEAKY"].set_index(["clf", "features"])
    infl = (r[["acc", "f1_macro", "f1_weighted"]] - g[["acc", "f1_macro", "f1_weighted"]])
    infl = infl.rename(columns=lambda c: c + "_inflation").reset_index()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "protocol_swap.csv", index=False)
    infl.to_csv(OUT_DIR / "inflation.csv", index=False)
    (OUT_DIR / "mechanism.json").write_text(
        json.dumps({"subject_overlap": overlap.to_dict("records"),
                    "identity_probe": probe, "identity_oracle": oracle}, indent=2),
        encoding="utf-8")

    print("\n=== E-A/E-B  protocol swap (same features, same model) ===")
    print(df.set_index(["protocol", "clf", "features"])[
        ["acc", "f1_macro", "f1_weighted"]].round(3).to_string())
    print("\n--- subject overlap between train and test ---")
    print(overlap.to_string(index=False))
    print("\n--- inflation (leaky minus leakage-free) ---")
    print(infl.round(4).to_string(index=False))

    print("\n=== E-C  identity probe: can we recognise WHO this is? ===")
    print(f"({probe['_n_subjects']} subjects, {probe['_n_recordings']} recordings)")
    for mod in ("physio", "audio", "video"):
        p = probe[mod]
        print(f"  {mod:7s} subject-ID acc = {p['subject_id_accuracy']:.3f} "
              f"(chance {p['chance']:.3f}, {p['times_chance']:.1f}x chance)")

    print("\n=== E-D  identity oracle: label from subject identity, zero signal ===")
    o = oracle["random_split_oracle"]
    print(f"  random split : acc {o['acc']:.3f}  macroF1 {o['f1_macro']:.3f}  "
          f"weightedF1 {o['f1_weighted']:.3f}")
    print(f"  test subjects already seen in train: "
          f"{oracle['test_subjects_seen_in_train']:.1%}")
    print(f"  within-subject label consistency   : "
          f"{oracle['mean_within_subject_label_consistency']:.3f}")
    print(f"  GroupKFold: {oracle['groupkfold_oracle']}")
    print(f"\n[proof] artifacts -> {OUT_DIR}")


if __name__ == "__main__":
    main()
