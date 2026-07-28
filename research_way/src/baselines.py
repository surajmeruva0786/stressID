"""Stage 0 / E1 - corrected classical baselines under the SAME GroupKFold splits.

Static hand-crafted features (whole-recording aggregation, i.e. exactly the
"one feature vector per task" regime the related work uses) fed to logistic
regression and random forest, run unimodal, feature-fusion and decision-fusion.

To quantify the leakage inflation the pipeline's Contribution #1 is about, every
baseline is ALSO run under a random (subject-ignoring) KFold. The gap between
the two columns is the inflation estimate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import Config
from .dataset import load_manifest
from .splits import load_splits


def _stats(x: np.ndarray, axis: int) -> np.ndarray:
    return np.concatenate([x.mean(axis), x.std(axis), np.percentile(x, 10, axis),
                           np.percentile(x, 90, axis)], axis=-1)


def extract_static_features(cfg: Config, man: pd.DataFrame) -> dict:
    """Whole-recording aggregation per modality -> {modality: [N, D]}, plus labels."""
    P, A, V = [], [], []
    for key in man["key"]:
        with np.load(cfg.cache_dir / f"{key}.npz") as z:
            m, ph, au, vi = z["mask"], z["physio"], z["audio"], z["video"]
            n = int(z["n_windows"])

        w = m[:n, 0] > 0
        p = ph[:n][w].reshape(-1, ph.shape[-1]) if w.any() else np.zeros((1, ph.shape[-1]), np.float32)
        # per-channel stats + first-difference energy (crude signal dynamics)
        P.append(np.concatenate([_stats(p, 0), np.abs(np.diff(p, axis=0)).mean(0)]))

        w = m[:n, 1] > 0
        a = au[:n][w].astype(np.float32).reshape(-1, au.shape[-1]) if w.any() \
            else np.zeros((1, au.shape[-1]), np.float32)
        A.append(np.concatenate([_stats(a, 0), np.full(1, float(w.any()))]))

        w = m[:n, 2] > 0
        v = vi[:n][w].astype(np.float32) if w.any() else np.zeros((1, 1, vi.shape[-2], vi.shape[-1]), np.float32)
        v = v.reshape(-1, vi.shape[-2] * vi.shape[-1])
        vm = v.mean(0)
        V.append(np.concatenate([[vm.mean(), vm.std()], np.abs(np.diff(v, axis=0)).mean(0)[:64],
                                 np.percentile(v, [10, 50, 90], axis=0).mean(1)]))

    return {"physio": np.nan_to_num(np.stack(P)),
            "audio": np.nan_to_num(np.stack(A)),
            "video": np.nan_to_num(np.stack(V))}


MODELS = {
    "logreg": lambda: make_pipeline(StandardScaler(),
                                    LogisticRegression(max_iter=2000, class_weight="balanced")),
    "rf": lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                         random_state=0, n_jobs=-1),
}


def _fit_eval(X: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray,
              name: str) -> tuple[np.ndarray, np.ndarray]:
    clf = MODELS[name]()
    clf.fit(X[tr], y[tr])
    return clf.predict_proba(X[te])[:, 1], y[te]


def run(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or Config()
    man = load_manifest()
    feats = extract_static_features(cfg, man)
    y = man["binary"].values
    subj = man["subject"].values

    folds = load_splits()
    grouped = [(np.where(~np.isin(subj, f["test"]))[0],
                np.where(np.isin(subj, f["test"]))[0]) for f in folds]
    kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    random_folds = list(kf.split(np.arange(len(man))))   # LEAKY on purpose (E1)

    configs = {
        "physio": ["physio"], "audio": ["audio"], "video": ["video"],
        "feature_fusion": ["physio", "audio", "video"],
    }

    rows = []
    for protocol, splits in (("groupkfold", grouped), ("random_kfold_LEAKY", random_folds)):
        for clf_name in MODELS:
            for cfg_name, mods in configs.items():
                X = np.concatenate([feats[m] for m in mods], axis=1)
                accs, f1s = [], []
                for tr, te in splits:
                    p, yt = _fit_eval(X, y, tr, te, clf_name)
                    accs.append(accuracy_score(yt, (p >= .5).astype(int)))
                    f1s.append(f1_score(yt, (p >= .5).astype(int), average="macro",
                                        zero_division=0))
                rows.append({"protocol": protocol, "clf": clf_name, "features": cfg_name,
                             "accuracy": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                             "f1_macro": float(np.mean(f1s)), "f1_std": float(np.std(f1s))})

            # decision fusion = mean of the three unimodal probabilities
            accs, f1s = [], []
            for tr, te in splits:
                ps = [_fit_eval(feats[m], y, tr, te, clf_name)[0] for m in MODALITIES_ORDER]
                p = np.mean(ps, axis=0)
                accs.append(accuracy_score(y[te], (p >= .5).astype(int)))
                f1s.append(f1_score(y[te], (p >= .5).astype(int), average="macro",
                                    zero_division=0))
            rows.append({"protocol": protocol, "clf": clf_name, "features": "decision_fusion",
                         "accuracy": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                         "f1_macro": float(np.mean(f1s)), "f1_std": float(np.std(f1s))})

    df = pd.DataFrame(rows)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.results_dir / "baselines.csv", index=False)

    # E1 inflation estimate: leaky protocol minus leakage-free protocol
    g = df[df.protocol == "groupkfold"].set_index(["clf", "features"])
    r = df[df.protocol == "random_kfold_LEAKY"].set_index(["clf", "features"])
    infl = (r[["accuracy", "f1_macro"]] - g[["accuracy", "f1_macro"]]).reset_index()
    infl.columns = ["clf", "features", "acc_inflation", "f1_inflation"]
    infl.to_csv(cfg.results_dir / "e1_leakage_inflation.csv", index=False)

    print("[baselines] GroupKFold (leakage-free):")
    print(g[["accuracy", "f1_macro"]].round(3).to_string())
    print("[baselines] leakage inflation (random KFold - GroupKFold):")
    print(infl.round(3).to_string(index=False))
    return df


MODALITIES_ORDER = ("physio", "audio", "video")

if __name__ == "__main__":
    run()
