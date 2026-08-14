"""SOTA campaign runner — paper-comparable (subject-shared) protocol.

WHAT THIS IS
------------
The StressID origin paper evaluates with a random 80/20 split over recordings,
so the same participant contributes rows to both train and test. We showed in
../../LEAKY_PROTOCOL.md that this inflates scores. This module deliberately
reproduces that protocol so our numbers sit on the same axis as the published
ones — and then pushes it as far as it will go.

Two tracks now exist in this repo and they must never be averaged together:

  reports/       subject GroupKFold  — honest generalisation to a new person
  reports_sota/  subject-shared CV   — comparable to the published 0.72

PROTOCOL DETAIL
---------------
* Outer: repeated stratified K-fold over *recordings* (subject-shared), seeded.
* Inner: stratified K-fold over the training rows only. Everything that gets
  chosen — which (features, view, model) candidates enter the ensemble, their
  weights, and the decision threshold — is chosen on inner out-of-fold
  predictions. The outer test fold is touched exactly once, for scoring.
  Even in a leaky protocol, selecting on the test fold would be a *second*,
  separate error, and it is the one that makes results irreproducible.
* Ensembling: bagged greedy selection with replacement (Caruana et al. 2004),
  which is far more robust to selection overfitting than "average the top k".

    python -m src.sota --run-name s1_baseline
    python -m src.sota --run-name s2_rich --views raw,rel,z --repeats 2
"""
from __future__ import annotations

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import sota_features as SF
from . import sota_models as SM
from . import sota_report
from .config import Config, DATA_DIR, full_config
from .sota_env import HAVE_GPU, JOBS

warnings.filterwarnings("ignore")


# --------------------------------------------------------------- model zoo

def model_zoo(fast: bool = False, jobs: int | None = None) -> dict:
    """`jobs` is the thread budget for ONE model. When the inner sweep runs
    across `n_par` processes the caller divides the core budget by `n_par`,
    otherwise 4 processes x 6 threads oversubscribes a 12-core box that is
    already running two other training jobs."""
    jobs = JOBS if jobs is None else max(1, jobs)
    n_tree = 300 if fast else 600
    zoo = {
        "logreg": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced")),
        "logreg_l2w": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=4000, C=0.05, class_weight="balanced")),
        "svc_rbf": lambda: make_pipeline(
            StandardScaler(),
            SVC(C=3.0, gamma="scale", class_weight="balanced",
                probability=True, random_state=0)),
        "rf": lambda: RandomForestClassifier(
            n_estimators=n_tree, min_samples_leaf=1, max_features="sqrt",
            class_weight="balanced_subsample", random_state=0, n_jobs=jobs),
        "extratrees": lambda: ExtraTreesClassifier(
            n_estimators=n_tree, min_samples_leaf=1, max_features="sqrt",
            class_weight="balanced_subsample", random_state=0, n_jobs=jobs),
        "mlp": lambda: make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(256, 64), alpha=1e-3, max_iter=600,
                          early_stopping=True, n_iter_no_change=25, random_state=0)),
    }
    try:
        import lightgbm as lgb
        zoo["lgbm"] = lambda: lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=15,
            min_child_samples=10, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.5, reg_lambda=1.0, class_weight="balanced",
            random_state=0, n_jobs=jobs, verbose=-1)
    except Exception:
        pass
    try:
        import xgboost as xgb
        dev = "cuda" if HAVE_GPU else "cpu"
        zoo["xgb"] = lambda: xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.5, reg_lambda=1.0,
            min_child_weight=2, random_state=0, n_jobs=jobs,
            tree_method="hist", device=dev, eval_metric="logloss")
        if not fast:
            zoo["xgb_deep"] = lambda: xgb.XGBClassifier(
                n_estimators=800, learning_rate=0.03, max_depth=7,
                subsample=0.7, colsample_bytree=0.3, reg_lambda=3.0,
                min_child_weight=1, random_state=1, n_jobs=jobs,
                tree_method="hist", device=dev, eval_metric="logloss")
    except Exception:
        pass
    # CatBoost is deliberately absent. On this box the GPU is shared with the
    # user's other training jobs and a single 600-iteration CatBoost fit took
    # 143 s against XGBoost's 6 s for an equal test score (0.642 vs 0.650) --
    # it would have consumed ~90% of the sweep budget for no gain.
    return zoo


# ------------------------------------------------------------ feature sets

FEATURE_SETS = {
    "phys":        ["physfeat", "physraw"],
    "audio":       ["audio"],
    "video":       ["videostat", "videofeat"],
    "audio+video": ["audio", "videostat", "videofeat"],
    "phys+video":  ["physfeat", "physraw", "videostat", "videofeat"],
    "phys+audio":  ["physfeat", "physraw", "audio"],
    "all":         ["physfeat", "physraw", "audio", "videostat", "videofeat"],
    "all+avail":   ["physfeat", "physraw", "audio", "videostat", "videofeat", "avail"],
}


def build_matrices(feats: dict, man: pd.DataFrame, views: list[str],
                   feature_sets: list[str] | None = None) -> dict:
    """{(feature_set, view) -> [N, D]} — every design matrix a candidate can use."""
    viewed = {}
    for v in views:
        fn = SF.VIEWS[v]
        # `avail` is three bits about which sensors exist; centring it per
        # subject is meaningless, so it is passed through unchanged.
        viewed[v] = {k: (val if k == "avail" else fn(val, man))
                     for k, val in feats.items()}
    wanted = feature_sets or list(FEATURE_SETS)
    out = {}
    for fs in wanted:
        for v in views:
            out[(fs, v)] = np.concatenate([viewed[v][b] for b in FEATURE_SETS[fs]],
                                          axis=1)
    return out


# --------------------------------------------------------------- selection

def greedy_ensemble(oof: dict, y: np.ndarray, max_size: int = 25,
                    n_bags: int = 12, bag_frac: float = 0.5,
                    seed: int = 0) -> dict:
    """Bagged greedy selection with replacement (Caruana et al., 2004).

    Plain "average the top k by inner score" overfits the inner estimate badly
    when there are ~150 candidates and ~560 training rows: the top of the list
    is dominated by whichever candidates got lucky. Greedy selection *with
    replacement* instead grows an average one member at a time, always adding
    whichever candidate most improves the current blend, so correlated
    candidates stop helping once one of them is in. Bagging it over random
    halves of the candidate pool damps the remaining selection variance.

    Returns {candidate: weight} summing to 1.
    """
    keys = list(oof)
    rng = np.random.default_rng(seed)
    counts: dict = {}
    for b in range(n_bags):
        pool = list(rng.choice(len(keys), size=max(2, int(len(keys) * bag_frac)),
                               replace=False))
        pool = [keys[i] for i in pool]
        cur = np.zeros(len(y))
        n = 0
        for _ in range(max_size):
            best_k, best_s = None, -1.0
            for k in pool:
                p = (cur * n + oof[k]) / (n + 1)
                s = f1_score(y, (p >= 0.5).astype(int), average="macro",
                             zero_division=0)
                if s > best_s:
                    best_k, best_s = k, s
            cur = (cur * n + oof[best_k]) / (n + 1)
            n += 1
            counts[best_k] = counts.get(best_k, 0) + 1
    tot = sum(counts.values())
    return {k: v / tot for k, v in sorted(counts.items(), key=lambda kv: -kv[1])}


def prune_weights(weights: dict, cum_keep: float = 0.90) -> dict:
    """Keep the heaviest members covering `cum_keep` of the total weight.

    Bagged greedy returns one entry per (bag, step) selection, which in R2 came
    to 63-92 distinct members per fold. Most of that is a long tail picked once
    in one bag and carrying ~1% weight. In R1 that tail was heavy enough that
    the ensemble scored *below* its own best single member. Trimming to the
    members that actually carry the blend keeps the diversity and drops the
    noise; the survivors are renormalised so the blend stays a mean.
    """
    if not weights:
        return weights
    ordered = sorted(weights.items(), key=lambda kv: -kv[1])
    keep, run = [], 0.0
    for k, w in ordered:
        keep.append((k, w))
        run += w
        if run >= cum_keep:
            break
    tot = sum(w for _, w in keep)
    return {k: w / tot for k, w in keep}


def tune_threshold(p: np.ndarray, y: np.ndarray) -> float:
    """Decision threshold maximising macro F1 on inner OOF predictions.

    Macro F1 on a 47/53 split is sensitive to the operating point, and 0.5 is
    only optimal when the classifier is perfectly calibrated. Tuned on inner
    OOF, never on the test fold.
    """
    grid = np.arange(0.20, 0.81, 0.01)
    scores = [f1_score(y, (p >= t).astype(int), average="macro", zero_division=0)
              for t in grid]
    return float(grid[int(np.argmax(scores))])


# ----------------------------------------------------------------- metrics

def score_all(y, pred, prob=None) -> dict:
    d = {"macro_f1": f1_score(y, pred, average="macro", zero_division=0),
         "weighted_f1": f1_score(y, pred, average="weighted", zero_division=0),
         "balanced_acc": balanced_accuracy_score(y, pred),
         "accuracy": accuracy_score(y, pred)}
    if prob is not None and len(np.unique(y)) > 1:
        d["roc_auc"] = roc_auc_score(y, prob)
    return d


# -------------------------------------------------------------- experiment

def _inner_oof(cand, tr_i, yy, splits) -> tuple[str, np.ndarray]:
    """Inner out-of-fold probabilities for one candidate.

    Top-level (not a closure) so the process pool can pickle it. `splits` index
    into `tr_i`, which in turn indexes into the scope subset; candidates always
    receive scope-subset positions and the full scope label vector.
    """
    p = np.full(len(tr_i), 0.5)
    for itr, ite in splits:
        try:
            p[ite] = cand.fit_predict(tr_i[itr], tr_i[ite], yy)
        except Exception:
            p[ite] = 0.5
    return cand.name, np.nan_to_num(p, nan=0.5)


def _serial_sweep(cands: list, tr_i, yy, splits, verbose: bool) -> list:
    """Serial inner sweep with progress. On this box the parallel path is not
    always the fast path -- with the other training jobs resident there is well
    under a gigabyte free, and three sweep workers push the machine into swap.
    Progress output exists so a slow round is distinguishable from a hung one."""
    out = []
    t0 = time.time()
    for i, c in enumerate(cands):
        out.append(_inner_oof(c, tr_i, yy, splits))
        if verbose and (i + 1) % 8 == 0:
            el = time.time() - t0
            print(f"    [{i + 1}/{len(cands)}] {el:.0f}s elapsed, "
                  f"{el / (i + 1):.1f}s/candidate", flush=True)
    return out


def run_scope(cands: list, yy: np.ndarray, n_folds: int, repeats: int, seed: int,
              max_size: int, n_bags: int, n_par: int = 1, verbose: bool = True,
              inner_folds: int = 4) -> tuple[dict, list, list]:
    """Nested CV over the scope subset. Returns (headline, per-fold, inner rank).

    `cands` are candidate objects (see `sota_models`), already restricted to the
    scope's rows, and `yy` is the matching label vector.
    """
    rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=repeats,
                                   random_state=seed)
    fold_rows, inner_acc, choice_rows = [], {}, []
    by_name = {c.name: c for c in cands}

    for fi, (tr_i, te_i) in enumerate(rskf.split(np.zeros(len(yy)), yy)):
        t0 = time.time()
        ytr, yte = yy[tr_i], yy[te_i]
        inner = StratifiedKFold(n_splits=inner_folds, shuffle=True,
                                random_state=seed + fi)
        splits = list(inner.split(np.zeros(len(ytr)), ytr))

        # Candidates are independent, so the inner sweep is embarrassingly
        # parallel. It is also where ~95% of the wall clock goes: 64 candidates
        # x 4 inner fits per outer fold. Serial, one outer fold took >20 min on
        # this (shared) box.
        gpu_c = [c for c in cands if getattr(c, "uses_gpu", False)]
        cpu_c = [c for c in cands if not getattr(c, "uses_gpu", False)]
        results = []
        if n_par > 1 and cpu_c:
            from joblib import Parallel, delayed
            try:
                # max_nbytes=None: candidates already reference their matrix by
                # path, so there is nothing large left to auto-memmap, and
                # joblib's temp-file machinery would only add I/O.
                results += Parallel(n_jobs=n_par, backend="loky", verbose=0,
                                    max_nbytes=None)(
                    delayed(_inner_oof)(c, tr_i, yy, splits) for c in cpu_c)
            except Exception as e:
                # Memory pressure on this box is not under our control -- the
                # other training jobs grow and shrink. A round that dies at
                # fold 3 after an hour is worse than a slow round, so fall back
                # rather than lose the work.
                print(f"  [warn] parallel sweep failed ({type(e).__name__}); "
                      f"falling back to serial", flush=True)
                results += _serial_sweep(cpu_c, tr_i, yy, splits, verbose)
        else:
            results += _serial_sweep(cpu_c, tr_i, yy, splits, verbose)
        # GPU candidates stay serial: one 4 GB card, already shared with the
        # user's other training jobs.
        results += _serial_sweep(gpu_c, tr_i, yy, splits, verbose)

        oof = {}
        for name, p in results:
            oof[name] = p
            inner_acc.setdefault(name, []).append(
                f1_score(ytr, (p >= 0.5).astype(int), average="macro",
                         zero_division=0))

        weights = greedy_ensemble(oof, ytr, max_size=max_size, n_bags=n_bags,
                                  seed=seed + fi)
        blend_oof = sum(w * oof[k] for k, w in weights.items())
        thr = tune_threshold(blend_oof, ytr)

        # ---- outer: refit every selected member on the full training fold
        probs = np.zeros(len(te_i))
        for key, w in weights.items():
            probs += w * by_name[key].fit_predict(tr_i, te_i, yy)
        pred = (probs >= thr).astype(int)

        # Reported for reference only. The best inner candidate is NOT
        # selectable as the headline -- reporting whichever single model won on
        # a given fold is how a search manufactures a result it cannot repeat.
        best_single = max(oof, key=lambda k: f1_score(
            ytr, (oof[k] >= 0.5).astype(int), average="macro", zero_division=0))
        p_single = by_name[best_single].fit_predict(tr_i, te_i, yy)

        fold_rows.append({
            "fold": fi, "n_test": len(te_i), "thr": thr,
            "n_members": len(weights),
            **score_all(yte, pred, probs),
            **{f"single_{k}": v for k, v in
               score_all(yte, (p_single >= 0.5).astype(int), p_single).items()},
        })
        choice_rows.append({
            "fold": fi, "thr": round(thr, 3),
            "inner_blend_f1": round(f1_score(ytr, (blend_oof >= thr).astype(int),
                                             average="macro", zero_division=0), 4),
            "top_members": " | ".join(f"{k}:{w:.2f}"
                                      for k, w in list(weights.items())[:5]),
        })
        if verbose:
            print(f"  fold {fi}: macro_f1={fold_rows[-1]['macro_f1']:.4f} "
                  f"thr={thr:.2f} members={len(weights)} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(fold_rows)
    headline = {}
    for c in ("macro_f1", "weighted_f1", "balanced_acc", "accuracy", "roc_auc"):
        if c in df:
            headline[c] = float(df[c].mean())
            headline[f"{c}_std"] = float(df[c].std())
    headline["single_macro_f1"] = float(df["single_macro_f1"].mean())
    headline["n_eval_folds"] = int(len(df))
    headline["n_recordings"] = int(len(yy))

    rank = sorted(((k, float(np.mean(v))) for k, v in inner_acc.items()),
                  key=lambda kv: -kv[1])[:25]
    inner_rank = [{"candidate": k, "inner_macro_f1": v} for k, v in rank]
    return headline, fold_rows + choice_rows, inner_rank


_WIN_CACHE: dict = {}


def _window_candidates(cfg: Config, man, rows, scope: str, views: list[str],
                       windows: list[str], torch_archs: list[str], zoo: dict
                       ) -> list:
    """Window-level and GPU sequence candidates for one scope.

    `windows` names the views to build window tensors for ("raw", "z"); an
    empty list skips window modelling entirely, which is what the earlier
    rounds do so that the window contribution can be attributed rather than
    assumed.
    """
    if not windows and not torch_archs:
        return []
    from . import sota_windows as SW
    if "W" not in _WIN_CACHE:
        W, V, _ = SW.build_windows(cfg, man)
        _WIN_CACHE["W"], _WIN_CACHE["V"] = W, V
    W, V = _WIN_CACHE["W"], _WIN_CACHE["V"]

    out = []
    for view in windows or ["raw"]:
        Wv = W if view == "raw" else SW.subject_center(W, V, man, mode=view)
        Wr, Vr = Wv[rows], V[rows]
        if windows:
            # Only the tree learners get the window treatment: they are the
            # ones starved by 560 rows, and an SVC on ~9000 window rows costs
            # more than the whole rest of the sweep.
            for mname in ("lgbm", "xgb", "extratrees"):
                if mname not in zoo:
                    continue
                for agg in ("mean", "trimmed"):
                    out.append(SM.WindowCandidate(
                        f"win-{view}|{agg}|{mname}", Wr, Vr, zoo[mname], agg))
        for arch in torch_archs:
            c = SM.TorchWindowCandidate(f"seq-{view}|{arch}|torch", Wr, Vr,
                                        arch=arch)
            c.uses_gpu = True
            out.append(c)
    return out


def run(cfg: Config, run_name: str, views: list[str], repeats: int, seed: int,
        max_size: int, n_bags: int, fast: bool, notes: str,
        scopes: list[str], feature_sets: list[str] | None = None,
        models: list[str] | None = None, n_par: int = 4,
        windows: list[str] | None = None,
        torch_archs: list[str] | None = None,
        inner_folds: int = 4) -> dict:
    t0 = time.time()
    man = pd.read_csv(cfg.manifest_path)
    feats = SF.build(cfg, man)
    mats = build_matrices(feats, man, views, feature_sets)
    zoo = model_zoo(fast, max(1, JOBS // max(1, n_par)))
    windows = windows or []
    torch_archs = torch_archs or []
    if models:
        zoo = {k: v for k, v in zoo.items() if k in models}
    y = man["binary"].values
    complete = ((man.has_physio == 1) & (man.has_audio == 1) &
                (man.has_video == 1)).values

    print(f"[sota] {len(mats)} feature-matrices x {len(zoo)} models = "
          f"{len(mats) * len(zoo)} candidates; dims="
          f"{ {k[0]: v.shape[1] for k, v in mats.items() if k[1] == views[0]} }",
          flush=True)

    headline, tables = {}, {}
    for scope in scopes:
        rows = np.ones(len(man), bool) if scope == "all700" else complete
        print(f"[sota] scope={scope} n={rows.sum()}", flush=True)
        # Each design matrix is written once, restricted to this scope, and
        # referenced by path; candidates memory-map it. See TabularCandidate for
        # why (a by-value version OOM-killed the 4-process sweep).
        #
        # The directory is per-run. Sharing one across runs cost R1 a third
        # attempt: orphaned workers from a previous crashed run still held
        # memmaps on the .npy files, and Windows then refuses to reopen them for
        # writing (OSError 22). Per-run paths cannot collide with a dead run.
        mat_dir = DATA_DIR / f"sotamat_{cfg.data_tag}" / run_name
        mat_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        for (fs, view), X in mats.items():
            p = mat_dir / f"{scope}_{fs}_{view}.npy"
            np.save(p, np.ascontiguousarray(X[rows], dtype=np.float32))
            paths[(fs, view)] = p
        cands = [SM.TabularCandidate(f"{fs}|{view}|{mname}", p, mk)
                 for (fs, view), p in paths.items()
                 for mname, mk in zoo.items()]
        cands += _window_candidates(cfg, man, rows, scope, views, windows,
                                    torch_archs, zoo)
        h, folds, rank = run_scope(cands, y[rows], cfg.n_folds, repeats,
                                   seed, max_size, n_bags, n_par,
                                   inner_folds=inner_folds)
        headline.update({f"{scope}_{k}": v for k, v in h.items()})
        tables[f"Per-fold — {scope}"] = folds
        tables[f"Inner-CV candidate ranking — {scope}"] = rank

    config = {"protocol": "subject-shared RepeatedStratifiedKFold (paper-style)",
              "n_folds": cfg.n_folds, "repeats": repeats, "seed": seed,
              "views": views, "feature_sets": list(FEATURE_SETS),
              "models": list(zoo), "greedy_max_size": max_size,
              "greedy_bags": n_bags, "feature_version": SF.FEATURE_VERSION,
              "scopes": scopes, "fast": fast, "n_par": n_par, "jobs": JOBS,
              "window_views": windows, "torch_archs": torch_archs,
              "inner_folds": inner_folds,
              "selection": "bagged greedy w/ replacement on inner OOF; "
                           "threshold tuned on inner OOF"}
    sota_report.write(run_name, headline, config, notes, tables,
                      duration_s=time.time() - t0)

    # Best-effort cleanup: ~100 MB of scratch matrices per run. Failure is fine
    # (a straggling worker may still hold a mapping) -- the directory is
    # gitignored and the next run writes its own.
    import shutil
    shutil.rmtree(DATA_DIR / f"sotamat_{cfg.data_tag}" / run_name,
                  ignore_errors=True)
    return headline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--views", default="raw")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-size", type=int, default=25)
    ap.add_argument("--bags", type=int, default=12)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--scopes", default="all700,c364")
    ap.add_argument("--feature-sets", default="", help="comma list; default all")
    ap.add_argument("--models", default="", help="comma list; default all")
    ap.add_argument("--inner-folds", type=int, default=4)
    ap.add_argument("--windows", default="",
                    help="comma list of views for window-level candidates, e.g. raw,z")
    ap.add_argument("--torch-archs", default="",
                    help="comma list of GPU sequence archs: gru,attn,mean")
    ap.add_argument("--n-par", type=int, default=4,
                    help="candidates evaluated in parallel processes")
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    h = run(full_config(), a.run_name, a.views.split(","), a.repeats, a.seed,
            a.max_size, a.bags, a.fast, a.notes, a.scopes.split(","),
            [s for s in a.feature_sets.split(",") if s],
            [s for s in a.models.split(",") if s], a.n_par,
            [s for s in a.windows.split(",") if s],
            [s for s in a.torch_archs.split(",") if s], a.inner_folds)
    for k, v in h.items():
        print(f"  {k:<32} {v}")


if __name__ == "__main__":
    main()
