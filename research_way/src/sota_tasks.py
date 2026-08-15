"""The other two StressID targets: 3-class affect, and the 0-10 stress score.

The binary runner in `sota.py` produced the campaign's headline and is left
untouched. This module reuses its features, views, feature sets and candidate
classes, but runs its own nested-CV loop, because the three targets differ in
ways that are not parameterisable without making the binary path worse:

  binary       one probability per recording; a decision threshold tuned on
               inner OOF; blend = weighted mean of probabilities
  affect3      a [n, 3] probability matrix; no threshold, argmax instead;
               blend = weighted mean of matrices
  regression   a real prediction; no threshold, no probabilities; blend =
               weighted mean; scored by RMSE / MAE / correlation

Everything the binary campaign established carries over unchanged: the
subject-relative view, window-level candidates, bagged greedy selection on
inner OOF, 90% weight pruning, and the rule that nothing is ever chosen on the
outer test fold.

    python -m src.sota_tasks --task affect3 --run-name t1_affect3
    python -m src.sota_tasks --task regression --run-name t2_score
"""
from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold, StratifiedKFold, KFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from scipy import stats as sst

from . import sota_features as SF
from . import sota_models as SM
from . import sota_report
from .config import Config, DATA_DIR, full_config
from .sota import FEATURE_SETS, build_matrices, greedy_ensemble, prune_weights
from .sota_env import HAVE_GPU, JOBS

warnings.filterwarnings("ignore")

N_AFFECT = 3


# --------------------------------------------------------------- model zoos

def zoo_multiclass(jobs: int) -> dict:
    z = {
        "logreg": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced")),
        "svc_rbf": lambda: make_pipeline(
            StandardScaler(),
            SVC(C=3.0, class_weight="balanced", probability=True, random_state=0)),
        "rf": lambda: RandomForestClassifier(
            n_estimators=300, min_samples_leaf=1, max_features="sqrt",
            class_weight="balanced_subsample", random_state=0, n_jobs=jobs),
        "extratrees": lambda: ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=1, max_features="sqrt",
            class_weight="balanced_subsample", random_state=0, n_jobs=jobs),
        "mlp": lambda: make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(256, 64), alpha=1e-3, max_iter=600,
                          early_stopping=True, n_iter_no_change=25, random_state=0)),
    }
    try:
        import lightgbm as lgb
        z["lgbm"] = lambda: lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=15,
            min_child_samples=10, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.5, reg_lambda=1.0, class_weight="balanced",
            random_state=0, n_jobs=jobs, verbose=-1)
    except Exception:
        pass
    try:
        import xgboost as xgb
        z["xgb"] = lambda: xgb.XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8,
            colsample_bytree=0.5, reg_lambda=1.0, min_child_weight=2,
            random_state=0, n_jobs=jobs, tree_method="hist",
            device="cuda" if HAVE_GPU else "cpu",
            objective="multi:softprob", num_class=N_AFFECT)
    except Exception:
        pass
    return z


def zoo_regression(jobs: int) -> dict:
    z = {
        "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "svr_rbf": lambda: make_pipeline(StandardScaler(), SVR(C=3.0, epsilon=0.5)),
        "rf": lambda: RandomForestRegressor(
            n_estimators=300, min_samples_leaf=1, max_features="sqrt",
            random_state=0, n_jobs=jobs),
        "extratrees": lambda: ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=1, max_features="sqrt",
            random_state=0, n_jobs=jobs),
        "mlp": lambda: make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(256, 64), alpha=1e-3, max_iter=600,
                         early_stopping=True, n_iter_no_change=25, random_state=0)),
    }
    try:
        import lightgbm as lgb
        z["lgbm"] = lambda: lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.05, num_leaves=15,
            min_child_samples=10, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.5, reg_lambda=1.0, random_state=0,
            n_jobs=jobs, verbose=-1)
    except Exception:
        pass
    try:
        import xgboost as xgb
        z["xgb"] = lambda: xgb.XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8,
            colsample_bytree=0.5, reg_lambda=1.0, min_child_weight=2,
            random_state=0, n_jobs=jobs, tree_method="hist",
            device="cuda" if HAVE_GPU else "cpu")
    except Exception:
        pass
    return z


# -------------------------------------------------------------- candidates

class MultiTabular(SM.TabularCandidate):
    """Recording-level candidate emitting a [n, 3] probability matrix."""

    def fit_predict(self, tr, te, y) -> np.ndarray:
        X = self.X
        c = self.factory()
        c.fit(np.ascontiguousarray(X[tr]), y[tr])
        p = c.predict_proba(np.ascontiguousarray(X[te]))
        # A fold can be missing a class; realign to the global class order so
        # every candidate's matrix has the same columns and blending is valid.
        out = np.zeros((len(te), N_AFFECT))
        for j, cls in enumerate(c.classes_ if hasattr(c, "classes_")
                                else range(p.shape[1])):
            out[:, int(cls)] = p[:, j]
        return out


class RegTabular(SM.TabularCandidate):
    """Recording-level candidate emitting a real-valued prediction."""

    def fit_predict(self, tr, te, y) -> np.ndarray:
        X = self.X
        c = self.factory()
        c.fit(np.ascontiguousarray(X[tr]), y[tr])
        return np.asarray(c.predict(np.ascontiguousarray(X[te), ])) \
            if False else np.asarray(c.predict(np.ascontiguousarray(X[te])))


class MultiWindow(SM.WindowCandidate):
    """Window-level candidate, averaging [n_win, 3] up to one matrix per recording."""

    def fit_predict(self, tr, te, y) -> np.ndarray:
        Xtr, otr = self._expand(tr)
        c = self.factory()
        c.fit(Xtr, y[tr][otr])
        Xte, ote = self._expand(te)
        p = c.predict_proba(Xte)
        aligned = np.zeros((len(Xte), N_AFFECT))
        for j, cls in enumerate(c.classes_):
            aligned[:, int(cls)] = p[:, j]
        out = np.full((len(te), N_AFFECT), 1.0 / N_AFFECT)
        for j in range(len(te)):
            q = aligned[ote == j]
            if len(q):
                out[j] = q.mean(0)
        return out


class RegWindow(SM.WindowCandidate):
    def fit_predict(self, tr, te, y) -> np.ndarray:
        Xtr, otr = self._expand(tr)
        c = self.factory()
        c.fit(Xtr, y[tr][otr])
        Xte, ote = self._expand(te)
        p = np.asarray(c.predict(Xte))
        out = np.zeros(len(te))
        for j in range(len(te)):
            q = p[ote == j]
            out[j] = q.mean() if len(q) else float(y[tr].mean())
        return out


# ----------------------------------------------------------------- scoring

def score_affect3(y, prob) -> dict:
    pred = prob.argmax(1)
    return {"macro_f1": f1_score(y, pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y, pred, average="weighted", zero_division=0),
            "balanced_acc": balanced_accuracy_score(y, pred),
            "accuracy": accuracy_score(y, pred)}


def score_regression(y, pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    r_p = float(sst.pearsonr(y, pred)[0]) if np.std(pred) > 1e-9 else 0.0
    r_s = float(sst.spearmanr(y, pred)[0]) if np.std(pred) > 1e-9 else 0.0
    return {"rmse": rmse, "mae": float(mean_absolute_error(y, pred)),
            "pearson_r": r_p, "spearman_r": r_s,
            "r2": float(r2_score(y, pred))}


def _sel_score(task: str):
    """Objective the greedy ensemble maximises on inner OOF."""
    if task == "affect3":
        return lambda y, p: f1_score(y, p.argmax(1), average="macro", zero_division=0)
    # regression: maximise negative RMSE
    return lambda y, p: -float(np.sqrt(mean_squared_error(y, p)))


def greedy(oof: dict, y, task: str, max_size: int, n_bags: int,
           bag_frac: float = 0.5, seed: int = 0) -> dict:
    """Bagged greedy selection with replacement, task-aware.

    Same algorithm as the binary campaign's, but the blend and the objective
    depend on the target: probability matrices average elementwise and are
    scored by macro F1 over the argmax, real predictions average and are scored
    by negative RMSE.
    """
    sc = _sel_score(task)
    keys = list(oof)
    rng = np.random.default_rng(seed)
    counts: dict = {}
    for _ in range(n_bags):
        pool = [keys[i] for i in rng.choice(
            len(keys), size=max(2, int(len(keys) * bag_frac)), replace=False)]
        cur, n = None, 0
        for _ in range(max_size):
            best_k, best_s = None, -np.inf
            for k in pool:
                blend = oof[k] if cur is None else (cur * n + oof[k]) / (n + 1)
                s = sc(y, blend)
                if s > best_s:
                    best_k, best_s = k, s
            cur = oof[best_k] if cur is None else (cur * n + oof[best_k]) / (n + 1)
            n += 1
            counts[best_k] = counts.get(best_k, 0) + 1
    tot = sum(counts.values())
    return {k: v / tot for k, v in sorted(counts.items(), key=lambda kv: -kv[1])}


# -------------------------------------------------------------- experiment

def _inner_oof(cand, tr_i, y, splits, task: str):
    shape = (len(tr_i), N_AFFECT) if task == "affect3" else (len(tr_i),)
    p = np.full(shape, 1.0 / N_AFFECT if task == "affect3" else float(np.mean(y)))
    for itr, ite in splits:
        try:
            p[ite] = cand.fit_predict(tr_i[itr], tr_i[ite], y)
        except Exception:
            pass
    return cand.name, np.nan_to_num(p)


def run(cfg: Config, task: str, run_name: str, views: list[str], repeats: int,
        seed: int, max_size: int, n_bags: int, cum_keep: float,
        windows: list[str], inner_folds: int, notes: str, n_folds: int = 5) -> dict:
    t0 = time.time()
    man = pd.read_csv(cfg.manifest_path)
    feats = SF.build(cfg, man)
    mats = build_matrices(feats, man, views)
    jobs = JOBS
    zoo = zoo_multiclass(jobs) if task == "affect3" else zoo_regression(jobs)
    y = (man["affect3"].values.astype(int) if task == "affect3"
         else man["stress_score"].values.astype(float))

    mat_dir = DATA_DIR / f"sotamat_{cfg.data_tag}" / run_name
    mat_dir.mkdir(parents=True, exist_ok=True)
    TabCls = MultiTabular if task == "affect3" else RegTabular
    cands = []
    for (fs, view), X in mats.items():
        p = mat_dir / f"{fs}_{view}.npy"
        np.save(p, np.ascontiguousarray(X, dtype=np.float32))
        for mname, mk in zoo.items():
            cands.append(TabCls(f"{fs}|{view}|{mname}", p, mk))

    if windows:
        from . import sota_windows as SW
        W, V, _ = SW.build_windows(cfg, man)
        WinCls = MultiWindow if task == "affect3" else RegWindow
        for view in windows:
            Wv = W if view == "raw" else SW.subject_center(W, V, man, mode=view)
            for mname in ("lgbm", "xgb", "extratrees"):
                if mname in zoo:
                    cands.append(WinCls(f"win-{view}|mean|{mname}", Wv, V,
                                        zoo[mname], "mean"))

    print(f"[tasks:{task}] {len(cands)} candidates, n={len(y)}", flush=True)

    splitter = (RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=repeats,
                                        random_state=seed) if task == "affect3"
                else RepeatedKFold(n_splits=n_folds, n_repeats=repeats,
                                   random_state=seed))
    strat = y if task == "affect3" else np.zeros(len(y))
    by_name = {c.name: c for c in cands}
    fold_rows, choice_rows, inner_acc = [], [], {}

    for fi, (tr_i, te_i) in enumerate(splitter.split(np.zeros(len(y)), strat)):
        ft = time.time()
        inner = (StratifiedKFold(inner_folds, shuffle=True, random_state=seed + fi)
                 if task == "affect3"
                 else KFold(inner_folds, shuffle=True, random_state=seed + fi))
        splits = list(inner.split(np.zeros(len(tr_i)), y[tr_i]))

        oof = {}
        sc = _sel_score(task)
        for i, c in enumerate(cands):
            name, p = _inner_oof(c, tr_i, y, splits, task)
            oof[name] = p
            inner_acc.setdefault(name, []).append(sc(y[tr_i], p))
            if (i + 1) % 20 == 0:
                el = time.time() - ft
                print(f"    [{i+1}/{len(cands)}] {el:.0f}s "
                      f"({el/(i+1):.1f}s/cand)", flush=True)

        full_w = greedy(oof, y[tr_i], task, max_size, n_bags, seed=seed + fi)
        weights = prune_weights(full_w, cum_keep) if cum_keep < 1.0 else full_w
        pred = sum(w * by_name[k].fit_predict(tr_i, te_i, y)
                   for k, w in weights.items())

        best = max(oof, key=lambda k: sc(y[tr_i], oof[k]))
        p_single = by_name[best].fit_predict(tr_i, te_i, y)

        sfn = score_affect3 if task == "affect3" else score_regression
        row = {"fold": fi, "n_test": len(te_i), "n_members": len(weights),
               **sfn(y[te_i], pred),
               **{f"single_{k}": v for k, v in sfn(y[te_i], p_single).items()}}
        fold_rows.append(row)
        choice_rows.append({
            "fold": fi, "n_members": len(weights),
            "top_members": " | ".join(f"{k}:{w:.2f}"
                                      for k, w in list(weights.items())[:5]),
            "weight_window": round(sum(w for k, w in weights.items()
                                       if k.startswith("win-")), 3)})
        key = "macro_f1" if task == "affect3" else "rmse"
        print(f"  fold {fi}: {key}={row[key]:.4f} members={len(weights)} "
              f"({time.time()-ft:.0f}s)", flush=True)

    df = pd.DataFrame(fold_rows)
    metrics = (["macro_f1", "weighted_f1", "balanced_acc", "accuracy"]
               if task == "affect3"
               else ["rmse", "mae", "pearson_r", "spearman_r", "r2"])
    headline = {}
    for m in metrics:
        headline[f"{task}_{m}"] = float(df[m].mean())
        headline[f"{task}_{m}_std"] = float(df[m].std())
    headline[f"{task}_single_{metrics[0]}"] = float(df[f"single_{metrics[0]}"].mean())
    headline[f"{task}_n_folds"] = int(len(df))
    headline[f"{task}_n_recordings"] = int(len(y))

    rank = sorted(((k, float(np.mean(v))) for k, v in inner_acc.items()),
                  key=lambda kv: -kv[1])[:20]
    sota_report.write(
        run_name, headline,
        {"task": task, "protocol": "subject-shared repeated K-fold (paper-style)",
         "n_folds": n_folds, "repeats": repeats, "seed": seed, "views": views,
         "window_views": windows, "models": list(zoo), "cum_keep": cum_keep,
         "inner_folds": inner_folds, "feature_version": SF.FEATURE_VERSION,
         "selection": "bagged greedy w/ replacement on inner OOF"},
        notes,
        {f"Per-fold — {task}": fold_rows,
         f"Selected members — {task}": choice_rows,
         f"Inner-CV ranking — {task}": [
             {"candidate": k, "inner_score": v} for k, v in rank]},
        duration_s=time.time() - t0)

    import shutil
    shutil.rmtree(mat_dir, ignore_errors=True)
    return headline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["affect3", "regression"], required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--views", default="raw,rel")
    ap.add_argument("--windows", default="")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--max-size", type=int, default=25)
    ap.add_argument("--bags", type=int, default=12)
    ap.add_argument("--cum-keep", type=float, default=0.90)
    ap.add_argument("--inner-folds", type=int, default=3)
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    h = run(full_config(), a.task, a.run_name, a.views.split(","), a.repeats,
            a.seed, a.max_size, a.bags, a.cum_keep,
            [s for s in a.windows.split(",") if s], a.inner_folds, a.notes)
    for k, v in h.items():
        print(f"  {k:<34} {v}")


if __name__ == "__main__":
    main()
