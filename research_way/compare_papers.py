"""Recompute our results under the StressID paper's own metrics so the
comparison against its Table 2 / Table 3 is apples-to-apples.

The paper reports WEIGHTED F1 and BALANCED ACCURACY over 10 random 80/20
task-level splits. Our pipeline reports MACRO F1 and plain accuracy under
subject-grouped 5-fold CV. Those are different numbers on different protocols,
so this script produces both metric families for both protocols.
"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, r"F:\stressID\research_way")

from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
from sklearn.model_selection import KFold

from src.config import full_config
from src.dataset import load_manifest
from src.splits import load_splits
from src.baselines import extract_static_features, _fit_eval, MODALITIES_ORDER

OUT = r"C:\Users\BIGDAT~1\AppData\Local\Temp\claude\F--stressID\c28df286-8bc7-4cdb-b8c0-59d1a9768c80\scratchpad"
cfg = full_config()
man = load_manifest(cfg)
print(f"manifest: {len(man)} recordings, {man.subject.nunique()} subjects")

complete = (man.has_physio == 1) & (man.has_audio == 1) & (man.has_video == 1)
print(f"all-3-modality recordings: {complete.sum()}  (paper's multimodal subset = 370 tasks)")

# ---------------------------------------------------------------- our model
pred = pd.read_csv(r"F:\stressID\research_way\results\full_run1_20260729\predictions.csv")
pred = pred[pred.condition == "full"].copy()
pred["complete"] = (pred.nat_physio == 1) & (pred.nat_audio == 1) & (pred.nat_video == 1)

rows = []
for scope, sub in (("all_700", pred), ("complete_364", pred[pred.complete])):
    for variant in ("static", "temporal"):
        s = sub[sub.variant == variant]
        per_run = []
        for (seed, fold), g in s.groupby(["seed", "fold"]):
            yb = g.y_binary.values
            pb = (g.prob.values >= 0.5).astype(int)
            y3 = g.y_affect3.values
            p3 = g[["logit3_0", "logit3_1", "logit3_2"]].values.argmax(1)
            per_run.append({
                "bin_f1_weighted": f1_score(yb, pb, average="weighted", zero_division=0),
                "bin_f1_macro":    f1_score(yb, pb, average="macro", zero_division=0),
                "bin_bal_acc":     balanced_accuracy_score(yb, pb),
                "bin_acc":         accuracy_score(yb, pb),
                "aff_f1_weighted": f1_score(y3, p3, average="weighted", zero_division=0),
                "aff_f1_macro":    f1_score(y3, p3, average="macro", zero_division=0),
                "aff_bal_acc":     balanced_accuracy_score(y3, p3),
                "aff_acc":         accuracy_score(y3, p3),
            })
        d = pd.DataFrame(per_run)
        r = {"source": "ours_model", "protocol": "groupkfold", "scope": scope,
             "method": f"MST-{variant}", "n_runs": len(d)}
        for c in d.columns:
            r[c] = d[c].mean(); r[c + "_std"] = d[c].std()
        rows.append(r)

# ------------------------------------------------------- classical baselines
feats = extract_static_features(cfg, man)
feats["availability"] = man[["has_physio", "has_audio", "has_video"]].values.astype(float)
y = man["binary"].values
subj = man["subject"].values
folds = load_splits(cfg)

CONFIGS = {"physio": ["physio"], "audio": ["audio"], "video": ["video"],
           "feature_fusion": ["physio", "audio", "video"],
           "availability_only": ["availability"]}

def eval_splits(splits, idx_pool, scope, protocol):
    out = []
    for clf_name in ("logreg", "rf"):
        for cname, mods in CONFIGS.items():
            X = np.concatenate([feats[m] for m in mods], axis=1)
            m_ = {k: [] for k in ("f1w", "f1m", "bacc", "acc")}
            for tr, te in splits:
                p, yt = _fit_eval(X, y, tr, te, clf_name)
                pb = (p >= .5).astype(int)
                m_["f1w"].append(f1_score(yt, pb, average="weighted", zero_division=0))
                m_["f1m"].append(f1_score(yt, pb, average="macro", zero_division=0))
                m_["bacc"].append(balanced_accuracy_score(yt, pb))
                m_["acc"].append(accuracy_score(yt, pb))
            out.append({"source": "ours_baseline", "protocol": protocol, "scope": scope,
                        "method": f"{cname}+{clf_name}", "n_runs": len(splits),
                        "bin_f1_weighted": np.mean(m_["f1w"]), "bin_f1_weighted_std": np.std(m_["f1w"]),
                        "bin_f1_macro": np.mean(m_["f1m"]), "bin_f1_macro_std": np.std(m_["f1m"]),
                        "bin_bal_acc": np.mean(m_["bacc"]), "bin_bal_acc_std": np.std(m_["bacc"]),
                        "bin_acc": np.mean(m_["acc"]), "bin_acc_std": np.std(m_["acc"])})
        # decision fusion
        m_ = {k: [] for k in ("f1w", "f1m", "bacc", "acc")}
        for tr, te in splits:
            ps = [_fit_eval(feats[m], y, tr, te, clf_name)[0] for m in MODALITIES_ORDER]
            p = np.mean(ps, axis=0); pb = (p >= .5).astype(int); yt = y[te]
            m_["f1w"].append(f1_score(yt, pb, average="weighted", zero_division=0))
            m_["f1m"].append(f1_score(yt, pb, average="macro", zero_division=0))
            m_["bacc"].append(balanced_accuracy_score(yt, pb))
            m_["acc"].append(accuracy_score(yt, pb))
        out.append({"source": "ours_baseline", "protocol": protocol, "scope": scope,
                    "method": f"decision_fusion+{clf_name}", "n_runs": len(splits),
                    "bin_f1_weighted": np.mean(m_["f1w"]), "bin_f1_weighted_std": np.std(m_["f1w"]),
                    "bin_f1_macro": np.mean(m_["f1m"]), "bin_f1_macro_std": np.std(m_["f1m"]),
                    "bin_bal_acc": np.mean(m_["bacc"]), "bin_bal_acc_std": np.std(m_["bacc"]),
                    "bin_acc": np.mean(m_["acc"]), "bin_acc_std": np.std(m_["acc"])})
    return out

# scope: all 700 recordings
grouped = [(np.where(~np.isin(subj, f["test"]))[0], np.where(np.isin(subj, f["test"]))[0])
           for f in folds]
rows += eval_splits(grouped, None, "all_700", "groupkfold")

# the paper's own protocol: random 80/20 task-level splits, 10 repetitions
rng = np.random.default_rng(1337)
idx = np.arange(len(man))
paper_splits = []
for rep in range(10):
    perm = rng.permutation(idx)
    cut = int(0.8 * len(perm))
    paper_splits.append((perm[:cut], perm[cut:]))
rows += eval_splits(paper_splits, None, "all_700", "random_80_20_PAPER_STYLE")

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/paper_comparable_metrics.csv", index=False)

pd.set_option("display.width", 200)
cols = ["source", "protocol", "scope", "method", "bin_f1_weighted", "bin_f1_macro",
        "bin_bal_acc", "bin_acc"]
print("\n" + df[cols].round(3).to_string(index=False))
print(f"\nwrote {OUT}/paper_comparable_metrics.csv")
