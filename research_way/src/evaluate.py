"""Stage 6 evaluation suite.

Reads `predictions.csv` (one row per recording x mask-condition x seed x fold)
and produces:

  E2 / E4   temporal vs static, full modalities        -> e2_temporal_vs_static.csv
  E5 / E6   1-missing and 2-missing degradation        -> e5e6_missing_modality.csv
  E8        3-class per-class recall                   -> e8_per_class_recall.csv
  E12       degradation curves, temporal vs static     -> e12_degradation.csv + .png
  E13       ECE / reliability under missing modalities -> e13_calibration.csv + .png

All headline numbers carry a bootstrap 95% CI over (seed x fold) runs, and E2 /
E12 carry a paired bootstrap p-value.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import Config
from .metrics import affect3_metrics, binary_metrics, bootstrap_ci, paired_bootstrap_pvalue

ONE_MISSING = ["no_physio", "no_audio", "no_video"]
TWO_MISSING = ["physio_only", "audio_only", "video_only"]   # only 1 modality kept
CONDITION_LEVEL = {"full": 0, **{c: 1 for c in ONE_MISSING}, **{c: 2 for c in TWO_MISSING}}


def _complete_only(preds: pd.DataFrame) -> pd.DataFrame:
    """Restrict to recordings where all 3 modalities NATURALLY exist.

    Otherwise a condition like `no_audio` would be compared against recordings
    that never had audio, and the degradation curve would be uninterpretable."""
    m = (preds["nat_physio"] == 1) & (preds["nat_audio"] == 1) & (preds["nat_video"] == 1)
    return preds[m].copy()


def _per_run(preds: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Metrics per (variant, condition, seed, fold) -- the unit for CI/p-values."""
    rows = []
    for (variant, cond, seed, fold), g in preds.groupby(
            ["variant", "condition", "seed", "fold"]):
        b = binary_metrics(g["prob"].values, g["y_binary"].values, cfg.ece_bins)
        b.pop("reliability")
        a = affect3_metrics(g[["logit3_0", "logit3_1", "logit3_2"]].values,
                            g["y_affect3"].values)
        rows.append({"variant": variant, "condition": cond, "seed": seed, "fold": fold,
                     **{f"bin_{k}": v for k, v in b.items()},
                     **{f"aff_{k}": v for k, v in a.items() if k != "confusion"}})
    return pd.DataFrame(rows)


def _summarise(per_run: pd.DataFrame, metric: str) -> pd.DataFrame:
    out = []
    for (variant, cond), g in per_run.groupby(["variant", "condition"]):
        mean, lo, hi = bootstrap_ci(g[metric].values)
        out.append({"variant": variant, "condition": cond, "metric": metric,
                    "mean": mean, "ci_lo": lo, "ci_hi": hi, "n_runs": len(g)})
    return pd.DataFrame(out)


def _paired(per_run: pd.DataFrame, metric: str, cond: str) -> dict:
    """Paired temporal-vs-static comparison on matched (seed, fold) runs."""
    t = per_run[(per_run.variant == "temporal") & (per_run.condition == cond)] \
        .set_index(["seed", "fold"])[metric]
    s = per_run[(per_run.variant == "static") & (per_run.condition == cond)] \
        .set_index(["seed", "fold"])[metric]
    idx = t.index.intersection(s.index)
    if len(idx) == 0:
        return {}
    a, b = t.loc[idx].values, s.loc[idx].values
    mean, lo, hi = bootstrap_ci(a - b)
    return {"condition": cond, "metric": metric, "temporal_mean": float(np.mean(a)),
            "static_mean": float(np.mean(b)), "delta": mean, "delta_ci_lo": lo,
            "delta_ci_hi": hi, "p_value": paired_bootstrap_pvalue(a, b),
            "n_pairs": len(idx)}


# ------------------------------------------------------------------- reporting

def run(cfg: Config | None = None) -> dict:
    cfg = cfg or Config()
    rd = cfg.results_dir
    preds_all = pd.read_csv(rd / "predictions.csv")
    preds = _complete_only(preds_all)
    per_run = _per_run(preds, cfg)
    per_run.to_csv(rd / "per_run_metrics.csv", index=False)

    uniq = preds.drop_duplicates("key")
    print(f"[eval] E5/E6/E12 evaluated on {len(uniq)} recordings with all 3 modalities "
          f"naturally present (of {preds_all['key'].nunique()} tested); "
          f"binary balance {uniq['y_binary'].mean():.2f}, "
          f"tasks {sorted(uniq['task'].unique())}")
    print("[eval] restricting to complete recordings keeps modality availability CONSTANT "
          "across conditions, so the degradation curve cannot be driven by the "
          "availability/label confound (see README).")

    report: dict = {}

    # ---- E2 / E4: temporal vs static at full modality
    e2 = _paired(per_run, "bin_f1_macro", "full")
    e2_acc = _paired(per_run, "bin_accuracy", "full")
    pd.DataFrame([e2, e2_acc]).to_csv(rd / "e2_temporal_vs_static.csv", index=False)
    report["E2_temporal_vs_static_full"] = [e2, e2_acc]

    # ---- E5 / E6: missing-modality degradation
    summ = pd.concat([_summarise(per_run, m) for m in
                      ("bin_f1_macro", "bin_accuracy", "bin_ece")], ignore_index=True)
    summ["n_missing"] = summ["condition"].map(CONDITION_LEVEL)
    summ = summ.sort_values(["metric", "variant", "n_missing", "condition"])
    summ.to_csv(rd / "e5e6_missing_modality.csv", index=False)
    report["E5E6_missing_modality"] = summ.to_dict("records")

    # ---- E8: per-class recall (3-class affect)
    e8 = pd.concat([_summarise(per_run, f"aff_recall_class{c}") for c in (0, 1, 2)]
                   + [_summarise(per_run, "aff_f1_macro")], ignore_index=True)
    e8 = e8[e8.condition == "full"]
    e8.to_csv(rd / "e8_per_class_recall.csv", index=False)
    report["E8_per_class_recall_full"] = e8.to_dict("records")

    # ---- E12 (HEADLINE): degradation curve, temporal vs static
    f1 = summ[summ.metric == "bin_f1_macro"]
    curve = []
    for variant in sorted(f1["variant"].unique()):
        base = f1[(f1.variant == variant) & (f1.condition == "full")]["mean"].iloc[0]
        for lvl, conds in ((0, ["full"]), (1, ONE_MISSING), (2, TWO_MISSING)):
            vals = per_run[(per_run.variant == variant)
                           & (per_run.condition.isin(conds))]["bin_f1_macro"].values
            mean, lo, hi = bootstrap_ci(vals)
            curve.append({"variant": variant, "n_missing": lvl, "f1_macro": mean,
                          "ci_lo": lo, "ci_hi": hi,
                          "rel_drop_vs_full": float(1 - mean / base) if base else np.nan})
    curve = pd.DataFrame(curve)
    e12_pairs = [_paired(per_run, "bin_f1_macro", c)
                 for c in ["full"] + ONE_MISSING + TWO_MISSING]
    curve.to_csv(rd / "e12_degradation.csv", index=False)
    pd.DataFrame([p for p in e12_pairs if p]).to_csv(
        rd / "e12_paired_tests.csv", index=False)
    report["E12_degradation_curve"] = curve.to_dict("records")
    report["E12_paired_tests"] = [p for p in e12_pairs if p]

    # ---- E13: calibration under missing modalities
    rel_rows, ece_rows = [], []
    for (variant, cond), g in preds.groupby(["variant", "condition"]):
        b = binary_metrics(g["prob"].values, g["y_binary"].values, cfg.ece_bins)
        ece_rows.append({"variant": variant, "condition": cond,
                         "n_missing": CONDITION_LEVEL[cond], "ece": b["ece"],
                         "accuracy": b["accuracy"], "mean_conf": float(
                             np.where(g["prob"] >= .5, g["prob"], 1 - g["prob"]).mean())})
        for bn in b["reliability"]:
            rel_rows.append({"variant": variant, "condition": cond, **bn})
    pd.DataFrame(ece_rows).sort_values(["variant", "n_missing"]).to_csv(
        rd / "e13_calibration.csv", index=False)
    pd.DataFrame(rel_rows).to_csv(rd / "e13_reliability_bins.csv", index=False)
    report["E13_calibration"] = ece_rows

    (rd / "report.json").write_text(json.dumps(report, indent=2, default=float),
                                    encoding="utf-8")
    _plots(cfg, curve, pd.DataFrame(ece_rows), pd.DataFrame(rel_rows))
    _print_summary(curve, e2, summ, pd.DataFrame(ece_rows), e8)
    return report


def _plots(cfg: Config, curve: pd.DataFrame, ece: pd.DataFrame, rel: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for variant, g in curve.groupby("variant"):
        g = g.sort_values("n_missing")
        ax[0].errorbar(g["n_missing"], g["f1_macro"],
                       yerr=[g["f1_macro"] - g["ci_lo"], g["ci_hi"] - g["f1_macro"]],
                       marker="o", capsize=4, label=variant)
    ax[0].set_xticks([0, 1, 2]); ax[0].set_xlabel("# modalities missing at inference")
    ax[0].set_ylabel("macro F1 (binary stress)")
    ax[0].set_title("E12 - degradation curve"); ax[0].legend(); ax[0].grid(alpha=.3)

    for variant, g in ece.groupby("variant"):
        g = g.groupby("n_missing")["ece"].mean().reset_index()
        ax[1].plot(g["n_missing"], g["ece"], marker="s", label=variant)
    ax[1].set_xticks([0, 1, 2]); ax[1].set_xlabel("# modalities missing")
    ax[1].set_ylabel("ECE"); ax[1].set_title("E13 - calibration"); ax[1].legend()
    ax[1].grid(alpha=.3)

    sub = rel[(rel.variant == "temporal") & (rel.condition.isin(["full", "physio_only"]))]
    for cond, g in sub.groupby("condition"):
        g = g.dropna(subset=["conf", "acc"])
        ax[2].plot(g["conf"], g["acc"], marker="o", label=cond)
    ax[2].plot([.5, 1], [.5, 1], "k--", lw=1, label="perfect")
    ax[2].set_xlabel("confidence"); ax[2].set_ylabel("accuracy")
    ax[2].set_title("E13 - reliability (temporal)"); ax[2].legend(); ax[2].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(cfg.results_dir / "figures.png", dpi=150)
    plt.close(fig)


def _print_summary(curve, e2, summ, ece, e8) -> None:
    print("\n=== E2/E4 temporal vs static (full modalities) ===")
    if e2:
        print(f"  macro-F1  temporal={e2['temporal_mean']:.3f}  static={e2['static_mean']:.3f}"
              f"  delta={e2['delta']:+.3f} [{e2['delta_ci_lo']:+.3f},{e2['delta_ci_hi']:+.3f}]"
              f"  p={e2['p_value']:.3f}  (n={e2['n_pairs']} runs)")

    print("\n=== E5/E6 missing-modality (macro-F1) ===")
    f1 = summ[summ.metric == "bin_f1_macro"]
    for variant, g in f1.groupby("variant"):
        print(f"  {variant}:")
        for _, r in g.sort_values(["n_missing", "condition"]).iterrows():
            print(f"    {r['condition']:<12} F1={r['mean']:.3f} "
                  f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]")

    print("\n=== E12 HEADLINE degradation ===")
    for _, r in curve.sort_values(["variant", "n_missing"]).iterrows():
        print(f"  {r['variant']:<8} missing={int(r['n_missing'])}  "
              f"F1={r['f1_macro']:.3f}  rel.drop={r['rel_drop_vs_full']*100:5.1f}%")

    print("\n=== E13 calibration (mean ECE by #missing) ===")
    print(ece.groupby(["variant", "n_missing"])["ece"].mean().round(3).to_string())

    print("\n=== E8 per-class recall (3-class, full modalities) ===")
    print(e8[["variant", "metric", "mean", "ci_lo", "ci_hi"]].round(3).to_string(index=False))


if __name__ == "__main__":
    run()
