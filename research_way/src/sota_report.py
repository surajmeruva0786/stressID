"""Reporting for the SOTA (paper-comparable, subject-shared) campaign.

Kept separate from `report.py` on purpose. That module's registry is keyed on
`complete364_macro_f1` under subject GroupKFold -- the leakage-free number. This
campaign optimises a *different* protocol and must not be mixed into the same
leaderboard, or a reader would compare 0.85 (subject-shared) against 0.52
(subject-held-out) as if they answered the same question.

    reports_sota/<run>/report.md | metrics.json
    reports_sota/RUNS.md, runs_index.csv        cross-run leaderboard
"""
from __future__ import annotations

import json
import platform
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import RESEARCH_ROOT

REPORTS_DIR = RESEARCH_ROOT / "reports_sota"
REGISTRY_CSV = REPORTS_DIR / "runs_index.csv"
REGISTRY_MD = REPORTS_DIR / "RUNS.md"

PRIMARY_METRIC = "all700_macro_f1"
# StressID origin paper, best reported fusion result (random 80/20 + SMOTE).
PAPER_REF_WEIGHTED_F1 = 0.72


def _table_md(rows: list[dict], cols: list[str] | None = None) -> str:
    if not rows:
        return "_(none)_"
    cols = cols or list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            cells.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def load_registry() -> pd.DataFrame:
    return pd.read_csv(REGISTRY_CSV) if REGISTRY_CSV.exists() else pd.DataFrame()


def best_so_far() -> tuple[str, float]:
    reg = load_registry()
    if reg.empty or PRIMARY_METRIC not in reg.columns:
        return ("none", float("nan"))
    reg = reg.dropna(subset=[PRIMARY_METRIC])
    if reg.empty:
        return ("none", float("nan"))
    r = reg.loc[reg[PRIMARY_METRIC].idxmax()]
    return (str(r["run_name"]), float(r[PRIMARY_METRIC]))


def write(run_name: str, headline: dict, config: dict, notes: str = "",
          tables: dict | None = None, duration_s: float | None = None) -> Path:
    out = REPORTS_DIR / run_name
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prev_name, prev_score = best_so_far()
    score = float(headline.get(PRIMARY_METRIC, float("nan")))
    delta = score - prev_score if prev_score == prev_score else float("nan")
    verdict = ("FIRST RUN" if prev_score != prev_score else
               "NEW BEST" if delta > 0 else "no improvement")

    lines = [
        f"# SOTA run — `{run_name}`", "",
        f"**Generated:** {ts}  ",
        f"**Primary ({PRIMARY_METRIC}):** **{score:.4f}**  ",
        f"**Best previous:** `{prev_name}` at {prev_score:.4f} → "
        f"**{verdict}** ({delta:+.4f})  ",
    ]
    if duration_s:
        lines.append(f"**Duration:** {duration_s:.0f} s ({duration_s / 60:.1f} min)  ")
    lines += [
        "", "> **Protocol.** Subject-shared repeated stratified CV — the split rule",
        "> the StressID origin paper uses (random 80/20 + SMOTE). Subjects appear",
        "> on both sides. Numbers here are comparable to published ones and are",
        "> **not** comparable to the GroupKFold track in `reports/`, which holds",
        f"> subjects out. Origin paper's best reported weighted F1: {PAPER_REF_WEIGHTED_F1:.2f}.", "",
    ]
    if notes:
        lines += ["## What changed", "", notes, ""]
    lines += ["## Headline metrics", "",
              _table_md([{"metric": k, "value": v} for k, v in headline.items()],
                        ["metric", "value"]), ""]
    for title, rows in (tables or {}).items():
        lines += [f"## {title}", "", _table_md(rows), ""]
    lines += ["## Config", "", "```json", json.dumps(config, indent=2, default=str),
              "```", "", "## Environment", "",
              f"- {platform.platform()}", f"- Python {platform.python_version()}", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    row = {"run_name": run_name, "timestamp": ts, "verdict": verdict,
           "delta_vs_best": delta, "duration_s": duration_s, **headline}
    (out / "metrics.json").write_text(json.dumps(row, indent=2, default=str),
                                      encoding="utf-8")

    reg = load_registry()
    if not reg.empty and "run_name" in reg:
        reg = reg[reg["run_name"] != run_name]
    reg = pd.concat([reg, pd.DataFrame([row])], ignore_index=True)
    REGISTRY_CSV.parent.mkdir(parents=True, exist_ok=True)
    reg.to_csv(REGISTRY_CSV, index=False)

    cols = ["run_name", PRIMARY_METRIC, "all700_weighted_f1", "all700_accuracy",
            "c364_macro_f1", "verdict", "timestamp"]
    cols = [c for c in cols if c in reg.columns]
    md = reg.sort_values(PRIMARY_METRIC, ascending=False, na_position="last")[cols]
    REGISTRY_MD.write_text(
        "# SOTA leaderboard (subject-shared / paper-comparable protocol)\n\n"
        f"Primary metric: `{PRIMARY_METRIC}`. Origin paper best weighted F1 = "
        f"{PAPER_REF_WEIGHTED_F1:.2f}.\n\n"
        "These numbers are **not** comparable to `reports/RUNS.md` (subject-held-out).\n\n"
        + _table_md(md.to_dict("records"), cols) + "\n", encoding="utf-8")

    print(f"[sota-report] {run_name}: {PRIMARY_METRIC}={score:.4f} ({verdict}) -> {out}")
    return out
