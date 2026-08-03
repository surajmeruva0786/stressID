"""Per-run reporting: every training/eval run leaves a durable record.

Writes into `reports/<run_name>/`:

    report.md    human-readable summary (config, metrics, deltas vs best)
    report.pdf   same content as a PDF (matplotlib PdfPages, no extra deps)
    metrics.json machine-readable row for the registry

and appends one line to `reports/RUNS.md` + `reports/runs_index.csv`, the
cross-run registry. Without this, runs overwrite each other's context and the
only record of what was tried is a git log message.

    from src.report import write_report
    write_report(run_name="a1a2", headline={...}, config={...}, notes="...")
"""
from __future__ import annotations

import json
import platform
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from .config import RESEARCH_ROOT

# Deliberately NOT under results/ -- that tree is gitignored (predictions.csv,
# checkpoints, caches). Reports are the durable record of what was tried, so
# they live in a tracked directory and are committed with the code that made them.
REPORTS_DIR = RESEARCH_ROOT / "reports"
REGISTRY_CSV = REPORTS_DIR / "runs_index.csv"
REGISTRY_MD = REPORTS_DIR / "RUNS.md"

# The number that matters: macro F1 on the all-modality subset under subject
# GroupKFold. Availability is constant there, so the shortcut carries no signal.
PRIMARY_METRIC = "complete364_macro_f1"
MAJORITY_REF = 0.418


def _fig_text(title: str, body: str, fontsize: int = 8.5):
    fig = plt.figure(figsize=(8.27, 11.69))          # A4 portrait
    fig.text(0.06, 0.955, title, fontsize=15, weight="bold", va="top")
    fig.text(0.06, 0.915, body, fontsize=fontsize, va="top", family="monospace",
             linespacing=1.45)
    return fig


def _table_md(rows: list[dict], cols: list[str]) -> str:
    if not rows:
        return "_(none)_"
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = []
    for r in rows:
        body.append("| " + " | ".join(
            f"{r.get(c, ''):.4f}" if isinstance(r.get(c), float) else str(r.get(c, ""))
            for c in cols) + " |")
    return "\n".join([head, sep] + body)


def load_registry() -> pd.DataFrame:
    if REGISTRY_CSV.exists():
        return pd.read_csv(REGISTRY_CSV)
    return pd.DataFrame()


def best_so_far() -> tuple[str, float]:
    """(run_name, score) of the best PRIMARY_METRIC recorded so far."""
    reg = load_registry()
    if reg.empty or PRIMARY_METRIC not in reg.columns:
        return ("none", float("nan"))
    reg = reg.dropna(subset=[PRIMARY_METRIC])
    if reg.empty:
        return ("none", float("nan"))
    r = reg.loc[reg[PRIMARY_METRIC].idxmax()]
    return (str(r["run_name"]), float(r[PRIMARY_METRIC]))


def write_report(run_name: str, headline: dict, config: dict,
                 notes: str = "", tables: dict | None = None,
                 figures: list[Path] | None = None,
                 duration_s: float | None = None) -> Path:
    """Write report.md + report.pdf for one run and update the registry.

    `headline` must carry PRIMARY_METRIC; other numeric keys are recorded too.
    `tables` maps a section title -> list of row dicts.
    """
    out = REPORTS_DIR / run_name
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prev_name, prev_score = best_so_far()

    score = float(headline.get(PRIMARY_METRIC, float("nan")))
    delta_best = score - prev_score if prev_score == prev_score else float("nan")
    verdict = ("FIRST RUN" if prev_score != prev_score else
               "NEW BEST" if delta_best > 0 else "no improvement")

    # ---------------------------------------------------------------- markdown
    lines = [
        f"# Run report — `{run_name}`", "",
        f"**Generated:** {ts}  ",
        f"**Primary metric ({PRIMARY_METRIC}):** **{score:.4f}**  ",
        f"**Majority-class reference:** {MAJORITY_REF:.4f} "
        f"(margin {score - MAJORITY_REF:+.4f})  ",
        f"**Best previous:** `{prev_name}` at "
        f"{prev_score:.4f} → **{verdict}** ({delta_best:+.4f})  ",
    ]
    if duration_s:
        lines.append(f"**Duration:** {duration_s:.0f} s ({duration_s / 60:.1f} min)  ")
    lines += ["", "> Primary metric is macro F1 on the 364 all-modality recordings",
              "> under subject GroupKFold. Availability is constant there, so the",
              "> modality-availability shortcut carries no signal (see §10.5).", ""]

    if notes:
        lines += ["## What changed", "", notes, ""]

    lines += ["## Headline metrics", "", _table_md(
        [{"metric": k, "value": v} for k, v in headline.items()],
        ["metric", "value"]), ""]

    for title, rows in (tables or {}).items():
        cols = list(rows[0].keys()) if rows else []
        lines += [f"## {title}", "", _table_md(rows, cols), ""]

    lines += ["## Config", "", "```json",
              json.dumps(config, indent=2, default=str), "```", ""]
    lines += ["## Environment", "",
              f"- {platform.platform()}", f"- Python {platform.python_version()}", ""]

    md = "\n".join(lines)
    (out / "report.md").write_text(md, encoding="utf-8")

    # --------------------------------------------------------------------- pdf
    with PdfPages(out / "report.pdf") as pdf:
        summary = [
            f"Generated      : {ts}",
            f"Primary metric : {score:.4f}   ({PRIMARY_METRIC})",
            f"Majority ref   : {MAJORITY_REF:.4f}   margin {score - MAJORITY_REF:+.4f}",
            f"Best previous  : {prev_name} @ {prev_score:.4f}",
            f"Verdict        : {verdict} ({delta_best:+.4f})",
        ]
        if duration_s:
            summary.append(f"Duration       : {duration_s:.0f}s ({duration_s/60:.1f} min)")
        summary += ["", "-" * 74, "", "HEADLINE METRICS", ""]
        summary += [f"  {k:<34} {v:.4f}" if isinstance(v, float) else f"  {k:<34} {v}"
                    for k, v in headline.items()]
        if notes:
            summary += ["", "-" * 74, "", "WHAT CHANGED", ""]
            summary += ["  " + ln for ln in notes.splitlines()]
        pdf.savefig(_fig_text(f"Run report — {run_name}", "\n".join(summary)))
        plt.close("all")

        for title, rows in (tables or {}).items():
            if not rows:
                continue
            df = pd.DataFrame(rows)
            txt = df.to_string(index=False, max_rows=55,
                               float_format=lambda v: f"{v:.4f}")
            pdf.savefig(_fig_text(title, txt, fontsize=7.5))
            plt.close("all")

        cfg_txt = json.dumps(config, indent=2, default=str)
        pdf.savefig(_fig_text("Config", cfg_txt, fontsize=7))
        plt.close("all")

        for f in (figures or []):
            if Path(f).exists():
                img = plt.imread(str(f))
                fig = plt.figure(figsize=(11.69, 8.27))
                ax = fig.add_subplot(111); ax.imshow(img); ax.axis("off")
                ax.set_title(Path(f).name, fontsize=10)
                pdf.savefig(fig); plt.close("all")

    # ---------------------------------------------------------------- registry
    row = {"run_name": run_name, "timestamp": ts, "verdict": verdict,
           "delta_vs_best": delta_best, "duration_s": duration_s, **headline}
    (out / "metrics.json").write_text(json.dumps(row, indent=2, default=str),
                                      encoding="utf-8")

    reg = load_registry()
    reg = reg[reg["run_name"] != run_name] if not reg.empty and "run_name" in reg else reg
    reg = pd.concat([reg, pd.DataFrame([row])], ignore_index=True)
    REGISTRY_CSV.parent.mkdir(parents=True, exist_ok=True)
    reg.to_csv(REGISTRY_CSV, index=False)

    md_rows = reg.sort_values(PRIMARY_METRIC, ascending=False, na_position="last")
    reg_md = ["# Run registry", "",
              f"Primary metric: `{PRIMARY_METRIC}` "
              f"(macro F1, 364 all-modality recordings, subject GroupKFold). "
              f"Majority reference {MAJORITY_REF:.3f}.", "",
              "| run | " + PRIMARY_METRIC + " | vs majority | verdict | when |",
              "|---|---|---|---|---|"]
    for _, r in md_rows.iterrows():
        s = r.get(PRIMARY_METRIC, float("nan"))
        reg_md.append(f"| `{r['run_name']}` | {s:.4f} | {s - MAJORITY_REF:+.4f} | "
                      f"{r.get('verdict', '')} | {r.get('timestamp', '')} |")
    REGISTRY_MD.write_text("\n".join(reg_md) + "\n", encoding="utf-8")

    print(f"[report] {run_name}: {PRIMARY_METRIC}={score:.4f} ({verdict}) -> {out}")
    return out
