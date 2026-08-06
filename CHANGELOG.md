# Changelog

## 2026-08-06 — Leakage proof, and a correction to a published-in-repo number

### Added

- **`LEAKY_PROTOCOL.md`** — standalone explanation of the leaky evaluation
  protocol with four experiments proving it inflates StressID scores.
- **`research_way/prove_leakage.py`** — regenerates every number in that document
  from cached features. ~4 min, CPU only, no GPU and no re-preprocessing.
- **`research_way/reports/leakage_proof/`** — raw outputs (`protocol_swap.csv`,
  `inflation.csv`, `mechanism.json`).

### Findings

Run on the full 700-recording / 64-subject corpus:

| # | Experiment | Result |
|---|---|---|
| E-A | Protocol swap — same features and model, only the split rule changes | physio + RF inflates **+0.098** macro F1 (0.540 → 0.638); every real modality gains |
| E-B | Negative control — `availability_only` carries no subject identity | **−0.003**, i.e. zero. Rules out "random folds were just easier" |
| E-C | Identity probe — can we recognise *who* a recording came from? | physio **0.649** accuracy across 64 subjects = **41.5× chance** |
| E-D | Identity oracle — predict the label from subject ID alone, no signal | **0.628** macro F1, beating the honest physio model's 0.540. Undefined under GroupKFold |

Measured train/test subject overlap: **58.4 of 64 subjects (91%)** under the leaky
protocol, **0.0** under GroupKFold.

E-D is the decisive one: a lookup table keyed on "which person is this, and what
did they usually report in training" outscores real models that process the
actual signals. Within-subject label consistency is 0.708 — people are
repetitive, so identity alone is worth ~0.63 macro F1, and a random split hands
that to the model for free.

### Corrected

**Physio leakage inflation was reported as +0.127 weighted F1 (0.670 → 0.543).
That figure does not reproduce and is unsupported by any stored artifact.**

| Source | physio + RF inflation |
|---|---|
| `results/full/e1_leakage_inflation.csv` | +0.093 |
| `results/full_run1_20260729/e1_leakage_inflation.csv` | +0.093 |
| fresh run, `prove_leakage.py` | **+0.098** |
| *prose claim* | *+0.127 — no run produced it* |

The leakage-free value (0.543) was correct; the leaky value (0.670) was not.
Corrected in `RESEARCH_PROGRESS.md` §10.3, §12 B1, §13, §14.1 and
`research_way/RESULTS_AND_COMPARISON.md` §3.2. The effect is **~23% smaller** than
previously written. Qualitative conclusions are unchanged — physiology still
inflates most by a wide margin, and the negative control still sits at zero.

§14.1's downstream arithmetic was corrected with it: expected handcrafted-feature
performance under GroupKFold moves 0.60 → **0.63**, and A1's real headroom
0.03 → **0.06**.

---

## 2026-08-04 — Dataset re-download, audit, and repo cleanup

### Changed

- **`research_way/src/config.py`** — `DATASET_ROOT` now points at
  `StressID Dataset new/`.
- **`.gitignore`** — added `StressID Dataset new/`. The previous rule listed only
  the old path, so the new folder was not covered.
- **`RESEARCH_PROGRESS.md`** — new §0 recording the audit below.

### Removed

- **`StressID Dataset/`** (old copy, 2611 files, 4.81 GB) — deleted after
  verifying it is byte-identical to the new copy.

### Audit: the re-download is byte-identical to the previous copy

The dataset was re-downloaded to test whether a corrupted copy explained the low
scores. It did not — nothing differs.

| Check | Old | New | Result |
|---|---|---|---|
| Physiological `.txt` (MD5, every file) | 777 | 777 | **0 differing hashes** |
| Audio `.wav` (name + size) | 378 | 378 | identical |
| Videos `.mp4` (name + size) | 629 | 629 | identical |
| `labels.csv`, `demographics.csv`, `self_assessments.csv`, `labels_supplementary.csv` (MD5) | — | — | identical |
| Total | 2611 files / 4.81 GB | 2611 files / 4.81 GB | identical |

Re-verified on the new copy: 700 labelled recordings, 64 subjects,
`P(stress | audio present) = 0.709` (n=378) vs `P(stress | audio absent) = 0.311`
(n=322). The modality-availability confound is structural in the recording design
and unaffected by re-downloading. Cached features under `research_way/data/`
remain valid; no regeneration was needed.

### Incident: 4.3 GB of dataset committed by the auto-commit hook

The repo's auto-commit hook stages everything not gitignored. Because
`.gitignore` covered only `StressID Dataset/`, the hook committed **1791 dataset
files (~4.3 GB)** into git history across two `auto: update progress` commits as
soon as the new folder appeared.

Both commits were local and unpushed (`origin/main` was still at `0dd3f67`), so
recovery was clean:

1. Added `StressID Dataset new/` to `.gitignore`
2. `git reset --mixed 0dd3f67` — undid both auto-commits, working tree untouched
3. Re-committed only the three genuine file changes
4. `git reflog expire --expire=now --all && git gc --prune=now`

`.git` went from **4.3 GB back to 42 MB**. No file content was lost — the dataset
only ever left the git index, never the disk.

**Standing rule:** any large directory added to this repo must be gitignored in
the same turn it is created, because the hook commits immediately.
