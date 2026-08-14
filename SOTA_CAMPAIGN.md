# SOTA campaign — pushing StressID as far as the published protocol allows

> **Living document.** One section per round. Every round: change → run →
> evaluate → analyse → commit → document → commit. Numbers here are regenerated
> by `research_way/src/sota.py`; per-run detail lives in
> `research_way/reports_sota/<run>/report.md` and the leaderboard in
> `research_way/reports_sota/RUNS.md`.

---

## 0. What this campaign is, and what it is not

This repository already contains a careful, leakage-free evaluation of StressID
(`LEAKY_PROTOCOL.md`, `reports/`). Under subject GroupKFold — no participant on
both sides of the split — the ceiling is ≈0.52 macro F1, and we proved that the
gap to published numbers is largely a measurement artefact, not a modelling gap.

**This campaign optimises a different question.** It adopts the split rule the
StressID origin paper actually uses — random splits over *recordings*, so the
same participant appears in train and test — and pushes performance as hard as
possible on that protocol, because that is the only protocol on which our
numbers are comparable to the published 0.72.

Both tracks are kept, and they are kept apart:

| | `reports/` (existing) | `reports_sota/` (this campaign) |
|---|---|---|
| Split unit | **subject** (GroupKFold) | **recording** (stratified K-fold) |
| Subject in train *and* test? | never | yes, by construction |
| Question answered | "…for a person it has never seen?" | "…for people it has already met?" |
| Comparable to the paper's 0.72 | no | **yes** |
| Primary metric | `complete364_macro_f1` | `all700_macro_f1` |

Separate leaderboards, separate reporters, separate primary metrics. A number
from one track must never be compared to a number from the other.

### The limitation, stated up front

Scores on this track are **inflated by subject leakage**, and we know by how
much because we measured it (`LEAKY_PROTOCOL.md` §3): +0.098 macro F1 for
physiology, +0.05 for video and fusion, ≈0 for a negative-control feature set
that carries no subject identity. A logistic regression identifies which of 64
participants a recording came from with 64.9% accuracy from physiology alone —
41.5× chance. That is the mechanism.

Two further design choices on this track are deliberately transductive, and are
named in every report rather than buried:

* **subject-relative view (`rel`)** — each recording minus that participant's
  own mean over their Relax/Breathing recordings. Standard baseline correction
  in affective computing; reads no labels.
* **subject-z view (`z`)** — each feature z-scored within participant across
  that participant's recordings. Reads no labels, but test rows contribute to
  their own participant's statistics.

Neither is admissible on the GroupKFold track and neither is used there.

### What is *not* done, even here

Protocol integrity inside the chosen protocol is maintained absolutely:

* Every choice — which candidates enter the ensemble, their weights, the
  decision threshold — is made on **inner out-of-fold predictions computed on
  training rows only**. The outer test fold is touched once, to score.
* No task identity feature. Task → label is near-deterministic in StressID
  (Stroop 0.77 stressed, Relax 0.13), so a task ID would be a label in
  disguise. `avail` (three modality-availability bits) is a weak task proxy and
  is therefore isolated in its own feature set and reported separately.

---

## 1. Method

### Features (`src/sota_features.py`)

Per recording, six fixed-width blocks, all disk-cached and winsorised to the
[0.5, 99.5] percentile per column:

| Block | Dim | Content |
|---|---|---|
| `physfeat` | 290 | neurokit2 HRV/EDA/RSP per 10 s window → 10 aggregates: mean, median, std, IQR, min, max, **linear slope**, last−first, first, last |
| `physraw` | 108 | ECG/EDA/RR waveform descriptors: moments, percentiles, diff energy, zero-crossing rate, Welch band powers, plus window-to-window drift |
| `audio` | 294 | log-mel per-band mean/std/p10/p50/p90 + delta stats, and spectral centroid / spread / flux / rolloff / flatness summarised over frames |
| `videostat` | 69 | coarse pixel statistics (the original baseline block) |
| `videofeat` | 744 | 4×4 regional motion dynamics + uniform-LBP texture → mean/std/max |
| `avail` | 3 | modality-availability bits (task proxy — isolated) |

The slope and last−first terms matter: a stress response *ramps*. A recording
whose EDA climbs through the task and one that starts high and decays have the
same mean and std, and the original aggregation could not tell them apart.

### Protocol (`src/sota.py`)

* **Outer**: `RepeatedStratifiedKFold` over recordings, 5 folds, seeded.
* **Inner**: 4-fold stratified over training rows. Produces one OOF probability
  vector per candidate, where a candidate is (feature set × view × model).
* **Selection**: bagged greedy ensemble selection with replacement (Caruana et
  al. 2004) on the inner OOF. Plain "average the top *k*" overfits the inner
  estimate badly with ~150 candidates and ~560 training rows — the top of the
  list is whoever got lucky. Greedy-with-replacement stops rewarding correlated
  candidates once one is in; bagging over random halves of the pool damps the
  rest of the selection variance.
* **Threshold**: tuned on inner OOF for macro F1. 0.5 is only optimal for a
  perfectly calibrated classifier, and macro F1 on a 47/53 split is sensitive
  to the operating point.

### Candidates (`src/sota_models.py`)

A *candidate* is anything that maps (train rows, test rows) to a probability per
test recording. One interface, three kinds, all competing on identical inner
OOF folds and all eligible for the same ensemble — which is the point, because
they fail differently:

* **`TabularCandidate`** — one feature vector per recording. The original setting.
* **`WindowCandidate`** — fits on *window* rows. A StressID recording is not one
  observation: it is a 60–90 s task cut into 16 overlapping 10 s windows, and
  the task label applies to all of them. Fitting at window level turns ~560
  training rows into ~9 000, then averages window probabilities back to a
  recording decision. The per-window label is noisier — not every second of a
  Stroop task is stressful — but the sample count grows 16×, and 700×1500 is
  exactly the ratio that has been binding on this dataset.
* **`TorchWindowCandidate`** — a masked sequence encoder on the GPU (~130 k
  parameters at `d_model=96`). Deliberately small: this repo already recorded
  that a 1.2 M-parameter transformer memorises 448 recordings instead of
  generalising, and the fix for that is not a bigger model. Its capacity goes
  into temporal aggregation — the shape of the response over the task — which
  is the one thing the tabular candidates structurally cannot express.

### Compute

The box is shared with two other training jobs. Budget: 6 of 12 cores, small
GPU footprint, and the inner sweep parallelised across 4 processes.

Measured on this hardware, on the real 700×1505 design matrix:

| Decision | Measurement | Outcome |
|---|---|---|
| XGBoost on the Quadro P1000 | 12 s CPU → 6 s GPU per fit, with the card already 95% busy from the other jobs | kept, on GPU |
| CatBoost on GPU | 143 s per fit, macro F1 0.642 | **dropped** — XGBoost scored 0.650 in 6 s |
| Serial inner sweep | >20 min for a *single* outer fold (~2.5 h/round) | **replaced** by a 4-process sweep |

The serial cost is why the harness was refactored onto candidate objects before
any result was collected: five rounds at 2.5 h each is not a loop, it is a
weekend. The refactor was verified behaviour-preserving — the smoke
configuration returns c364 macro F1 `0.5863077474990014` before and after,
digit for digit.

---

## 2. Rounds

### R1 — baseline, `raw` view only

Purpose: the number every later round must beat, with **no** subject-referenced
normalisation, so the contribution of the `rel`/`z` views can be attributed
cleanly in R2 rather than assumed.

Configuration: 8 feature sets × 8 models = 64 candidates, 5 outer folds,
3 inner folds, scopes `all700` and `c364`. Runtime 60 min.

| Scope | Macro F1 | Weighted F1 | Balanced acc | Accuracy | ROC AUC |
|---|---|---|---|---|---|
| **all700** | **0.7419 ± 0.0338** | 0.7426 | 0.7420 | 0.7429 | 0.8091 |
| `c364` | 0.6518 ± 0.0608 | 0.7302 | 0.6444 | 0.7473 | 0.6999 |

Per-fold `all700`: 0.6851 / 0.7500 / 0.7703 / 0.7637 / 0.7403.

The starting point already sits at or above both reference numbers — the origin
paper's 0.72 weighted F1 and this repo's earlier leaky-protocol 0.728 macro F1
— before any subject-referenced view has been used.

`c364` scores lower on macro F1 and *higher* on accuracy, which is what a
class-imbalanced subset looks like: the all-modality recordings are mostly the
speech tasks, which are mostly stressful. It is not a regression, and the two
scopes are not comparable to each other.

**Analysis — two things wrong, to be fixed in separate rounds.**

1. *The ensemble is not earning its keep.* It scores 0.7419 against its own best
   single member's 0.7429. Bagged greedy selection returned 37–53 members per
   fold, and the long low-weight tail dilutes rather than diversifies. Fixing
   this belongs in its own round, not folded silently into another change.
2. *The candidate pool is unbalanced.* The inner-CV top ten is entirely tree
   learners on wide fusion sets (`extratrees`/`lgbm`/`rf` over `all`,
   `all+avail`, `phys+audio`). Linear and kernel models never reach it, so they
   are spending sweep budget without contributing selections.

Top inner-CV candidates (mean over folds):

| Candidate | Inner macro F1 |
|---|---|
| `all+avail｜raw｜extratrees` | 0.7336 |
| `all｜raw｜extratrees` | 0.7331 |
| `all+avail｜raw｜lgbm` | 0.7329 |
| `all｜raw｜rf` | 0.7297 |
| `phys+audio｜raw｜extratrees` | 0.7293 |

### R2 — subject-referenced views  *(running)*

The single change: every feature block is additionally offered as `rel`
(centred on the participant's own Relax/Breathing baseline) and `z` (z-scored
within participant). 24 matrices × 8 models = 192 candidates. `all700` only, so
the view effect is isolated before spending a round on both scopes.

This is the lever expected to matter most on a subject-shared protocol, and it
is also the one most in need of the honesty note in §0: it is transductive, and
it is admissible here only because this track already permits the participant
to appear on both sides of the split.

_Result: pending._

---

## 3. Reference points

| Source | Protocol | Weighted F1 | Macro F1 |
|---|---|---|---|
| StressID paper, SVM + average-rule fusion | random 80/20 + SMOTE | **0.72** | not reported |
| This repo, `decision_fusion+rf` (prior work) | random 80/20 | 0.726 | 0.728 |
| This repo, best GroupKFold (different question) | subject-held-out | — | 0.519 |
| Majority class, `all700` | — | — | 0.344 |

`all700` is 700 recordings from 64 participants, 368 stressed / 332 not.
`c364` is the 364 recordings that carry all three modalities, where the
availability shortcut is constant and therefore carries no signal.
