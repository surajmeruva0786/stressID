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

### R2 — subject-referenced views → **0.7517** (+0.0099)

The single change: every feature block is additionally offered as `rel`
(centred on the participant's own Relax/Breathing baseline) and `z` (z-scored
within participant). 24 matrices × 8 models = 192 candidates, `all700` only, so
the view effect is isolated. Runtime 99 min.

| Metric (`all700`) | R1 | **R2** | Δ |
|---|---|---|---|
| Macro F1 | 0.7419 ± 0.034 | **0.7517 ± 0.032** | **+0.0099** |
| Weighted F1 | 0.7426 | 0.7525 | +0.0099 |
| Balanced acc | 0.7420 | 0.7517 | +0.0097 |
| Accuracy | 0.7429 | 0.7529 | +0.0100 |
| ROC AUC | 0.8091 | 0.8133 | +0.0042 |

Per-fold: 0.7064 / 0.7563 / 0.7966 / 0.7426 / 0.7567. The gain is not uniform —
fold 3 went *down* 0.021 — which is the expected shape for a +0.01 mean effect
against a ±0.03 fold spread. It is a real but modest improvement, not a
breakthrough, and it should be read that way.

**The finding that matters is in the inner ranking, not the headline.** Of the
top 25 inner candidates, **19 use `rel`, 6 use `raw`, and zero use `z`**:

| Candidate | Inner macro F1 |
|---|---|
| `all｜rel｜extratrees` | 0.7503 |
| `all｜rel｜lgbm` | 0.7474 |
| `all+avail｜rel｜extratrees` | 0.7471 |
| `all+avail｜rel｜lgbm` | 0.7457 |
| `all｜rel｜rf` | 0.7434 |

So the two subject-referenced views are not interchangeable, and the difference
is interpretable rather than incidental:

* **`rel` works.** Subtracting a participant's own resting baseline is
  physiologically motivated — what matters for stress is the *deviation* from
  that person's calm state, not the absolute level, and absolute levels differ
  enormously between people.
* **`z` fails.** Dividing by the participant's spread across their own
  recordings destroys exactly the signal being measured: a participant whose
  physiology swings a lot across tasks is swinging *because* some of those
  tasks stressed them. Normalising that away normalises away the label.

`z` cost a third of this round's 99 minutes and contributed nothing. It is
dropped from R3 onward.

Secondary result: the ensemble now clearly beats its own best single member
(0.7517 vs 0.7337), reversing R1's defect #1. More candidate diversity was
enough to fix it — but the member count also grew to 63–92 per fold, so the
dilution question is still open and gets tested directly in R3.

### R3 — ensemble pruning → 0.7505 (pruning **+0.0041**, but net −0.0012)

Keeps only the members covering 90% of cumulative greedy weight, with the
unpruned blend scored **in the same run on the same folds** as a reference.
Views `raw,rel`; 128 candidates; `all700`; runtime 67 min.

| Blend (identical folds) | Members | Macro F1 |
|---|---|---|
| **Pruned, 90% cumulative weight** | 40.4 | **0.7505** |
| Unpruned | 66.0 | 0.7464 |
| Best single member | 1 | 0.7337 |

Pruning is worth **+0.0041** and cuts the member count by a third. Keep it.

**But R3's headline is 0.0012 *below* R2, and the in-run reference explains
why — by contradicting the conclusion I drew in R2.**

R2's unpruned blend (with `z`) scored 0.7517. R3's unpruned blend (without `z`)
scored 0.7464. So **dropping `z` cost −0.0053**, and pruning then recovered
+0.0041 of it. `z` was contributing to the ensemble *as diversity* even though
not one `z` candidate reached the top 25 by individual inner score.

That is a real correction to R2's write-up. The accurate statement is narrower
than the one I made:

* ✅ `z` is a poor *standalone* representation — physiologically explicable, and
  the top-25 evidence supports it.
* ❌ "`z` contributed nothing" was wrong. A candidate can be individually weak
  and still earn its place in a blend by being wrong in a *different direction*
  from the strong candidates. Individual inner rank is the wrong instrument for
  deciding ensemble membership; that is precisely what greedy selection is for.

The cost is −0.0053 against a ±0.035 fold spread, i.e. well inside noise, for
+50% runtime. So `z` stays out of the fast exploratory rounds and comes back
for the final run, where the budget is worth spending.

**Method note for the write-up.** The in-run reference is what made this
visible. Had R3 been compared against R2 across runs, "pruning helps by
−0.0012" would have been recorded, which is both wrong and the wrong sign.

### R4 — window-level and GPU sequence candidates → **0.7572** (+0.0066)

Adds two candidate families to the existing recording-level pool, which is left
untouched so the newcomers must earn selection against it. R4 differs from R3 in
exactly this one respect — same views, pruning, seed and folds — so the
cross-run delta is clean. Runtime 84 min.

| Metric (`all700`) | R3 | **R4** | Δ |
|---|---|---|---|
| Macro F1 | 0.7505 ± 0.035 | **0.7572 ± 0.046** | **+0.0066** |
| Unpruned blend | 0.7464 | 0.7540 | +0.0077 |
| Weighted F1 | 0.7512 | 0.7580 | +0.0068 |
| Accuracy | 0.7514 | 0.7586 | +0.0071 |
| ROC AUC | 0.8154 | 0.8228 | +0.0074 |

Per-fold: 0.6994 / 0.7571 / **0.8268** / 0.7494 / 0.7531. Fold 2 is the highest
single fold of the campaign. Fold variance also rose (std 0.035 → 0.046).

The unpruned blend moved as well, so the gain is not an artefact of pruning
interacting with a larger pool.

**Window candidates earn their place on individual merit.**
`win-raw｜mean｜extratrees` and `win-raw｜trimmed｜extratrees` rank 6th and 8th
of ~148 candidates — above every recording-level model except the five best.
Training on ~9 000 window rows instead of 560 recording rows relieves exactly
the constraint identified at the start: 700 recordings against ~1 500 columns.

**Whether the GPU sequence models contributed is, as of R4, unknown.** They took
no top-25 slot — but R3 established that this does not answer the question, and
the report was silently dropping the field that does. Which is a defect worth
recording:

> Per-fold metrics and per-fold *selections* were being concatenated into one
> table, so the report writer took its columns from the first row and dropped
> `top_members` entirely — the only record of which candidates the ensemble
> actually chose. Fixed, with a new breakdown of ensemble weight by candidate
> family.

The fix immediately justified itself. In the smoke configuration,
`seq-rel｜gru｜torch` ranked **last** individually (0.5594 against 0.6002) and
still took **59% of the ensemble weight** on one fold. A candidate's individual
inner score and its ensemble contribution are close to unrelated quantities.

### R5 — final configuration → **0.7604** (best)

Everything the campaign established, combined: views `raw` + `rel` + `z`
(restored), window-level tree candidates on `raw`/`rel`, GPU sequence candidates
(`gru`, `attn`), ensemble pruned to 90% cumulative weight, both scopes.
Runtime 3.1 h.

| Metric | `all700` R1 | `all700` **R5** | Δ vs R1 |
|---|---|---|---|
| **Macro F1** | 0.7419 ± 0.034 | **0.7604 ± 0.037** | **+0.0185** |
| Weighted F1 | 0.7426 | 0.7612 | +0.0186 |
| Balanced acc | 0.7420 | 0.7602 | +0.0182 |
| Accuracy | 0.7429 | 0.7614 | +0.0185 |
| ROC AUC | 0.8091 | 0.8226 | +0.0135 |

Per-fold: 0.7064 / 0.7642 / 0.8118 / 0.7637 / 0.7559. Fold variance came back
down (0.046 → 0.037) while the mean rose, which is the shape a genuine
improvement should have.

#### The GPU sequence models do not earn their place

With the selection table fixed, the question R4 could not answer is now settled.
Mean share of ensemble weight across the five `all700` folds:

| Candidate family | Weight share | Folds present |
|---|---|---|
| Recording-level | **88.4%** | 5 / 5 |
| Window-level | **10.8%** | 5 / 5 (up to 19.3%) |
| **GPU sequence (`gru`/`attn`)** | **0.7%** | **2 / 5** |

Window-level candidates are real contributors — present in every fold, and the
largest single member in fold 3 (`win-raw｜mean｜extratrees`, 9% alone). The
GPU sequence models are not: 0.7% mean weight, and **zero weight in three of
five folds**. They were given a fair test — two architectures, two views, the
same inner folds as everyone else — and the ensemble declined them.

This is consistent with what this repo already found on the other track, and it
is the same lesson twice: on 700 recordings, learned sequence encoders lose to
tree learners on aggregated descriptors. The right conclusion is not "tune the
network harder", it is that this dataset is too small for representation
learning to pay, and the window-level trick captures most of what temporal
structure was there anyway.

#### The `c364` scope got *worse* over the campaign

| Scope | R1 | R5 | Δ |
|---|---|---|---|
| `all700` | 0.7419 | **0.7604** | **+0.0185** |
| `c364` | **0.6518** | 0.6435 | **−0.0083** |

Everything tuned across five rounds was selected while watching `all700`, and
`c364` — 364 recordings, class-imbalanced, no availability shortcut — drifted
down. Two readings, and honesty requires holding both:

1. The `c364` change (−0.008) is small against its own fold spread (±0.056), so
   this may be noise.
2. But the *direction* is systematic across rounds, and that is what a
   configuration tuned on one scope and applied to another looks like. The
   headline gain is partly a gain at fitting `all700` specifically.

`all700` remains the protocol-matched number, since the origin paper also
evaluates over all recordings. But `c364` is the scope where the
modality-availability shortcut is constant, so it is the more conservative
measure of whether the model learned anything about *stress*.

### R6/R7 — measuring the selection bias, and the honest delta

All five rounds selected against the same five outer folds (seed 42), so the
campaign maximum over that partition is optimistically biased. R6 re-runs the
frozen R5 configuration on unseen seed-101 partitions; R7 runs the R1 *baseline*
on those same partitions, supplying the term R6 alone was missing — comparing
0.7467 (seed 101) against 0.7419 (seed 42) would compare two different
partitions, not two configurations.

| Configuration | Partition | `all700` macro F1 |
|---|---|---|
| Final (R5) | seed 42 — **searched** | 0.7604 |
| Final (R6) | seed 101 — **unseen** | 0.7467 |

**Measured campaign selection bias: +0.0137 macro F1.** Five rounds of choosing
against one partition bought that much apparent score that does not transfer.

**This also corrects the concern raised in R5.** R5's write-up claimed `c364`
had declined systematically and that part of the gain was specific to fitting
`all700`. On the unseen partition `c364` comes out *higher* (0.6721 vs 0.6435),
its fold spread halves (±0.056 → ±0.036), and the final configuration beats the
baseline on `c364` too (+0.0117). The seed-42 `c364` folds were simply hard.
The accurate statement is that `c364` is highly partition-sensitive — what 364
class-imbalanced recordings split five ways should look like — not that the
campaign overfitted one scope.

### R8 — the powered comparison → **significant**

R6/R7 gave +0.0195 at paired *p* = 0.22 on five folds: consistent in sign across
every metric and both scopes, but underpowered, with the mean pulled by a single
+0.064 fold. R8 re-runs both configurations at 3 repeats (**15 outer folds**) on
unseen seed-101 partitions.

| | 15 unseen folds |
|---|---|
| Final configuration | **0.7484 ± 0.0331** |
| Baseline configuration | 0.7322 ± 0.0301 |
| **Mean paired difference** | **+0.0162** |
| 95% CI | **[+0.0013, +0.0311]** |
| Paired *t* | **p = 0.035** |
| Wilcoxon signed-rank | **p = 0.031** |
| Folds improved | 10 / 15 |
| Cohen's *dz* | 0.60 (medium) |

Tripling the fold count *shrank* the point estimate (+0.0195 → +0.0162) while
making it significant. That is the signature of a real effect that was
underpowered, rather than a fluke regressing to the mean — a fluke would have
shrunk toward zero, not tightened around a positive value.

**How to state this.** The lower CI bound is +0.0013, so the effect is
significant but its *magnitude* is not tightly pinned. The defensible claim is
"a small but statistically significant improvement over a strong baseline
(+0.016 macro F1, paired *p* = 0.035, n = 15 folds)" — not "a substantial gain".

---

## 2b. Binary campaign — conclusion

Eight rounds, ~14 h of compute. The campaign is **closed for the binary task**;
what follows is the state to cite.

### The number

| | Value |
|---|---|
| **Headline** (`all700`, unseen folds, n = 15) | **macro F1 0.7484 ± 0.033** |
| Improvement over baseline | **+0.0162**, 95% CI [+0.0013, +0.0311] |
| Significance | paired *t* p = 0.035, Wilcoxon p = 0.031, 10/15 folds |
| On the searched partition (do **not** cite) | 0.7604 |
| Origin paper | 0.72 weighted F1 |
| Majority class | 0.344 |

### What worked, in order of contribution

1. **Subject-relative baseline correction (`rel`)** — express each recording as
   a deviation from that participant's own resting state. 19 of the top 25
   candidates use it. Physiologically motivated: stress is a *deviation*, and
   absolute levels differ enormously between people.
2. **Window-level training** — fit trees on ~9 000 window rows rather than 560
   recording rows, then average probabilities back to a recording. 10.8% of
   ensemble weight, present in every fold. Directly relieves the
   700-recordings-by-1500-columns ratio that binds this dataset.
3. **Bagged greedy ensembling with inner-OOF threshold tuning**, pruned to 90%
   cumulative weight (+0.0041, measured in-run).

### What did not work — the negative results

* **GPU sequence models (`gru`, `attn`) were declined by the ensemble**: 0.7%
  mean weight, zero weight in three of five folds, after a fair test against
  identical inner folds. On 700 recordings, learned sequence encoders lose to
  tree learners over aggregated descriptors. This repo now has that result
  twice, on two different protocols.
* **Subject-z (`z`) as a standalone representation** — never ranked. It did
  contribute as ensemble diversity, which is a different thing, and confusing
  the two was one of this campaign's errors.
* **CatBoost** — 143 s per GPU fit for a lower score than XGBoost's 6 s.

### Errors made and corrected

Recorded because the corrections are more informative than the claims were:

| Claim | Correction | How it surfaced |
|---|---|---|
| "`z` contributed nothing" (R2) | It contributed −0.0053 worth of ensemble diversity | R3's in-run unpruned reference |
| "`c364` is degrading systematically" (R5) | One hard partition; `c364` is *higher* on unseen folds | R6 on a fresh seed |
| "pruning helps by −0.0012" (implied by cross-run comparison) | Pruning helps by **+0.0041** | Scoring both blends inside one run |

The common thread: **every one of these was a cross-run comparison done by eye.**
Two runs differing in more than one respect cannot attribute a delta. In-run
references and paired tests on identical folds fixed all three, and
`src/sota_summary.py` now generates them so it is not done by hand again.

### Limitations to state in the paper

1. **Subject leakage inflates every number here**, by a margin measured
   separately in `LEAKY_PROTOCOL.md` (+0.098 macro F1 on physiology, ≈0 on a
   subject-identity-free control). The deployment-realistic estimate remains
   ≈0.52 on the GroupKFold track.
2. **Campaign selection bias is +0.0137**, measured. Cite 0.748, not 0.760.
3. **The protocol is comparable to the origin paper's, not identical** — theirs
   is random 80/20 with SMOTE, ours subject-shared 5-fold without. Describe as
   "competitive under a comparable protocol", not a like-for-like win.
4. **`rel` and `z` are transductive.** No labels are read, but test rows
   contribute to their own participant's statistics. Admissible only on this
   track.

---

## 2bb. All three StressID targets

StressID ships three targets. The campaign optimised the binary one; these are
the other two, run with the same recipe (raw + `rel` views, window-level
candidates, bagged greedy selection on inner OOF, 90% pruning) on unseen
seed-101 partitions.

| Target | Metric | Result | Reference point |
|---|---|---|---|
| **Binary stress** | macro F1 | **0.7484 ± 0.033** | paper 0.72 weighted F1; majority 0.344 |
| **3-class affect** | macro F1 | **0.6214 ± 0.014** | chance 0.333 |
| | accuracy | 0.6471 ± 0.018 | |
| **Stress score (0–10)** | Pearson *r* | **0.673 ± 0.030** | |
| | Spearman *r* | 0.660 ± 0.031 | |
| | RMSE | **1.857 ± 0.107** | predicting the mean = 2.488 |
| | R² | 0.440 ± 0.040 | |

Three things worth noting rather than just tabulating:

* **The regression is the most interpretable result.** The target is a
  self-reported 0–10 rating with SD 2.488, so predicting the mean for every
  recording gives RMSE 2.488. The model reaches 1.857 — a **25% error
  reduction**, explaining 44% of the variance in what people said about their
  own stress.
* **Pearson and Spearman agree closely** (0.673 vs 0.660), so the relationship
  is essentially monotonic rather than being carried by a handful of extreme
  recordings.
* **The affect task is the most stable**, with a fold spread of ±0.014 against
  the binary task's ±0.033 — which is what a better-balanced target
  (253/202/245) should look like.

## 2c. StressID has a *second* protocol confound: recording duration

Found while adding whole-recording physiology, and caught before it reached any
reported number. It is worth as much attention as the score.

### The finding

StressID task durations are near-deterministic per task:

| Task | Duration | Stress rate |
|---|---|---|
| Breathing | 177 s | 0.17 |
| Video1 | 171 s | 0.29 |
| Relax | 148 s | 0.13 |
| Video2 | 117 s | 0.40 |
| Counting 1–3, Math, Reading, Speaking, Stroop | **all 59 s** | 0.55–0.77 |

Low-stress tasks are long; every high-stress task is 59 seconds. So **recording
length is the task label**, and the task label is very nearly the stress label.

> **A random forest given duration alone — one scalar, no physiology of any
> kind — scores 0.5923 ± 0.018 macro F1**, against a 0.344 majority baseline and
> this campaign's full multimodal 0.7484. Duration by itself reaches **79%** of
> the complete pipeline's score.

### How it was caught

The first version of `physio_global.py` produced 33 features. Its single
strongest stress discriminator, by a wide margin, was `g_n_beats` — Cohen's
*d* 0.74, *p* = 4×10⁻²³, ahead of every genuine physiological marker. It
correlates **0.937 with recording duration**. A beat count is heart rate
multiplied by elapsed time, so it was measuring the clock wearing a stethoscope.

### The fix

Every feature in the block is now a rate, a mean, or a per-second slope, and
never a count:

| Removed / changed | Why |
|---|---|
| `g_n_beats`, `g_n_breaths`, `g_scr_count` | pure duration × rate |
| `g_scr_rate_min` | kept — count *per minute* is duration-free |
| `g_phasic_auc` → `g_phasic_mean_abs` | running sum → per-sample mean |
| all slopes | per-sample → **per second** (the same total drift over 59 s and 177 s would otherwise differ threefold) |

After the fix the largest feature-duration correlation falls from **0.937 to
0.317**, and the top discriminators become textbook stress physiology:

| Feature | Cohen's *d* | Expected direction |
|---|---|---|
| `g_phasic_mean_abs` (EDA phasic activity) | +0.41 | ↑ sympathetic arousal ✓ |
| `g_phasic_std` | +0.37 | ✓ |
| `g_hr_mean` | +0.36 | ↑ heart rate ✓ |
| `g_scl_mean` | +0.35 | ↑ skin conductance ✓ |
| `g_lf` | −0.26 | ✓ |

The residual ~0.3 correlations are not leakage: heart rate genuinely differs
between a long relaxation task and a short stressful one, and that is the
signal, not the shortcut.

### Why this matters beyond our pipeline

This is the **second** protocol artefact this repository has measured on
StressID, and the two are the same species of problem — a property of the
*experimental protocol* masquerading as a property of the *participant*:

| Confound | Control that isolates it | Macro F1 from the confound alone |
|---|---|---|
| Modality availability | `availability_only` (3 bits) | ~0.697 |
| **Recording duration** | **duration-only (1 scalar)** | **0.592** |
| Subject identity | subject-ID probe | 41.5× chance from physiology |

Any StressID result that does not report controls of this kind is difficult to
interpret, because a model with access to the raw recording has access to all
three for free. Ours reports them.

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
