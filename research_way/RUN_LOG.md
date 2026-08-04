# StressID — Complete Run Log

> **Maintained:** 2026-08-04
> Every training and evaluation run performed on this project, in chronological
> order, with configuration, result, and verdict. Nothing has been pruned —
> failed runs are recorded with the same detail as successful ones, because the
> count of attempts is part of the result (see §7).
>
> Companion documents:
> `../RESEARCH_PROGRESS.md` (narrative + findings) ·
> `RESULTS_AND_COMPARISON.md` (vs. prior work) ·
> `reports/<run>/report.{md,pdf}` (per-run reports) ·
> `reports/runs_index.csv` (machine-readable registry)

---

## 1. How to read this log

### 1.1 The primary metric

**`complete364_macro_f1`** — macro F1 on the **364 recordings that natively have
all three modalities**, under **subject-grouped 5-fold cross-validation**.

Every design choice in that sentence matters:

| Choice | Why |
|---|---|
| **364-recording subset** | Modality availability leaks the label on the full 700 (P(stress\|audio)=0.709 vs 0.311). On this subset every recording has all three modalities, so availability is constant and carries no signal. |
| **Macro F1** | The subset is 71.7% positive. Weighted F1 rewards predicting the majority; macro F1 does not. The same model reads 0.628 weighted vs 0.487 macro — a 0.14 gap that is pure imbalance. |
| **Subject GroupKFold** | No subject appears in both train and test. Random splits inflate physiological models by +0.127 because they memorise individual baselines. |

**Reference points on this subset (n=364, 71.7% positive):**

| Classifier | macro F1 | weighted F1 | balanced acc | accuracy |
|---|---|---|---|---|
| always predict "stressed" | **0.418** | 0.599 | 0.500 | 0.717 |

Any macro F1 at or below 0.418 means the model has learned nothing usable.

### 1.2 Scope labels used below

- **all-700** — every recording; contains the availability shortcut. Useful only
  for measuring the shortcut, never for claiming performance.
- **complete-364** — the leakage-free evaluation subset. **The number that counts.**
- **complete-64** — the equivalent subset of the early 16-subject pilot.

---

## 2. Deep-learning runs

### 2.1 `smoke` — pipeline smoke test (2026-07-28)

| | |
|---|---|
| Purpose | Verify the pipeline executes end to end |
| Data | 16 subjects, 6 tasks, 2 folds, 6 epochs |
| Result | Ran to completion; numbers meaningless at this scale |
| Verdict | **Infrastructure OK** |

### 2.2 `small` — 16-subject pilot (2026-07-28)

| | |
|---|---|
| Data | 16 subjects, 6 tasks, 96 recordings, 4-fold GroupKFold, 3 seeds |
| Model | MST, `d_model=128`, fusion×1, temporal×2, dropout 0.2, 40 epochs |
| Artefacts | `results/small/` |

| scope / variant | macro F1 | majority ref | margin |
|---|---|---|---|
| complete-64, static | 0.4519 | 0.4234 | **+0.028** |
| complete-64, temporal | 0.4169 | 0.4234 | **−0.007** |
| all-96, temporal | 0.7047 | 0.3469 | +0.358 |

**Verdict: at chance.** The temporal model was *below* the majority reference on
the leakage-free subset. The high all-96 number is the availability shortcut, not
skill. Attributed to subset size — which motivated the full-corpus run.

### 2.3 `full` run 1 — first full corpus (2026-07-29)

| | |
|---|---|
| Data | **64 subjects, 11 tasks, 700 recordings**, 5-fold GroupKFold |
| Design | 3 seeds × 5 folds × {temporal, static} = **30 fold-models** |
| Model | MST, `d_model=128`, fusion×1, temporal×2, dropout 0.2, 40 epochs, AdamW+OneCycle |
| Runtime | **10,640 s (~3.0 h)** on Quadro P1000 (4 GB) |
| Artefacts | `results/full_run1_20260729/` (frozen) |

| scope / variant | macro F1 [95% CI] | accuracy |
|---|---|---|
| **complete-364, temporal** | **0.4850** [0.451, 0.523] | 0.694 |
| **complete-364, static** | **0.4759** [0.452, 0.504] | 0.667 |
| all-700, temporal | 0.6710 | 0.679 |
| all-700, static | 0.6540 | 0.659 |

**This run produced the project's headline finding** — see §6.1.

**Weights were not saved.** `train_one_fold` kept the best state in memory and
persisted only predictions. Discovered 2026-07-30; §2.4 exists to fix it.

### 2.4 `full` run 2 — retrain with checkpointing (2026-07-30)

| | |
|---|---|
| Change | **Only** the addition of checkpoint persistence — identical config, splits, seeds |
| Runtime | 11,453 s (~3.2 h) |
| Output | **30 checkpoints, 129 MB**, `results/full/checkpoints/` + `index.csv` |

| metric | run 1 | run 2 | Δ |
|---|---|---|---|
| complete-364, static | 0.4759 | 0.4759 | **0.0000** |
| complete-364, temporal | 0.4850 | 0.4850 | +0.0000 |
| all-700, static | 0.6540 | 0.6540 | **0.0000** |
| all-700, temporal | 0.6710 | 0.6726 | +0.0016 |

**Verdict: reproduces.** Static is bit-identical; temporal drifts ±0.002 from
cuDNN nondeterminism in the attention/LSTM kernels. Every conclusion held.

Each checkpoint is self-contained: weights, full `Config`, `subject_vocab`,
temporal flag, best epoch/val-F1, the fold's train/val/test subject lists, and
the **fold-specific train-only audio/video normalisation stats** (reusing another
fold's stats at inference would reintroduce the leakage the split prevents).

### 2.5 `a1a2` — domain features + capacity reduction (2026-08-02)

| | |
|---|---|
| A1 | Physio raw waveform → **29 neurokit2 HRV/EDA/RSP descriptors** per window |
| A2 | `d_model` 128→64, fusion×1, temporal 2→1, dropout 0.2→0.4, early stop patience 8 |
| Params | **311,685 (25% of 1,243,109)** |
| Runtime | 3,798 s (63 min) — 3× faster |
| Output | 30 checkpoints, 35 MB |

| scope / variant | control (`full`) | `a1a2` | Δ |
|---|---|---|---|
| **complete-364, temporal** | 0.4850 | **0.4439** | **−0.041** |
| **complete-364, static** | 0.4759 | 0.4752 | −0.001 |
| all-700, temporal | 0.6726 | 0.6574 | −0.015 |
| all-700, static | 0.6540 | 0.6445 | −0.010 |

**Verdict: FAILED.** Temporal lost 0.041 and fell *below* static, reversing the
control's ordering.

**Two errors recorded (§7.2, §7.3):** A1's rationale was invalid, and five knobs
changed in one run making the result unattributable.

Diagnostic that explains the failure: **early stopping fired on 30/30 folds**,
and mean best epoch moved only 9.6 → 6.9. The model peaks early and decays
*regardless of size* — capacity was never the binding constraint.

---

## 3. Classical / hybrid runs

All use nested CV: model and feature-set selection run on an **inner GroupKFold
over training subjects only**. Zoo = 7 models (logreg ×2, SVC ×2, RF, ExtraTrees,
HGB) × 7 feature sets = 49 configs per fold.

### 3.1 The diagnostic that redirected the campaign (2026-08-03)

Before any classical run: *what do simple models score on the leakage-free
subset?* All prior baselines had been measured on all-700, where availability
inflates them. On complete-364 under GroupKFold:

| model | macro F1 | |
|---|---|---|
| video + SVC | **0.544** | ← beats the deep model |
| video + logreg | 0.538 | ← |
| all-features + HGB | 0.532 | ← |
| video + RF | 0.510 | ← |
| video + HGB | 0.508 | ← |
| physfeat+audio+video + HGB | 0.504 | ← |
| physio_raw + logreg | 0.486 | ← |
| **deep MST-temporal** | **0.485** | |
| deep MST-static | 0.476 | |

**Seven classical configurations beat a 1.2 M-parameter transformer.** The deep
architecture was never the ceiling — it loses to an SVC on 32×32 pixel statistics.

> These were read off outer-fold scores, i.e. test-set fishing. Nested CV later
> discounted the best to 0.533 — the 0.544 was ~0.011 optimistic. Recorded
> because the gap is the point.

### 3.2 `c1_nested_ensemble` (2026-08-03, 291 s)

| | |
|---|---|
| Design | Nested GroupKFold; inner CV ranks 49 configs; **soft-vote top-3, k fixed a priori** |
| Result | **macro F1 0.5327** · weighted 0.643 · balanced acc 0.547 · acc 0.679 |
| Single best model (reference) | 0.4870 |

**Verdict: BEST NON-PERSONALISED RESULT.** +0.048 over the deep model. The gain
comes from ensembling — the single best model alone matches the deep model.

### 3.3 `c2_subject_relative` (2026-08-03, 294 s)

| | |
|---|---|
| Change | Every feature expressed relative to that subject's own **Relax/Breathing baseline** |
| Result | ensemble **0.5199** · single model **0.5371** |

**Verdict: mixed.** Calibration substantially helps the single model
(0.487 → 0.537) but hurts the ensemble.

> **This is a *personalised* setting** and is reported separately. Relax/Breathing
> carry no audio, so they are outside the 364-recording evaluation subset — no
> evaluated sample is touched and no label is read. Legitimate calibration
> matching a real deployment, but it answers a different question than c1.

### 3.4 `c3_inner_selected_k` (2026-08-03, 298 s)

| | |
|---|---|
| Change | Ensemble size k chosen **per fold by inner CV** (grid 1/2/3/5/8) instead of fixed |
| Result | **0.4898** (mean chosen k = 2.8) · single 0.5088 |

**Verdict: FAILED, −0.043 vs c1.** Adaptive selection on ~290 training samples
costs more in variance than the flexibility buys. Landed back at the deep
model's 0.485.

C3 was run to *remove* a post-hoc choice and instead surfaced a larger one — see
§7.4.

### 3.5 `c4_video_features` (2026-08-03, 549 s) — **INVALID**

| | |
|---|---|
| Intent | 248 new video features: 4×4 regional dynamics + per-region uniform LBP |
| Result | 0.4979 |

**Verdict: INVALID — do not cite.** The run's notes claimed k was fixed at 3, but
the code still used C3's adaptive selection (`mean_chosen_k = 3.4`), confounding
the feature change with a mechanism worth −0.043. Caught in the output and
re-run as c4b. Retained in the registry because the attempt consumed an
evaluation on the same folds and therefore counts toward the multiple-comparisons
burden.

### 3.6 `c4b_video_features_fixedk` (2026-08-03, 531 s)

| | |
|---|---|
| Change | Same 248 video features, **k = 3 fixed** — directly comparable to c1 |
| Result | **0.5100** · single 0.5116 |

**Verdict: FAILED, −0.023 vs c1.** Richer features on the *strongest* modality
made it worse. Strong evidence the limit is data, not description.

### 3.7 `c5_confirm_*` — held-out confirmation (2026-08-03)

All five stored outer folds had been reused by every iteration, so the campaign
maximum over them is optimistically biased. The **frozen c1 protocol** (7
original feature sets, 7 models, k=3 fixed, no personalisation) was re-evaluated
on **three subject partitions generated from seeds never used during the search**.

| run | partition seed | macro F1 | runtime |
|---|---|---|---|
| `c5_confirm_seed101` | 101 | 0.5035 | 281 s |
| `c5_confirm_seed202` | 202 | 0.5499 | 278 s |
| `c5_confirm_seed303` | 303 | 0.5043 | 287 s |
| **mean** | | **0.5192** (sd 0.0265) | |

**Search maximum 0.5327 → confirmed 0.5192; the search was +0.0135 optimistic**,
almost exactly the drift predicted when the bias was flagged.

**The reportable number is 0.519, not 0.533.** The 0.5499 from seed 202 is the
best single partition and must **not** be quoted alone — that would repeat the
error the confirmation exists to correct.

---

## 4. All runs ranked

| # | run | date | macro F1 (complete-364) | vs. majority | verdict |
|---|---|---|---|---|---|
| 1 | `c5_confirm_seed202` | 08-03 | 0.5499 | +0.132 | single partition — **do not quote alone** |
| 2 | `c1_nested_ensemble` | 08-03 | 0.5327 | +0.115 | search maximum |
| 3 | `c2_subject_relative` | 08-03 | 0.5199 (single 0.5371) | +0.102 | personalised setting |
| — | **`c5` confirmation mean** | 08-03 | **0.5192** | **+0.101** | **REPORTABLE RESULT** |
| 4 | `c4b_video_features_fixedk` | 08-03 | 0.5100 | +0.092 | failed |
| 5 | `c5_confirm_seed303` | 08-03 | 0.5043 | +0.086 | confirmation |
| 6 | `c5_confirm_seed101` | 08-03 | 0.5035 | +0.086 | confirmation |
| 7 | `c4_video_features` | 08-03 | 0.4979 | +0.080 | **invalid** |
| 8 | `c3_inner_selected_k` | 08-03 | 0.4898 | +0.072 | failed |
| 9 | `deep_full_temporal` | 07-30 | 0.4850 | +0.067 | deep control |
| 10 | `deep_full_static` | 07-30 | 0.4759 | +0.058 | deep control |
| 11 | `deep_a1a2_static` | 08-02 | 0.4752 | +0.057 | failed |
| 12 | `deep_a1a2_temporal` | 08-02 | 0.4439 | +0.026 | failed |
| — | *majority baseline* | — | *0.4180* | — | *reference* |

**Confirmed gain of the classical pipeline over the deep model: +0.034.**

---

## 5. Compute summary

| Run | Wall time | Output |
|---|---|---|
| preprocess (full corpus, cached) | ~1 h | 700 × `.npz`, video decode dominates |
| physfeat cache (A1) | 65 s | 700 × 29 features/window |
| videofeat cache (C4) | ~25 s | 700 × 248 features/window |
| `full` run 1 | 10,640 s | 30 fold-models (weights lost) |
| `full` run 2 | 11,453 s | 30 checkpoints, 129 MB |
| `a1a2` | 3,798 s | 30 checkpoints, 35 MB |
| `c1`–`c5` (8 runs) | ~2,800 s total | reports + registry |
| **Total** | **≈ 8.5 h GPU + ~1.5 h CPU** | |

---

## 6. Findings, by strength of evidence

### 6.1 Modality availability leaks the label — **the headline**

Audio exists only for the 7 speaking tasks, which are disproportionately the
stressful ones: **P(stress \| audio present) = 0.709** (378 recs) vs.
**0.311** (322 recs).

A classifier given **only three presence bits** — audio? video? physio? — and
zero signal content scores **0.697 macro F1 on all-700**, beating:

| | macro F1 |
|---|---|
| **`availability_only` (3 bits, no signal)** | **0.697** |
| decision fusion (RF) | 0.688 |
| audio (logreg) | 0.686 |
| feature fusion (RF) | 0.685 |
| **MST-temporal (1.2 M params)** | **0.671** |
| physio (logreg) | 0.565 |

**Internal validity check:** `availability_only` inflates by **−0.002** under
leaky random splits — exactly the zero it must be, since presence bits carry no
subject identity. Physio inflates by **+0.127**. That contrast is what makes the
leakage measurement credible rather than a resampling artefact.

### 6.2 A leakage-free ceiling of ~0.52, confirmed out-of-search

Five convergent nulls support this being a **data** ceiling, not a modelling
failure:

1. Temporal modelling: +0.011, p = 0.677
2. Cross-modal fusion: 0.671 vs. 0.688 for plain decision fusion — *behind*
3. Missing-modality robustness: curves flat-to-rising, nothing survives Bonferroni
4. Capacity/representation (A1+A2): −0.041
5. Richer video features on the strongest modality: −0.023
6. Adaptive model selection: −0.043

When better features and better selection both *hurt*, the model is not
underfitting — the data is exhausted.

### 6.3 We did not reach published SOTA, and matching it would be meaningless

The origin paper reports **0.72 weighted F1 / 0.65 balanced accuracy** under
random 80/20 task splits (subjects on both sides) with SMOTE. Under **that**
protocol our own classical baselines already reach **0.740 weighted F1**.

So "beating SOTA" has been available the whole time for the price of adopting the
leaky protocol — and it would mean nothing. Under a protocol where no subject
appears in both train and test, the ceiling is ~0.52 and this project reached it.

---

## 7. Mistakes and corrections — recorded deliberately

### 7.1 Weights discarded for the first full run
Three hours of training produced no saved model. Found only because the terminal
was closed and the state had to be reconstructed. Fixed by `src/checkpoint.py`;
cost one full retrain.

### 7.2 A1's rationale compared a leaky number to a leakage-free one
Justified by "the paper's physio features reach 0.73 where our encoder gets
0.54–0.57" — but 0.73 is the paper's **leaky** figure. Adjusted for the +0.127
inflation, real headroom was ~0.03, not ~0.19. **This is the exact error the
project exists to document, committed by the person documenting it.**

### 7.3 Five variables changed in one run
A1 and A2 were bundled "because they move in the same direction". The resulting
−0.041 is unattributable. Saved one run, cost the ability to interpret it.

### 7.4 Campaign-level selection bias
Every iteration scored on the same five outer folds, keeping the best. Each run
individually unbiased; **the maximum over runs is not.** Mitigated by logging all
runs including failures, and by the §3.7 confirmation on fresh partitions —
which measured the optimism at +0.0135.

### 7.5 A run whose notes did not match its code
`c4`'s notes claimed k fixed at 3; the code used adaptive k. Caught by reading
`mean_chosen_k = 3.4` in the output rather than trusting the label. Re-run as
`c4b`. Both retained.

### 7.6 PowerShell `$?` silently skipped the evaluate stage
`... --stage train ; if ($?) { ... --stage evaluate }` — PowerShell 5.1 sets `$?`
false when a native exe writes to stderr under `2>&1`, and torch emits
`UserWarning`s. Left `results/full/` with fresh predictions beside day-old
evaluation CSVs. **Chain on `$LASTEXITCODE`.**

---

## 8. Reproducing any run

```bash
cd research_way

# caches (once)
python run_full.py --stage preprocess --workers 6     # ~1 h, video decode
python run_full.py --stage physfeat  --workers 6      # 65 s

# deep models
python run_full.py --stage train --variant full       # ~3.2 h -> results/full/
python run_full.py --stage evaluate --variant full
python run_full.py --stage train --variant a1a2       # ~1 h  -> results/a1a2/

# classical (each ~5 min, writes reports/<name>/)
python -m src.classical --run-name c1_nested_ensemble
python -m src.classical --run-name c2_subject_relative --subject-relative
python -m src.classical --run-name c3_inner_selected_k
python -m src.classical --run-name c4b_video_features_fixedk --fixed-k
python -m src.classical --run-name c5_confirm_seed101 --fixed-k --c1-only --splits-seed 101

# load a trained deep model
python -c "from src.checkpoint import load_model; m,meta = load_model('results/full/checkpoints/temporal_seed1_fold4.pt')"
```

Splits are fixed in `data/splits_full.json`. `results/` is gitignored (weights,
predictions, caches); `reports/` is tracked and committed.

**Environment:** Windows 11 · Python 3.11 · torch 2.6.0+cu124 · CUDA on Quadro
P1000 (4 GB) · scikit-learn 1.8.0 · neurokit2 0.2.12 · scikit-image 0.26.0

---

## 9. What would actually move the number

Not architecture — that axis is answered by six nulls. Both remaining levers add
**data or supervision**:

1. **Multi-dataset pretraining** (WESAD / K-EmoCon / SWELL — §12.2 B3, objectives
   doc O7). The binding constraint is 290 training recordings; this is the only
   planned intervention that changes that number.
2. **Continuous 0–10 supervision** instead of a binarised label (§12.1 A4) —
   more signal from labels already collected.

Also outstanding and Q1-critical per the objectives doc: **E0**, re-implementing
the three competitor papers on these GroupKFold splits. Given §6.1 they very
likely sit inside the same confound, which would turn the finding from "our model
underperforms" into "published numbers on this benchmark need revisiting".
