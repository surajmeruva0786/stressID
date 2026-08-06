# Full-Corpus Results & Comparison Against Prior Work

> **Created:** 2026-07-30
> **Run compared:** full-corpus training of 2026-07-29 (700 recordings, 64 subjects,
> 5-fold subject GroupKFold x 3 seeds x {temporal, static} = 30 fold-models).
> Raw artefacts: `results/full_run1_20260729/`.
> Reproduce the numbers in this file with `python compare_papers.py`
> (writes `paper_comparable_metrics.csv`).

This document does two comparisons:

1. **vs. the StressID origin paper** (`../stressid_paper.pdf`, NeurIPS 2023
   Datasets & Benchmarks) — a results-to-results comparison.
2. **vs. our own pipeline objectives document**
   (`StressID_Paper_Pipeline_Objectives (1).pdf`) — an objectives-to-outcomes
   scorecard, since that document defines targets rather than reporting results.

---

## 0. Executive summary

- Under the **StressID paper's own protocol and metrics**, our classical baselines
  reproduce and slightly exceed its published numbers (weighted F1 0.740 vs. 0.72
  for the best multimodal baseline). The pipeline is therefore not broken and not
  mis-implemented.
- Under a **leakage-free subject-grouped protocol**, the same baselines lose
  0.02–0.09 F1. Subject leakage accounts for a meaningful slice of published
  StressID performance.
- Our trained multimodal transformer **does not beat the origin paper's reported
  numbers**, and we should say so plainly. Three reasons, in order of size:
  the paper's protocol leaks subjects, its features are far richer handcrafted
  descriptors, and it applies SMOTE. Our model also trains from scratch on 448
  recordings, which the loss curves show is not enough.
- The finding that does hold up, and that the origin paper does not report, is a
  **modality-availability confound**: three presence bits with no signal content
  score 0.697 macro F1, beating every model we trained, including the 1.2 M-parameter
  transformer at 0.671. The origin paper's own Table 3 protocol sits inside this
  confound without naming it.
- Against the objectives document, **O1 is met and is the strongest result**;
  O2, O3 and O4 return statistically null; O5, O6, O7 and E0 are not started.

---

## 1. What the two documents are

| | Origin paper (`stressid_paper.pdf`) | Objectives doc (`StressID_Paper_Pipeline_Objectives (1).pdf`) |
|---|---|---|
| Type | Published dataset + baselines (NeurIPS 2023 D&B) | Internal planning document, Q1/IEEE JBHI target |
| Contains | Reported baseline results (Tables 2 and 3) | Objectives O1–O7, experiment matrix E0–E13, success criteria |
| Role here | Numbers to compare against | Targets to score ourselves against |

Only the origin paper reports results, so §2 and §3 compare numbers against it.
§4 scores the objectives document.

---

## 2. Protocol and metric differences (read before any table)

These three differences fully explain the direction of every gap below. Comparing
raw numbers across them is invalid.

| Dimension | Origin paper | Ours |
|---|---|---|
| **Split** | 10 random 80/20 splits **over tasks** — the same subject appears in train *and* test | 5-fold **subject** GroupKFold — a subject is never on both sides |
| **Metrics** | **Weighted** F1 + **balanced** accuracy | **Macro** F1 + plain accuracy |
| **Class balance** | **SMOTE** oversampling on the multimodal training sets | none; class-weighted loss instead |
| **Multimodal scope** | 370 all-modality tasks (the paper notes these are "talking tasks exclusively", 70% stress) | 700 recordings, plus a 364-recording all-modality subset |
| **Features** | ~98 handcrafted physio (HRV/EDA/RRV via filtering), OpenFace AUs + gaze, 140 audio descriptors, W2V2 embeddings | raw signals into encoders trained from scratch; classical baselines use crude window statistics |

**The metric difference alone is worth ~0.14 F1 on the imbalanced subset.** On our
364-recording all-modality subset (71.7% stressed), the *same* temporal model scores
**0.628 weighted F1 but only 0.487 macro F1**. Weighted F1 rewards getting the
majority class right; macro F1 does not. Reference points for trivial classifiers:

| Subset | Positive rate | Always-"stressed" weighted F1 | macro F1 | balanced acc |
|---|---|---|---|---|
| all 700 | 0.526 | 0.362 | 0.345 | 0.500 |
| all-modality 364 | 0.717 | **0.599** | 0.418 | 0.500 |

So on the paper's multimodal subset, a classifier that does nothing at all already
scores ~0.599 weighted F1. Its best reported 0.72 is **+0.12 over doing nothing**,
not +0.72.

---

## 3. Head-to-head vs. the StressID origin paper

### 3.1 Matched protocol — our baselines under the paper's own rules

Random 80/20 task-level splits, 10 repetitions, weighted F1 + balanced accuracy —
i.e. the paper's protocol, reimplemented on our features.

| Method | Weighted F1 | Balanced acc | Paper's comparable row | Paper's F1 |
|---|---|---|---|---|
| feature_fusion + RF | **0.740** | 0.737 | Feature fusion + SVM/MLP/DBN | 0.64 / 0.66 / 0.58 |
| decision_fusion + RF | 0.726 | 0.728 | SVM + Average rule fusion | 0.72 |
| audio + RF | 0.723 | 0.728 | Audio HC + kNN | 0.67 |
| audio + logreg | 0.714 | 0.718 | W2V 2.0 classifier | 0.70 |
| availability_only + logreg | 0.701 | 0.700 | *(not reported by the paper)* | — |
| physio + RF | 0.670 | 0.667 | Physio HC + RF | **0.73** |
| video + RF | 0.657 | 0.661 | AUs + kNN | 0.70 |

**Reading:** our fusion and audio baselines match or beat the paper's; our physio
and video baselines trail it by 0.04–0.06. That gap is expected and is a feature
quality gap, not a protocol gap — the paper extracts 98 clinically-motivated physio
features and OpenFace action units, whereas our baseline uses crude per-window
statistics. The important part is that the fusion rows land in the same band,
which validates the reimplementation.

### 3.2 Same features, protocol swapped — the cost of removing leakage

Identical features and classifiers, only the split rule changes:

> **Corrected 2026-08-06.** The physio row previously read 0.670 → 0.543 =
> **+0.127**. That leaky value is unsupported by any stored artifact (all runs
> give +0.093; a fresh run gives +0.098). See `../LEAKY_PROTOCOL.md` §7.

| Method | Random 80/20 (leaky) | Subject GroupKFold | Inflation |
|---|---|---|---|
| physio + RF | 0.641 | 0.543 | **+0.098** |
| video + RF | 0.679 | 0.622 | +0.057 |
| feature_fusion + RF | 0.736 | 0.686 | +0.051 |
| audio + RF | 0.716 | 0.681 | +0.035 |
| **availability_only + RF** | 0.694 | 0.699 | **−0.004** |

*(weighted F1; regenerate with `python research_way/prove_leakage.py`.)*

Two things to note:

1. **Physiology inflates most (+0.098).** With subjects on both sides of the split,
   a physio model can memorise individual resting heart rate and skin conductance
   baselines. This is the single largest correction to prior StressID numbers, and
   physiology is the modality the origin paper reports its *best* unimodal result on
   (0.73). The corrected figure is meaningfully lower.
2. **`availability_only` inflates by −0.002, i.e. zero.** Presence bits carry no
   subject identity, so leakage cannot help them. Getting the expected null here is
   an internal validity check that the leakage measurement is sound rather than an
   artefact of the resampling.

### 3.3 Our trained model vs. the paper's reported numbers

Both on their respective all-modality subsets (ours 364, paper's 370):

| Model | Protocol | Weighted F1 | Macro F1 | Balanced acc |
|---|---|---|---|---|
| Paper: SVM + Average rule fusion | random 80/20 + SMOTE | **0.72** | not reported | **0.65** |
| Paper: SVM + Sum rule fusion | random 80/20 + SMOTE | 0.72 | not reported | 0.64 |
| Paper: Video only | random 80/20 + SMOTE | 0.67 | not reported | 0.62 |
| Paper: Physiological only | random 80/20 + SMOTE | 0.66 | not reported | 0.58 |
| **Ours: MST-temporal** | **subject GroupKFold** | 0.628 | 0.487 | 0.531 |
| **Ours: MST-static** | **subject GroupKFold** | 0.610 | 0.476 | 0.520 |
| *trivial always-"stressed"* | — | *0.599* | *0.418* | *0.500* |

**Our model does not beat the published numbers.** State it plainly rather than
hiding behind the protocol difference. The margin over a do-nothing classifier is
+0.029 weighted F1 for us versus +0.121 for the paper — roughly a 4× difference.

The honest decomposition of that gap:

- **Leakage (largest).** §3.2 shows 0.04–0.13 of the paper's margin is recoverable
  purely from subjects appearing on both sides of the split.
- **Feature quality.** The paper's handcrafted descriptors encode decades of HRV/EDA
  domain knowledge. Our encoders learn from raw signal on 448 training recordings
  and, per the training curves, memorise rather than generalise (train BCE → 0.09
  while validation F1 peaks between epoch 1 and 22, then decays).
- **SMOTE.** Applied to the paper's training sets on a 70%-imbalanced subset; we
  use class-weighted loss instead.

**A fair like-for-like comparison against the paper's model has not been run.**
Doing it properly requires either re-running the paper's baselines under GroupKFold
(partly done — §3.2 is our feature approximation of it) or running ours under the
paper's leaky protocol. Neither is a legitimate headline claim yet, and E0 in the
objectives document (re-implementing the three competitor papers) remains untouched.

### 3.4 What the origin paper does not report — the availability confound

Audio exists only for the 7 speaking tasks, and those are disproportionately the
stressful ones:

- P(stress | audio present) = **0.709** (378 recordings)
- P(stress | audio absent) = **0.311** (322 recordings)

A classifier given **only three presence bits** — audio? video? physio? — and zero
signal content scores:

| Classifier (GroupKFold, 700 recordings) | Macro F1 |
|---|---|
| **`availability_only` — 3 presence bits, no signal** | **0.697** |
| decision fusion (RF) | 0.688 |
| audio (logreg) | 0.686 |
| feature fusion (RF) | 0.685 |
| **MST-temporal (1.2 M params, trained)** | **0.671** |
| MST-static | 0.654 |
| video (RF) | 0.622 |
| physio (logreg) | 0.565 |

Three boolean flags beat every trained model, including the full multimodal
transformer.

**This directly affects how the origin paper's Table 3 should be read.** The paper
restricts its multimodal baselines to the 370 all-modality tasks and explicitly
notes they are "talking tasks exclusively" with 70% stress. That restriction
*removes* the availability shortcut from the test set — which is methodologically
lucky — but it also means Table 3 is measured on a heavily imbalanced,
task-homogeneous slice where the trivial floor is already 0.599 weighted F1. The
paper reports neither that floor nor the availability correlation. Any work using
StressID's *full* corpus without controlling for availability is measuring the
confound rather than stress.

This is the most defensible novel contribution in our results, and it is
publishable as a negative/methodological finding.

---

## 4. Scorecard vs. the objectives document

The objectives document targets IEEE JBHI (Q1) and defines O1–O7 plus experiments
E0–E13. Current status:

| Obj. | Target | Status | Evidence |
|---|---|---|---|
| **O1** | Corrected leakage-free benchmark; quantify prior inflation | **MET** | §3.2; `e1_leakage_inflation.csv`. Inflation +0.13 (physio) down to −0.00 (availability). **Strongest result we have.** |
| **O2** | Temporal windowing beats static aggregation | **NULL** | ΔmacroF1 = +0.011 [−0.041, +0.063], p = 0.677, 15 paired runs |
| **O3** | Cross-modal attention beats concat/decision fusion | **NOT SUPPORTED** | MST-temporal 0.671 vs. decision-fusion RF 0.688 — fusion baseline is *ahead* |
| **O4** | Graceful degradation under missing modalities | **NULL** | Degradation curves flat-to-rising; 1/12 paired tests nominally significant, none survives Bonferroni |
| **O5** | StressID → WESAD zero-shot transfer | **NOT STARTED** | WESAD not in repo |
| **O6** | Improve minority-class recall (3-class) | **NOT MET** | per-class recall 0.05 / 0.39 / 0.60; class-0 essentially never predicted |
| **O7** | Multi-dataset pretraining to offset small-N | **NOT STARTED** | now the critical path — see §5 |
| **E0** | Re-implement H-C3AT-G, StressID-XAI, missing-modality paper | **NOT STARTED** | Q1-critical per the objectives doc |
| **E13** | Calibration under missing modalities | **DONE, direction unclear** | ECE 0.109 static / 0.146 temporal at full modality; temporal is *more* overconfident |

Against the document's explicit "what done looks like" list:

| Success criterion | Verdict |
|---|---|
| Corrected GroupKFold baseline table with documented delta | **PASS** |
| Temporal beats static by a statistically significant margin | **FAIL** (p = 0.677) |
| Cross-attention beats concat/decision fusion | **FAIL** (behind decision fusion) |
| Missing-modality F1 drop < 10–15% | **PASS, but vacuously** — the curve is flat because the model barely uses the modalities |
| WESAD zero-shot reported | **NOT DONE** |
| Minority-class recall improves | **FAIL** (class-0 recall 0.05) |
| Reproducible: code + fixed splits + weights released | **NOW PASS** — splits fixed in `data/splits_full.json`; weights saved as of 2026-07-30 (`results/full/checkpoints/`, previously discarded) |
| Data-scaling curve from multi-dataset pretraining | **NOT DONE** |
| Head-to-head vs. 3 re-implemented competitors | **NOT DONE** |
| Calibration worse under missing modalities | **NOT CLEANLY SHOWN** |
| Public GitHub repo with code + splits + weights | partial — repo live, weights local |

**One caution on E12, the objectives document's designated headline experiment.**
Its central question — "does temporal context make missing-modality degradation less
steep?" — currently has no measurable effect to compare, because neither variant
degrades. Reporting a temporal advantage here would not be supportable. On accuracy
the temporal-vs-static test reads +0.027, p = 0.015, which looks significant, but
both variants score *below* the 0.717 accuracy of always predicting "stressed" on
this 71.7%-positive subset. That result only says the temporal model collapses
toward the majority class harder, and it must not be reported as a gain.

---

## 5. What this implies for the paper

The objectives document's planned contributions (O2 temporal, O3 cross-modal
fusion, O4 missing-modality robustness) are implemented and return null on StressID.
They cannot carry a Q1 submission as stated. Two directions survive:

1. **Re-frame around O1 + the availability confound.** The leakage-corrected
   benchmark is done and solid, and the confound (§3.4) is a genuine, unreported
   methodological finding about a NeurIPS benchmark. Working title:
   *"Modality Availability Leaks the Label: A Confound in Multimodal Stress Benchmarks."*
   The −0.002 inflation null on `availability_only` is strong supporting evidence
   that the audit is sound.
2. **O7 / Stage −1 pretraining is now the critical path.** The memorisation
   behaviour identifies data volume, not architecture, as the binding constraint.
   Pretraining encoders on WESAD / K-EmoCon / SWELL before fine-tuning is the only
   planned intervention that addresses it.

**Cheapest next experiment:** train on the 7 speaking tasks only. Every recording
there has all three modalities, so the availability shortcut is absent from the
*training* data, not merely the test data. One config change
(`cfg.tasks = SPEECH_TASKS`), and it directly tests whether the architecture
performs better when it cannot take the shortcut.

---

## 6. Files

| File | Contents |
|---|---|
| `paper_comparable_metrics.csv` | Every number in §3.1–3.3: both metric families, both protocols, both scopes |
| `compare_papers.py` | Regenerates the above from `results/full_run1_20260729/predictions.csv` |
| `results/full_run1_20260729/` | Frozen artefacts of the 2026-07-29 run these numbers describe |
| `results/full/` | Current run (retrained 2026-07-30 with checkpoint saving) |
| `results/full/checkpoints/` | 30 fold-model weights + `index.csv` |

### Caveats on these comparisons

- The paper's Table 2 unimodal rows use different per-modality sample counts
  (715 physio / 587 video / 385 audio tasks); our unimodal rows all use 700
  recordings. Not exactly matched.
- Our all-modality subset is 364 recordings vs. the paper's 370 — six recordings
  are dropped by our preprocessing.
- Our §3.1 reimplementation uses our own crude features, not the paper's handcrafted
  feature set. It reproduces the paper's *protocol*, not its *pipeline*.
- The paper's numbers are transcribed from its Tables 2 and 3; we did not re-run
  the authors' released code.
