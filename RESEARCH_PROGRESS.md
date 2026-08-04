# StressID — Research Progress & Future Work Plan

> **Last updated:** 2026-08-04  
> **Status:** Full-corpus run complete and benchmarked against prior work —
> see `research_way/` and §9–§11 below. Improvement plan in §12, project
> history in §13. **Latest:** §12 A1 (domain physio features) + A2 (capacity
> reduction) were run and **failed** — no gain, temporal −0.041 (§14).
> The planned architecture is implemented and trained on all 700 recordings.
> Headline finding is negative-but-publishable: a modality-availability confound
> in StressID dominates every multimodal number, and the temporal/fusion
> contributions are not supported once it is controlled for.
> Model weights are now persisted per fold and the corpus has been retrained to
> produce them — 30 checkpoints in `research_way/results/full/checkpoints/` (§11).
> The retrain reproduces every headline number (§11.5).

---

## 0. Dataset location & re-download audit (2026-08-04)

**Canonical dataset root is now `StressID Dataset new/`** (`DATASET_ROOT` in
`research_way/src/config.py`). The dataset was re-downloaded to rule out a
corrupted copy as the cause of the low scores.

**Audit result: the re-download is byte-identical to the previous copy.**

| check | old | new | verdict |
|---|---|---|---|
| Physiological `.txt` (MD5, all files) | 777 | 777 | 0 differing hashes |
| Audio `.wav` (name + size) | 378 | 378 | identical |
| Videos `.mp4` (name + size) | 629 | 629 | identical |
| `labels.csv`, `demographics.csv`, `self_assessments.csv`, `labels_supplementary.csv` (MD5) | — | — | identical |

Re-verified on the new copy: 700 labelled recordings, 64 subjects,
`P(stress | audio present) = 0.709` (n=378) vs `P(stress | audio absent) = 0.311`
(n=322); 364 all-modality recordings at 0.717 stress. **The modality-availability
confound (§10.5) is structural in the recording design — audio was only captured
for the 7 speech tasks — and is unaffected by re-downloading.**

Two things must not be conflated:

- **Subject leakage** is a property of the *evaluation protocol*, not of the files.
  It *inflates* scores (§10.3); removing it is why our numbers are below the
  paper's. No download can fix or cause it.
- **The availability confound** is in the data, but it is by design and present in
  every copy of StressID.

Cached features in `research_way/data/` remain valid — no regeneration needed.

---

## 1. Project Overview

**StressID** is a multimodal dataset for stress identification published at NeurIPS 2023 (Datasets & Benchmarks).  
Paper: [openreview.net/pdf?id=qWsQi9DGJb](https://openreview.net/pdf?id=qWsQi9DGJb)  
Webpage: [project.inria.fr/stressid](https://project.inria.fr/stressid/)

**Dataset facts:**
- ~65 subjects, 11 tasks (Counting ×3, Stroop, Math, Speaking, Reading, Breathing, Video ×2, Relax)
- 3 modalities: Physiological (ECG, EDA, Respiration @ 500 Hz), Video (face), Audio (speech)
- 370 subject-task samples with all 3 modalities
- Self-report labels: stress, relax, valence, arousal (0–10 scale)
- Binary stress label: 255 stressed / 115 not-stressed (imbalanced)
- 3-class affect: 181 class-2 / 121 class-1 / 68 class-0

---

## 2. Current Pipeline

```
Raw Signals
  ├── Physiological (ECG, EDA, Resp @ 500Hz)
  │     └── neurokit2 clean → HRV time/freq/nonlinear + EDA stat/time + Resp → 50+ features
  ├── Video (face recording)
  │     └── OpenFace → Action Units + Gaze → mean/std stats → ~100 features
  └── Audio (speech)
        ├── librosa → MFCC + spectral features (handcrafted, ~140 features)
        └── wav2vec (frozen) → 512-dim embedding

Feature Matrix
  └── Merge by subject_task index → 356 combined features

Classification
  ├── Unimodal:       RF / KNN / SVC / MLP on each modality separately
  ├── Feature fusion: concat all → PCA → RF/SVC/MLP/DBN
  └── Decision fusion: SVM per modality → sum/product/avg/max rule

Labels: binary (0/1) and 3-class (calm/neutral/stressed)
```

### Baseline Results

| Task | Method | F1 | Accuracy |
|---|---|---|---|
| Binary stress | Decision fusion (avg rule) | 0.681 | 0.620 |
| Binary stress | Video SVC | 0.677 | 0.622 |
| Binary stress | Audio RF | 0.678 | 0.622 |
| Binary stress | Physio RF | 0.662 | 0.572 |
| 3-class affect | Decision fusion (avg rule) | 0.627 | 0.585 |
| 3-class affect | Video SVC | 0.583 | 0.563 |
| 3-class affect | Audio SVC | 0.562 | 0.544 |

---

## 3. Critical Weaknesses

| # | Weakness | Impact |
|---|---|---|
| 1 | No temporal modeling — features aggregated per entire task | Loses stress buildup/decay dynamics |
| 2 | Frozen wav2vec — not fine-tuned on stress | Sub-optimal audio representations |
| 3 | Simple fusion — concat or voting, no cross-modal attention | Modalities don't inform each other |
| 4 | Subject leakage in multimodal notebooks — random splits not GroupKFold | Results likely inflated |
| 5 | Labels subjective (self-report only, no physiological ground truth) | Noisy supervision |
| 6 | No cross-dataset generalization test (WESAD/DEAP/MAHNOB) | Can't claim robustness |
| 7 | Missing modality not handled — whole pipeline breaks | Not robust to real-world use |
| 8 | Small dataset (370 samples) — overfitting risk for DL | Limits deep model capacity |
| 9 | 3-class collapse — model predicts class-2 most of the time | Poor minority class recall |

---

## 4. Future Work — Prioritized

### Tier 1 — Highest impact, implement first

#### 4.1 Temporal Stress Modeling
- Sliding window within each task → sequence of feature vectors
- LSTM / Transformer over time → capture stress buildup and decay
- Currently: 1 vector per 3–5 min task
- Target: 30-second windows → sequence of ~8–10 steps per task
- **Expected gain:** Significant — temporal dynamics carry most stress signal

#### 4.2 End-to-end Fine-tuned Audio Encoder
- Replace frozen wav2vec with fine-tuned wav2vec2 / HuBERT on stress labels
- Multi-task: predict stress label + speaker identity contrastive loss (remove speaker-specific bias)
- Ablation: frozen vs fine-tuned → easy publication delta

#### 4.3 Cross-modal Attention Fusion
- Physiological / video / audio tokens → Transformer with cross-modal attention
- Each modality attends to all others (not just concatenation)
- Missing modality → mask token (handles absence gracefully)
- **Main architectural contribution**

---

### Tier 2 — Publishable extensions

#### 4.4 Evaluation Methodology Fix
- Re-run all baselines with strict GroupKFold (subjects never shared across train/test)
- Show corrected (likely lower) numbers → motivates Tier 1 work
- Cross-dataset: train on StressID, test on WESAD (ECG/EDA only) → domain adaptation

#### 4.5 Label Refinement via Physiology
- Cluster physiological responses per task
- High-confidence samples: agreement between physio cluster + self-report label
- Train on confident subset, evaluate on all
- Semi-supervised learning on noisy labels

#### 4.6 Subject-Adaptive Personalization (Few-shot)
- MAML or Prototypical Networks
- N-shot adaptation to new subject using 2–3 labeled calibration samples
- Practical for wearable deployment scenarios

#### 4.7 Self-supervised Pre-training on Unlabeled Physiological Data
- SimCLR-style contrastive pre-training on raw physiological windows
- Fine-tune on small labeled set → addresses the 370-sample limitation

---

### Tier 3 — Longer-term

#### 4.8 Continuous Stress Estimation (Regression)
- Labels currently discretized from 0–10 continuous self-report
- Predict raw stress score directly → richer supervision
- Ordinal regression head on top of Tier 1 architecture

#### 4.9 Explainability Study
- SHAP values per modality, per feature
- Which AUs / HRV features drive predictions?
- Clinically actionable stress biomarkers

#### 4.10 Real-time Inference Pipeline
- Streaming ECG + webcam + mic → prediction in <1 second
- ONNX export of trained model + streaming buffer
- Enables wearable/edge deployment

---

## 5. Target Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│               MULTIMODAL STRESS TRANSFORMER (MST)                   │
└─────────────────────────────────────────────────────────────────────┘

INPUT (per 30-second window)
  Physio : [ECG_t, EDA_t, RSP_t] raw sequence @ 500Hz
  Audio  : raw waveform @ 16kHz
  Video  : AU vectors sequence @ 25fps

MODALITY ENCODERS
  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
  │  1D-CNN        │  │  wav2vec2      │  │  AU Transformer│
  │  + BiLSTM      │  │  (fine-tuned)  │  │  (per frame)   │
  │  → Physio Toks │  │  → Audio Toks  │  │  → Video Toks  │
  └────────┬───────┘  └───────┬────────┘  └───────┬────────┘
           │                  │                    │
           ▼                  ▼                    ▼
      [P1 .. Pn]          [A1 .. Am]          [V1 .. Vk]

CROSS-MODAL FUSION  ← KEY CONTRIBUTION
  ┌────────────────────────────────────────────────────┐
  │  Multi-head Cross-Attention                        │
  │  PhysioQ → AudioKV, VideoKV                        │
  │  AudioQ  → PhysioKV, VideoKV                       │
  │  VideoQ  → PhysioKV, AudioKV                       │
  │  + CLS token aggregation                           │
  └───────────────────────┬────────────────────────────┘
                          │
               Fused Representation [512-d]

TASK CONDITIONING
  Task type embedding → concat with fused repr
  → LayerNorm → MLP Head

OUTPUT
  ├── Binary stress     (sigmoid)
  ├── 3-class affect    (softmax)
  └── Stress score 0–10 (regression, optional)

TRAINING STRATEGY
  Loss = BCE_binary + CE_3class + λ·MSE_regression
  + Contrastive subject loss (disentangle subject identity from CLS)
  Optimizer: AdamW, lr=1e-4, warmup 10%
  Subject-grouped cross-validation (strict, no leakage)
  Modality dropout during training: 25% chance each modality masked
```

---

## 6. Immediate Next Steps

| Step | Task | Status |
|---|---|---|
| 1 | Fix evaluation — re-run baselines with strict GroupKFold | **DONE** (5-fold, 700 recs) |
| 2 | Temporal baseline — sliding window + LSTM on physiological only | **DONE** — no effect |
| 3 | Fine-tune wav2vec2 on audio modality | TODO (`transformers` not installed) |
| 4 | Implement cross-modal attention fusion (MST architecture) | **DONE** — no measurable gain |
| 5 | Benchmark MST on WESAD for generalization | TODO (dataset not in repo) |
| 6 | Label refinement via physiological clustering | TODO |
| 7 | **Stage −1 multi-dataset pretraining** — now the critical path | TODO |
| 8 | Speech-tasks-only corpus (removes the confound from training) | TODO — one config change |
| 9 | Persist trained fold weights (was discarded after each run) | **DONE** (2026-07-30, §11) |
| 10 | Head-to-head comparison vs. origin paper + objectives doc | **DONE** — `research_way/RESULTS_AND_COMPARISON.md` |
| 11 | E0 — re-implement the 3 competitor papers on our splits | TODO — Q1-critical per objectives doc |

---

## 7. Suggested Paper Titles

- *"Temporal Multimodal Stress Detection via Cross-Modal Transformer with Subject-Invariant Representations"*
- *"From Static to Temporal: Rethinking Multimodal Stress Classification on StressID"*
- *"Missing-Robust Multimodal Stress Estimation with Cross-Modal Masked Transformers"*

---

## 8. Change Log

| Date | Change |
|---|---|
| 2026-06-04 | Initial research plan created — baselines analyzed, architecture designed, future work prioritized |
| 2026-06-04 | Added `.claude/settings.local.json` with PostToolUse hook for auto-commit and push on every Write/Edit |
| 2026-07-28 | `research_way/` implementation built and validated on a 16-subject / 6-task subset |
| 2026-07-29 | **Trained on the entire corpus** — 64 subjects, 11 tasks, 700 recordings, 5-fold GroupKFold, 30 fold-models. Results in `research_way/results/full/` |
| 2026-07-30 | Benchmarked against the StressID origin paper and the pipeline objectives doc → `research_way/RESULTS_AND_COMPARISON.md` (§10) |
| 2026-07-30 | Added checkpoint persistence (`src/checkpoint.py`) and re-ran the full corpus to produce saved weights (§11) |
| 2026-07-30 | Re-run finished (11,453 s): 30 checkpoints / 135 MB in `results/full/checkpoints/`. Headline metrics reproduce; static is bit-identical, temporal drifts ±0.002. All §9/§10 conclusions unchanged (§11.5) |
| 2026-07-30 | Untracked `research_way/src/__pycache__` — gitignored but still tracked, so the auto-commit hook committed `.pyc` churn on every run |

---

## 9. Full-Corpus Results (2026-07-29)

Protocol: 5-fold subject GroupKFold, 3 seeds × 5 folds × {temporal, static} = 30
fold-models, best-val-F1 checkpoint per fold. 700 recordings / 64 subjects / 11 tasks.

### 9.1 The finding: modality availability leaks the label

Audio exists only for the 7 speech tasks, which are mostly the stressful ones:
P(stress | audio) = 0.71 vs P(stress | no audio) = 0.31.

| classifier (leakage-free, 700 recordings) | macro F1 |
|---|---|
| `availability_only` — **3 presence bits, no signal content** | **0.697** |
| feature fusion (RF) | 0.685 |
| audio (logreg) | 0.686 |
| physio (logreg) | 0.565 |
| trained multimodal transformer (temporal) | 0.671 |

The 1.2 M-parameter multimodal model **loses to three presence bits**. Any
missing-modality claim on StressID is uninterpretable without controlling for this.

### 9.2 Model performance with the shortcut removed

Scored only on the 364 recordings where all three modalities naturally exist
(availability constant, so it cannot be read):

| variant | macro F1 [95% CI] | majority reference | margin |
|---|---|---|---|
| static | 0.476 [0.452, 0.504] | 0.418 | +0.058 |
| temporal | 0.487 [0.451, 0.525] | 0.418 | +0.069 |

Small but reliably above the majority reference — an improvement on the 16-subject
subset, where the margin was −0.007 / +0.028 (at chance).

### 9.3 Null results (report as null)

- **Temporal vs static:** +0.011 [−0.041, +0.063], p = 0.677 on macro F1. Weakness #1
  in §3 is not addressed by temporal modelling at this data scale. (The same test on
  *accuracy* reads +0.027, p = 0.014 — but both variants score below the 0.717 accuracy
  of a predict-always-stressed classifier on this 0.717-positive subset, so that
  "significant" result only says temporal collapses toward the majority class harder.
  Do not report it.)
- **Missing-modality cost:** of 12 paired condition-vs-full tests, one is nominally
  significant and none survives Bonferroni. Degradation curves are flat-to-rising.
- **3-class collapse (weakness #9) persists:** per-class recall 0.05 / 0.39 / 0.60,
  macro F1 0.318 — essentially unchanged by 7× more data than the subset run.
- **Overfitting:** train BCE → 0.09 while val F1 peaks between epoch 1 and 22 and
  decays. 448 training recordings cannot train three encoders + fusion from scratch.

### 9.4 What this changes about the plan

§4.1 (temporal) and §4.3 (cross-modal fusion) are implemented and do **not** produce
gains on StressID alone — so they cannot carry the paper by themselves. The two viable
directions are now §4.4-as-contribution (the confound + leakage-free protocol is the
result) and §4.7 / Stage −1 self-supervised or multi-dataset pretraining, which the
memorisation behaviour shows is the binding constraint.

Suggested retitle, replacing §7's options:
*"Modality Availability Leaks the Label: A Confound in Multimodal Stress Benchmarks"*

---

## 10. Comparison Against Prior Work (2026-07-30)

Full write-up: **`research_way/RESULTS_AND_COMPARISON.md`**.
Regenerate with `python research_way/compare_papers.py` →
`research_way/paper_comparable_metrics.csv`.

Compared against two documents: the **StressID origin paper**
(`stressid_paper.pdf`, NeurIPS 2023 D&B — reports results) and the
**pipeline objectives doc** (`research_way/StressID_Paper_Pipeline_Objectives (1).pdf`
— defines targets O1–O7 / E0–E13).

### 10.1 Metric and protocol mismatch (must be stated in any comparison)

The origin paper reports **weighted F1 + balanced accuracy** over **10 random
80/20 task-level splits** (subjects on both sides) with **SMOTE**. We report
**macro F1** under **subject GroupKFold** with no SMOTE. The metric choice alone
is worth ~0.14 F1 on the imbalanced subset: the same temporal model scores
**0.628 weighted F1 but 0.487 macro F1** on the 364 all-modality recordings.

Trivial-classifier floors (always predict "stressed"):

| Subset | pos rate | weighted F1 | macro F1 | balanced acc |
|---|---|---|---|---|
| all 700 | 0.526 | 0.362 | 0.345 | 0.500 |
| all-modality 364 | 0.717 | **0.599** | 0.418 | 0.500 |

The paper's best multimodal result (0.72 weighted F1) is **+0.12 over doing
nothing**, not +0.72.

### 10.2 Matched-protocol reproduction

Our classical baselines re-run under the paper's own protocol (random 80/20 ×10,
weighted F1) match or beat its published numbers:

| Ours (paper protocol) | wF1 | Paper's comparable row | Paper wF1 |
|---|---|---|---|
| feature_fusion + RF | **0.740** | Feature fusion + SVM / MLP / DBN | 0.64 / 0.66 / 0.58 |
| decision_fusion + RF | 0.726 | SVM + Average rule fusion | 0.72 |
| audio + RF | 0.723 | Audio HC + kNN | 0.67 |
| physio + RF | 0.670 | Physio HC + RF | **0.73** |
| video + RF | 0.657 | AUs + kNN | 0.70 |

Physio/video trail because the paper uses far richer handcrafted features
(98 HRV/EDA/RRV descriptors, OpenFace AUs) vs. our crude window statistics.
The fusion rows landing in the same band validates the reimplementation.

### 10.3 Cost of removing subject leakage

Same features, only the split rule changes (weighted F1):

| Method | random 80/20 (leaky) | subject GroupKFold | inflation |
|---|---|---|---|
| physio + RF | 0.670 | 0.543 | **+0.127** |
| feature_fusion + RF | 0.740 | 0.687 | +0.053 |
| audio + RF | 0.723 | 0.681 | +0.042 |
| decision_fusion + RF | 0.726 | 0.690 | +0.036 |
| **availability_only + RF** | 0.697 | 0.699 | **−0.002** |

Physiology inflates most — subject-specific baselines get memorised — and it is
the modality the origin paper reports its *best* unimodal number on (0.73).
`availability_only` inflates by ~zero, exactly as it must (presence bits carry no
subject identity); that expected null is an internal validity check on the audit.

### 10.4 Our model vs. the paper's reported numbers — we do not win

| Model | protocol | wF1 | macro F1 | bal acc |
|---|---|---|---|---|
| Paper: SVM + Average rule fusion | random 80/20 + SMOTE | **0.72** | n/r | **0.65** |
| Ours: MST-temporal | subject GroupKFold | 0.628 | 0.487 | 0.531 |
| Ours: MST-static | subject GroupKFold | 0.610 | 0.476 | 0.520 |
| *trivial always-"stressed"* | — | *0.599* | *0.418* | *0.500* |

Margin over the trivial floor: **+0.029 for us vs. +0.121 for the paper** (~4×).
Decomposition, largest first: (1) the paper's protocol leaks subjects (§10.3),
(2) its handcrafted features encode decades of HRV/EDA domain knowledge while our
encoders learn from raw signal on 448 recordings and memorise, (3) SMOTE.
**A true like-for-like comparison has not been run** — E0 (re-implementing the
three competitor papers on our splits) remains untouched and is Q1-critical.

### 10.5 What the origin paper does not report

P(stress | audio present) = **0.709** (378 recs) vs. P(stress | audio absent) =
**0.311** (322 recs). The paper's Table 3 restricts to the 370 all-modality tasks
and notes they are "talking tasks exclusively" with 70% stress — which
incidentally removes the shortcut from its test set, but it reports neither the
0.599 trivial floor that restriction implies nor the availability correlation
itself. Any work using the *full* corpus without controlling for availability is
measuring the confound. This remains our most defensible novel contribution.

### 10.6 Objectives scorecard

| Obj. | Target | Status |
|---|---|---|
| **O1** | Leakage-free corrected benchmark | **MET** — strongest result we have |
| **O2** | Temporal beats static | **NULL** (p = 0.677) |
| **O3** | Cross-modal attention beats concat/decision fusion | **NOT SUPPORTED** (0.671 vs 0.688 — baseline ahead) |
| **O4** | Graceful missing-modality degradation | **NULL** — curves flat-to-rising |
| **O5** | StressID → WESAD transfer | NOT STARTED |
| **O6** | Minority-class recall | **NOT MET** — class-0 recall 0.05 |
| **O7** | Multi-dataset pretraining | NOT STARTED — now the critical path |
| **E0** | Re-implement 3 competitors | NOT STARTED — Q1-critical |
| **E13** | Calibration under missing modalities | DONE, direction unclear (temporal *more* overconfident) |

**Caution on E12**, the objectives doc's designated headline experiment: its
question ("does temporal context make degradation less steep?") has no measurable
effect to compare, because neither variant degrades. The accuracy-based
temporal-vs-static test (+0.027, p = 0.015) must **not** be reported as a gain —
both variants sit below the 0.717 accuracy of always predicting "stressed", so it
only shows temporal collapses toward the majority class harder.

---

## 11. Reproducibility & Saved Weights (2026-07-30)

### 11.1 The gap this closes

Runs before 2026-07-30 kept the best-val-F1 state in memory per fold and wrote
only `predictions.csv` + metrics. Results were fully analysable but **no weights
survived**, so inference, demos and further fine-tuning all required a ~3 h
retrain. The objectives doc lists "code + fixed splits + **trained weights**
released" as a non-negotiable Q1 requirement, so this was also a submission
blocker.

### 11.2 What was added

- **`research_way/src/checkpoint.py`** — `save_checkpoint` / `load_checkpoint` /
  `load_model` / `apply_norm` / `list_checkpoints`.
- **`research_way/src/train.py`** — `train_one_fold(..., save_ckpt=True)` writes
  the best-val-F1 state; `run()` writes `checkpoints/index.csv` ranked by val F1.

Each checkpoint (~4.9 MB, 30 total ≈ 150 MB) is **self-contained**: weights, full
`Config`, `subject_vocab`, `n_tasks`, temporal flag, best epoch / val F1, the
fold's train/val/test subject lists, and the **fold-specific train-only audio and
video normalisation stats**.

> Those norm stats matter: they are computed from that fold's training subjects
> only. Reusing another fold's stats at inference reintroduces exactly the leakage
> the split exists to prevent — so always use the ones shipped in the checkpoint.

### 11.3 Usage

```python
from src.checkpoint import load_model, list_checkpoints

model, meta = load_model("results/full/checkpoints/temporal_seed1_fold4.pt")
# meta: cfg, norm, subject_vocab, fold_subjects, best_val_f1, best_epoch, device
```

Verified before the production run via a 1-epoch round-trip on real data:
checkpoint saves, reloads, and reproduces logits bit-identically
(max |Δ| = 0.0).

### 11.4 Artefact layout

| Path | Contents | Tracked? |
|---|---|---|
| `results/full/` | current run (retrained 2026-07-30, **with** weights) | gitignored |
| `results/full/checkpoints/` | 30 fold models + `index.csv` | gitignored |
| `results/full_run1_20260729/` | frozen artefacts of the run §9 and §10 describe | gitignored |
| `data/splits_full.json` | fixed GroupKFold splits | **tracked** |
| `RESULTS_AND_COMPARISON.md`, `compare_papers.py`, `paper_comparable_metrics.csv` | §10 | **tracked** |

`results/` is gitignored, so weights do not bloat the repo. **For the paper they
must be published separately** (release asset / Zenodo) to satisfy the
reproducibility requirement.

### 11.5 Retrain reproducibility — completed 2026-07-30 22:38

The re-run used the same fixed splits and seeds and took 11,453 s (30 fold-models,
30 checkpoints, 135 MB). Headline metrics reproduce:

| Metric (macro F1 unless noted) | run 1 (07-29) | run 2 (07-30) | Δ |
|---|---|---|---|
| all-700, static | 0.6540 | 0.6540 | **0.0000** |
| all-700, temporal | 0.6710 | 0.6726 | +0.0016 |
| complete-364, static | 0.4759 | 0.4759 | **0.0000** |
| complete-364, temporal | 0.4866 | 0.4850 | −0.0016 |
| E2 temporal−static (F1) | +0.011, p = 0.677 | +0.009, p = 0.721 | still null |
| E2 temporal−static (acc) | +0.027, p = 0.015 | +0.023, p = 0.037 | still below the 0.717 majority floor |

The **static** variant reproduces bit-identically; **temporal** drifts by ±0.002
(cuDNN nondeterminism in the attention/LSTM kernels). **Every conclusion in §9 and
§10 is unchanged**, including the E12 caution: temporal accuracy is again
nominally significant and again sits below the 0.717 always-"stressed" floor
(0.690 temporal / 0.667 static), so it still must not be reported as a gain.
E12b again returns exactly one nominally significant cell of 12
(static / no_audio, p = 0.019), which again fails Bonferroni (0.05/12 = 0.0042).

**The numbers quoted in §9 and §10 refer to the 2026-07-29 run** preserved in
`results/full_run1_20260729/`; `results/full/` now holds run 2 plus the weights.

> **Operational note.** The re-run was launched as
> `python run_full.py --stage train ... ; if ($?) { ... --stage evaluate }`.
> The evaluate stage silently did **not** fire: in PowerShell 5.1, `2>&1` on a
> native exe wraps stderr lines as ErrorRecords and sets `$?` to `$false` even on
> exit code 0 — and torch emits `UserWarning`s to stderr. That briefly left
> `results/full/` holding new `predictions.csv` beside evaluation CSVs from the
> previous day. Fixed by running the evaluate stage separately (4 s). Chain
> pipeline stages on explicit `$LASTEXITCODE`, not `$?`.

### 11.6 Environment

Windows 11 · Python 3.11 · torch 2.6.0+cu124 · CUDA on Quadro P1000 (4 GB) ·
train stage ≈ 10,640 s (~3 h) for 30 fold-models.

---

## 12. Improvement Plan (2026-08-02)

### 12.0 Realistic target

On 364 recordings / 64 subjects with noisy self-report labels and the
availability shortcut removed, **0.55–0.60 macro F1 would be a strong result**
(current: 0.487). Anything near 0.75 on this data implies either subject leakage
or weighted F1 read on an imbalanced subset. Calibrate to that band.

### 12.1 Tier A — cheap, highest payoff per unit effort

| # | Action | Rationale | Status |
|---|---|---|---|
| **A1** | **Replace raw-signal physio encoding with domain features** (neurokit2 HRV / EDA / RSP per *window*, feeding the same temporal model) | ~~The origin paper's handcrafted physio features reach 0.73 where our learned encoder gets 0.54–0.57.~~ **Premise was invalid — see §14.1.** | **DONE — FAILED** (§14) |
| **A2** | **Cut capacity + regularise**: `d_model` 128→64, fusion/temporal layers →1, dropout →0.4, early stopping on val F1 (patience ~8) | 1.2 M parameters on 448 training recordings. Train BCE → 0.09 while val F1 peaks at epoch 1–22 then decays. | **DONE — FAILED** (§14) |
| **A3** | **Speaking-tasks-only training** (`cfg.tasks` = the 7 speech tasks) | Every such recording has all three modalities, so the availability shortcut is absent from *training*, not merely from evaluation. Currently the model spends capacity learning the trapdoor. One config change, ~3 h. | TODO |
| **A4** | **Use the continuous 0–10 stress score as primary supervision** (raise `w_regression`, or ordinal regression + threshold) | Binarising throws away information we already have. Same labels, strictly more signal, and it attacks the class-collapse problem. Head already exists at `w_regression=0.2`. | TODO |

### 12.2 Tier B — real work, real payoff

| # | Action | Rationale |
|---|---|---|
| **B1** | **Subject-adaptive calibration (few-shot)** — normalise/adapt against 2–3 labelled samples per test subject (e.g. their Relax baseline) | Inter-subject variation is the dominant noise source here — precisely why leakage inflated physio by +0.127. Turns the leakage problem into a feature, and is realistic for deployment. Highest-value item the objectives doc under-prioritised. |
| **B2** | **Fine-tune wav2vec2 / HuBERT** for audio | Audio is already the strongest honest modality (0.686 vs physio 0.565) and is the one place an off-the-shelf foundation model exists. `AudioEncoder` has a documented drop-in slot. Needs `pip install transformers`. |
| **B3** | **Stage −1 multi-dataset pretraining (O7)** | Correct diagnosis of the overfitting, but the most expensive item (acquire + harmonise WESAD / K-EmoCon / SWELL). Do only after Tier A says whether the architecture deserves it. |

### 12.3 Explicitly not doing

More temporal-architecture variants or cross-modal-attention tuning. Both were
tested properly and returned null (§9.3). Adding capacity to a model that already
memorises by epoch 1 will not help.

### 12.4 Highest-value item for the paper

**E0 — re-implement the three competitor papers on our GroupKFold splits.** The
objectives doc calls it Q1-critical, and given §10.5 they very likely sit inside
the same availability confound. That converts the finding from "our model
underperforms" into "published numbers on this benchmark need revisiting", which
is a far stronger paper than any accuracy gain Tier A is likely to buy.

---

## 13. Project Journey

Condensed history of how this project reached its current state, including the
wrong turns — they are the reason several results are trustworthy.

### Phase 1 — Baseline analysis (2026-06-04)

Started from the StressID origin paper and the `stressID-main/` reference
notebooks. Catalogued 9 critical weaknesses (§3), of which #4 — random splits
instead of subject-grouped — turned out to matter most. Designed the target
architecture (§5) and prioritised future work into three tiers (§4). Added a
PostToolUse hook that auto-commits and pushes on every Write/Edit.

### Phase 2 — Implementation and subset validation (2026-07-28)

Built `research_way/` from scratch: windowing, three modality encoders,
cross-modal fusion with modality dropout, temporal aggregation, three output
heads, and a Stage-6 evaluation suite (E1–E13). Validated on a 16-subject /
6-task subset. **Result: the model sat at chance** (margin over majority −0.007
static / +0.028 temporal). Attributed to subset size, which motivated Phase 3.

### Phase 3 — Full corpus (2026-07-29)

Scaled to all 64 subjects / 11 tasks / 700 recordings. 5-fold GroupKFold × 3
seeds × 2 variants = 30 fold-models, ~3 h. Two things emerged:

1. Model performance improved over the subset but stayed modest (0.487 macro F1
   against a 0.418 floor).
2. **The availability confound** (§9.1) — the decisive finding. Three presence
   bits with no signal content beat every trained model.

Temporal modelling, cross-modal fusion and missing-modality robustness all
returned statistically null (§9.3).

### Phase 4 — Weights lost, then recovered (2026-07-30)

The terminal running Phase 3 was closed. Investigation confirmed training had
completed cleanly (`[train] done in 10640s`, 30 fold-models logged, all
downstream evaluation artefacts present, everything committed). **However: no
weights had ever been written to disk** — `train_one_fold` kept the best state
in memory and persisted only predictions. Since the objectives doc lists released
weights as a non-negotiable Q1 requirement, this was a submission blocker, not
just an inconvenience.

Added `src/checkpoint.py`, wired it into `train.py`, verified with a 1-epoch
round-trip on real data (reload reproduced logits bit-identically), backed up the
Phase-3 artefacts to `results/full_run1_20260729/`, and retrained. **Static
reproduced bit-identically, temporal drifted ±0.002; every conclusion held**
(§11.5).

*Mistake worth recording:* the retrain was launched as
`... --stage train ; if ($?) { ... --stage evaluate }`. PowerShell 5.1 sets `$?`
to `$false` when a native exe writes to stderr under `2>&1` — and torch emits
`UserWarning`s — so the evaluate stage silently never ran, briefly leaving
`results/full/` with fresh predictions beside stale evaluation CSVs. Chain on
`$LASTEXITCODE`, not `$?`.

### Phase 5 — Benchmarking against prior work (2026-07-30)

Compared against the origin paper and the internal objectives doc (§10,
`research_way/RESULTS_AND_COMPARISON.md`). The comparison could not be done
naively: the paper reports weighted F1 + balanced accuracy over leaky random
splits with SMOTE, we report macro F1 under GroupKFold. **The metric choice alone
is worth ~0.14 F1** — the same model reads 0.628 weighted vs 0.487 macro. So the
results were recomputed under the paper's own metrics *and* protocol
(`compare_papers.py`).

Outcome: our baselines reproduce the paper under its rules (0.740 vs 0.72), the
leakage correction costs 0.04–0.13 with physio worst at +0.127, and **our trained
model does not beat the published numbers** (+0.029 over the trivial floor vs
their +0.121). Recorded plainly rather than framed away.

### Phase 6 — Improvement plan (2026-08-02)

Accepted that the planned architectural contributions (O2/O3/O4) are null and
cannot carry a paper, and re-planned around what the evidence supports: §12.
Began A1 + A2.

### What made the negative results trustworthy

Two internal controls, both of which returned the value they had to:

- `availability_only` inflates by **−0.002** under leaky splits — presence bits
  carry no subject identity, so leakage cannot help them. Confirms the leakage
  measurement is real and not a resampling artefact.
- Our classical baselines **reproduce the origin paper under its own protocol**
  (0.740 vs 0.72). Confirms the pipeline is not simply broken.

Without those, a 0.487 result would be indistinguishable from a bug. With them,
it is a measurement.

### Phase 7 — A1 + A2 attempted and failed (2026-08-02)

See §14. Both interventions were implemented, run on the full corpus, and did
not improve the leakage-free number. Recorded as a null.

### Recurring lesson

Three separate times the flattering number was the wrong one: random splits
(+0.127 on physio), weighted F1 on a 72%-positive subset (+0.14), and the
temporal accuracy result that reads p = 0.015 while sitting *below* a
do-nothing classifier. Each looked like a result and was an artefact. Every
headline number in this project is now reported against an explicit trivial
baseline for that exact subset.

---

## 14. A1 + A2 Result — Negative (2026-08-02)

Run: `results/a1a2/` · config `a1a2_config()` · same corpus, splits and seeds as
`results/full/` (the control) · 3,798 s (63 min) vs. 11,453 s · 30 checkpoints,
37 MB vs. 135 MB.

### 14.1 A1's stated rationale was invalid

A1 was justified by "the origin paper's handcrafted physio features reach 0.73
where our learned encoder gets 0.54–0.57." **That comparison was wrong, and it is
the exact error this project exists to document**: 0.73 is the paper's *leaky*
number, 0.54–0.57 is ours *leakage-free*. §10.3 measures physio inflation at
+0.127, so handcrafted features under GroupKFold should land near **0.60**, not
0.73. The real headroom was ~0.03, not ~0.19.

A cheap GroupKFold check on the extracted features confirmed this before the
training run:

| Physio representation | macro F1 |
|---|---|
| raw waveform + logreg | 0.565 |
| A1 features + logreg | 0.565 (±0.000) |
| raw waveform + RF | 0.540 |
| A1 features + RF | **0.573** (+0.033) |

### 14.2 Result: no improvement; temporal got worse

macro F1, leakage-free scope (364 all-modality recordings) in bold:

| Scope / variant | control (`full`) | A1+A2 (`a1a2`) | Δ |
|---|---|---|---|
| **complete-364, temporal** | **0.4850** | **0.4439** | **−0.041** |
| **complete-364, static** | **0.4759** | **0.4752** | −0.001 |
| all-700, temporal | 0.6726 | 0.6574 | −0.015 |
| all-700, static | 0.6540 | 0.6445 | −0.010 |

Static is unchanged; **temporal lost 0.041 and now scores *below* static**
(0.444 vs 0.475), reversing the control's ordering. Both remain far from the
§12.0 target of 0.55–0.60. Bootstrap CIs overlap
(temporal control [0.451, 0.523] vs a1a2 [0.415, 0.479]), so this is best read
as "no gain, probably a loss for temporal" rather than a precise effect size.

### 14.3 Why it likely failed

- **Capacity was not the binding constraint.** Early stopping fired on **30/30**
  folds, and mean best epoch only moved 9.6 → 6.9. The model peaks early and
  decays *regardless* of size — shrinking it 1.24 M → 312 k params (25%) did not
  change that shape, it just lowered the ceiling.
- **The temporal transformer needed the width.** It lost most from
  `d_model` 128→64 and `temporal_layers` 2→1, which is consistent with temporal
  being the component with the most parameters to spare.
- **Early stopping may have truncated late peaks.** Control best epochs reach 22;
  A1+A2 never exceeds 15. Patience 8 from an early peak can stop before a second
  rise.

### 14.4 Design flaw to avoid repeating

**Five things changed at once** (physio representation, `d_model`,
`fusion_layers`, `temporal_layers`, `dropout`, early stopping), so the −0.041 is
unattributable. A1 and A2 were bundled because they "move in the same direction";
that reasoning was wrong — it bought one run's worth of time and cost the ability
to interpret the result. Ablate one axis at a time, or at minimum keep A1 and A2
in separate runs.

### 14.5 What this adds to the story

A fourth null, and a consistent one: temporal modelling, cross-modal fusion,
missing-modality robustness, and now representation + capacity changes all fail
to move this dataset. Every intervention that redistributes *how* a fixed 448
training recordings are modelled returns nothing. That is increasingly strong
support for the §12.2 B3 / O7 diagnosis — **data volume, not architecture or
representation, is the binding constraint** — and it makes the confound finding
(§10.5) the paper's viable centre of gravity.

### 14.6 Next

Do **not** run more capacity/representation variants. Remaining Tier A items
(A3 speaking-tasks-only, A4 continuous-score supervision) test *different*
axes — the training distribution and the supervision signal — and are still
worth running, one change per run. B1 (subject-adaptive calibration) remains the
highest-value untested idea.

---

## 15. SOTA Campaign (2026-08-03 →)

Standing instruction: iterate — change, retrain, analyse, document, commit,
push, repeat — until results plateau or a target is met. Every run leaves a
report (§15.2).

### 15.1 What "SOTA" means here, stated up front

The published StressID best is **0.72 weighted F1 / 0.65 balanced accuracy**,
measured under **random 80/20 task splits with subjects on both sides** and with
SMOTE. That number is reachable by adopting that protocol — and doing so would
reproduce the exact error §10 exists to document. **This campaign therefore
targets the best number under subject GroupKFold on the 364 all-modality
recordings, and reports it against the published figure honestly rather than
switching protocols to close the gap.**

Primary metric: `complete364_macro_f1`. Majority-class reference **0.418**.

### 15.2 Reporting infrastructure (built 2026-08-03)

`src/report.py` writes, for **every** run, into **`research_way/reports/<run>/`**:
`report.md`, `report.pdf`, `metrics.json` — plus a cross-run registry at
`reports/RUNS.md` and `reports/runs_index.csv` ranking every run by the primary
metric with its delta vs. the previous best.

Reports live outside `results/` **because `results/` is gitignored**. They are
tracked and committed with the code that produced them, so progress survives.

### 15.3 The finding that redirected the campaign

A diagnostic that should have been run much earlier: **what do simple classical
models score on the leakage-free subset?** All prior baseline numbers (§10) were
measured on all 700 recordings, where the availability shortcut inflates them.

On complete-364 under GroupKFold, **seven classical configurations beat the
1.2 M-parameter transformer**:

| model | macro F1 |
|---|---|
| video + SVC | **0.544** |
| video + logreg | 0.538 |
| all-features + HGB | 0.532 |
| video + RF | 0.510 |
| physfeat+audio+video + HGB | 0.504 |
| **deep MST-temporal** | **0.485** |
| **deep MST-static** | 0.476 |

The deep architecture was never the ceiling — it is worse than an SVC on video
features. Deep representation learning on 448 training recordings is the wrong
tool, and the campaign moved to `src/classical.py`.

> Those numbers were picked by looking at outer-fold scores, i.e. test-set
> fishing. Under proper nested CV the honest figure is 0.533 (§15.4) — the 0.544
> was ~0.011 optimistic. Recorded because the gap is the point.

### 15.4 Results so far

| run | what changed | ensemble macro F1 | single-model macro F1 |
|---|---|---|---|
| *(control)* deep MST-temporal | — | — | 0.4850 |
| `c1_nested_ensemble` | classical zoo, nested GroupKFold selection, soft-vote top-3 | **0.5327** | 0.4870 |
| `c2_subject_relative` | + features relative to each subject's own Relax/Breathing baseline | 0.5199 | **0.5371** |

**Best honest number so far: 0.5371** (c2, single inner-selected model),
**+0.052 over the deep model** and **+0.119 over the majority reference**.

Two things to be careful about:

- **c2 is a *personalised* setting.** Features are expressed relative to that
  subject's own Relax/Breathing baseline. Those tasks carry no audio, so they sit
  outside the 364-recording evaluation subset — no evaluated sample is touched
  and no label is read. It is legitimate calibration matching a real deployment,
  but it answers a different question than c1 and is reported separately.
- **Ensemble-vs-single must not be chosen post hoc.** c1 favours the ensemble,
  c2 the single model. Picking the better one *after* seeing outer scores is the
  same fishing that inflated 0.544 → the aggregation strategy must itself be
  selected by inner CV. **That is the next iteration's fix, not a result.**

### 15.5 Next iterations

1. **C3 — select aggregation by inner CV** (single vs. top-k ensemble chosen on
   inner folds), removing the post-hoc choice above. Correctness fix first.
2. **C4 — better video features.** Video is the strongest single modality yet
   uses crude 32×32 pixel statistics. Proper AUs (OpenFace) or a face-embedding
   model is the largest untapped feature-side gain.
3. **C5 — A4 continuous-score supervision** as an auxiliary target.
4. **C6 — calibrated probability stacking** across modalities.

### 15.6 C3 result + a campaign-level bias warning (2026-08-03)

| run | selection of ensemble size k | macro F1 | single-model (reference) |
|---|---|---|---|
| `c1_nested_ensemble` | **k = 3 fixed a priori** | **0.5327** | 0.4870 |
| `c2_subject_relative` | k = 3 fixed a priori, + personalised features | 0.5199 | 0.5371 |
| `c3_inner_selected_k` | **k chosen per fold by inner CV** (mean k = 2.8) | **0.4898** | 0.5088 |

**Letting inner CV choose k made things worse** (0.533 → 0.490), landing back at
the deep model's 0.485. Adaptive selection on ~290 training samples adds more
variance than the flexibility is worth. c1's fixed k = 3 is itself a valid
nested procedure — k was set a priori, not tuned on outer folds — so **0.5327
remains the best legitimate non-personalised result**, and C3 is recorded as a
null for adaptive k.

#### The bias this campaign is accumulating

C3 was run to remove a post-hoc choice, and it surfaced a larger one. **Every
iteration is scored on the same five outer folds, and the campaign keeps the
best.** Each run's number is individually unbiased, but *the maximum over many
runs is not* — this is the same selection-on-test problem as §15.3, one level up.

Concretely: three runs so far, plus the 49 configs each explores internally.
The more variants tried, the more the campaign best drifts above its true
expected value. This does not invalidate any single number, but it does mean
**the campaign's running best must not be quoted as a clean estimate.**

Mitigations adopted from here:

1. **Every run is logged in `reports/runs_index.csv`, including failures** — the
   count of attempts is part of the result and is never pruned to flatter it.
2. **Pre-register each iteration's hypothesis in §15.5 before running it**, so
   the run count is honest rather than reconstructed afterwards.
3. **A final held-out confirmation is required before any SOTA claim.** The
   campaign best gets re-validated once, under a protocol fixed in advance, and
   *that* number is what gets reported — not the max over the search.

Given four consecutive nulls at the architecture level (§14.5) and now a null on
adaptive selection, the realistic expectation is that the leakage-free ceiling on
this data sits near **0.50–0.55**, not the 0.72 the published leaky protocol
reports.

### 15.7 Campaign result: plateau at ~0.52, confirmed out-of-search (2026-08-03)

**Stop condition met** — three consecutive iterations failed to beat `c1`
(C3 0.4898, C4 0.4979, C4b 0.5100 vs. 0.5327), so the campaign stopped and ran
the §15.6 confirmation instead of continuing to search.

#### Confirmation on fresh subject partitions

The five stored outer folds were reused by every iteration, so the campaign
maximum over them is biased. The frozen `c1` protocol (7 feature sets, 7 models,
k = 3 fixed a priori, no personalisation) was therefore re-evaluated on **three
subject partitions generated from seeds never used during the search**:

| partition | macro F1 |
|---|---|
| seed 101 | 0.5035 |
| seed 202 | 0.5499 |
| seed 303 | 0.5043 |
| **mean** | **0.5192** (sd 0.0265) |

**Search maximum 0.5327 → confirmed 0.5192. The search was +0.0135 optimistic**,
almost exactly the drift §15.6 predicted. The reportable number is **0.519**,
not 0.533.

#### Final standing

| system | macro F1 (leakage-free, 364 recs) | vs. majority (0.418) |
|---|---|---|
| **classical ensemble, confirmed out-of-search** | **0.519** | **+0.101** |
| classical ensemble, search maximum | 0.533 | +0.115 |
| deep MST-temporal (1.2 M params) | 0.485 | +0.067 |
| deep MST-static | 0.476 | +0.058 |
| majority class | 0.418 | — |

Net gain over the deep model: **+0.034** confirmed. Every iteration, including
the failures, is in `reports/runs_index.csv`; none were pruned.

#### Did we reach SOTA?

**No — and the honest answer is that the published number is not a target we can
legitimately match.** The 0.72 weighted F1 / 0.65 balanced accuracy in the origin
paper is measured with subjects on both sides of the split and SMOTE applied.
Under that protocol our own classical baselines already reach 0.740 weighted F1
(§10.2) — so "beating SOTA" is available any time we are willing to adopt the
leaky protocol, and it would mean nothing.

Under a protocol where no subject appears in both train and test, the ceiling on
this data is **~0.52 macro F1**, and this campaign reached it. Evidence that this
is a data ceiling and not a modelling failure:

- Four architecture-level nulls (temporal, cross-modal fusion, missing-modality,
  capacity/representation) — §9.3, §14.
- A 1.2 M-parameter transformer loses to an SVC on 32×32 pixel statistics — §15.3.
- Richer video features (248 regional-dynamics + LBP descriptors) made it
  *worse*, −0.023 — C4b.
- Adaptive model selection made it worse, −0.043 — C3.
- Nothing beat a simple soft-vote of three classical models on 290 training
  recordings.

#### What would actually move it

Not architecture. The two remaining levers both add **data or supervision**:

1. **§12.2 B3 / O7 — multi-dataset pretraining** (WESAD, K-EmoCon, SWELL). The
   binding constraint is 290 training recordings, and this is the only planned
   intervention that changes that number.
2. **§12.1 A4 — continuous 0–10 supervision** instead of a binarised label,
   which extracts more signal from labels already collected.

#### For the paper

The result to lead with is unchanged and is *strengthened* by this campaign: the
modality-availability confound (§10.5), the leakage-corrected benchmark (§10.3),
and now a documented, honestly-confirmed ceiling of ~0.52 under a leakage-free
protocol — against a literature reporting 0.72 under a leaky one. **The gap
between those two numbers is the paper.**
