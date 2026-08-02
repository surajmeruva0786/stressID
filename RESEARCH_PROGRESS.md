# StressID — Research Progress & Future Work Plan

> **Last updated:** 2026-08-02  
> **Status:** Full-corpus run complete and benchmarked against prior work —
> see `research_way/` and §9–§11 below. Improvement plan in §12, project
> history in §13. **In progress:** §12 A1 (domain physio features) + A2
> (capacity reduction / regularisation).
> The planned architecture is implemented and trained on all 700 recordings.
> Headline finding is negative-but-publishable: a modality-availability confound
> in StressID dominates every multimodal number, and the temporal/fusion
> contributions are not supported once it is controlled for.
> Model weights are now persisted per fold and the corpus has been retrained to
> produce them — 30 checkpoints in `research_way/results/full/checkpoints/` (§11).
> The retrain reproduces every headline number (§11.5).

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
