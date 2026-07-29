# StressID — Research Progress & Future Work Plan

> **Last updated:** 2026-07-29  
> **Status:** Full-corpus run complete — see `research_way/` and §9 below.
> The planned architecture is implemented and trained on all 700 recordings.
> Headline finding is negative-but-publishable: a modality-availability confound
> in StressID dominates every multimodal number, and the temporal/fusion
> contributions are not supported once it is controlled for.

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
