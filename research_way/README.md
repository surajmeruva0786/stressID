# StressID research pipeline — small-scale implementation

Runnable implementation of the pipeline in `StressID_Paper_Pipeline_Objectives (1).pdf`,
built and trained on a **small subset** of StressID so the whole thing runs end to end
in minutes rather than days.

```
python run_small.py            # full run
python run_small.py --smoke    # ~2 min sanity run (1 seed, 2 folds, 6 epochs)
python run_small.py --stage evaluate
```

## Small subset

| | |
|---|---|
| subjects | 16 (of 64), all with physio + video + audio |
| tasks | Breathing, Relax, Reading, Math, Stroop, Speaking |
| recordings | 96 (of 700) |
| binary-stress balance | 0.53 |
| audio coverage | 67% — Breathing/Relax have **no audio in StressID**, so the missing-modality path is exercised by real data, not just simulation |

Subject selection is seeded and deterministic (`Config.seed`): subjects with complete
modality coverage are ranked by how balanced their own labels are, then drawn.
Change `n_subjects` / `tasks` in `src/config.py` to scale up to the full dataset —
nothing else has to change.

## Stage map

| Stage (PDF) | Code | Status |
|---|---|---|
| −1 multi-dataset pretraining | — | **not implemented** (needs WESAD/SWELL-KW/K-EmoCon/… which aren't in this repo) |
| 0 leakage-free protocol + corrected baselines | `src/splits.py`, `src/baselines.py` | done (E1) |
| 1 temporal windowing | `src/preprocess.py` | done |
| 2 modality encoders | `src/model.py` | done |
| 3 cross-modal fusion + modality dropout | `src/model.py` | done |
| 4 temporal aggregation | `src/model.py` | done |
| 5 multi-task heads + losses | `src/model.py`, `src/losses.py` | done |
| 5.5 competitor re-implementation | — | **not implemented** (E0 needs the 3 competitor papers' code) |
| 6 evaluation suite | `src/evaluate.py` | E2, E4, E5, E6, E8, E12, E13 done; E7/E10/E11 need other datasets |

## Deviations from the planning document (and why)

1. **10 s windows, not 30 s.** The PDF assumes 3–5 min recordings. Real StressID tasks
   are 60–90 s, so 30 s / 50 % overlap yields only 3 windows per task — not enough of a
   sequence to test temporal modelling. 10 s / 5 s hop gives ~11 windows for a 60 s task.
   Set `window_sec`/`hop_sec` in `src/config.py` to change this.
2. **Log-mel + CNN instead of wav2vec2.** `transformers` isn't installed and a wav2vec2
   fine-tune doesn't fit a 4 GB GPU at this scale. `AudioEncoder` is a drop-in slot:
   swap it for a wav2vec2 feature extractor emitting `[N, k, d]` and E9 becomes runnable.
3. **Face-crop pixels instead of OpenFace AUs.** OpenFace isn't available here. Video is
   decoded at 2.5 fps, the face is located once with a Haar cascade and the crop is
   resized to 32×32 grayscale. `VideoEncoder` is a drop-in slot: replace `frame_cnn`
   with a linear projection of per-frame AU vectors and keep the frame Transformer.
   The repo does ship *task-level* AU aggregates
   (`stressID-main/Feature Extraction/Features/video11tasks_*.csv`) but those are static
   per-task vectors, which is exactly what the temporal stage needs to avoid.
4. **Static baseline = mean-pooled window sequence.** The E12 "static" arm shares the
   encoders and fusion block with the temporal arm and differs only in Stage 4
   (mean pool vs. Transformer over time), so the ablation isolates temporal modelling
   and nothing else.

## ⚠ Confound found while building this: modality availability leaks the label

In StressID, audio exists **only for the 7 speech tasks**, and those are mostly the
stressful ones. Across the **whole dataset** (not just this subset):

```
P(stress | audio file exists) = 0.683
P(stress | no audio file)     = 0.246
```

So "is there an audio file?" is itself a strong stress cue. The `availability_only`
row in `baselines.csv` makes the cost concrete — it is a classifier fed **only the
three modality-presence bits and no signal content whatsoever**:

| baseline (GroupKFold) | accuracy | macro F1 |
|---|---|---|
| physio (logreg) | 0.646 | 0.642 |
| audio (logreg) | 0.708 | 0.704 |
| video (logreg) | 0.510 | 0.503 |
| feature fusion (RF) | 0.750 | 0.747 |
| **availability_only** | **0.781** | **0.774** |

Three modality-presence bits beat every real multimodal baseline. Any model that can
observe which modalities are present — which is exactly what a missing-modality
architecture does by construction — can score ~0.78 without learning anything about
stress. This matters directly for the paper's headline claim (E12): a
missing-modality robustness result on StressID is **not interpretable** unless
availability is decorrelated from the label.

What this implementation does about it:

1. **E5/E6/E12 are evaluated only on recordings where all three modalities naturally
   exist** (64 of 96 here), so availability is constant across the compared conditions
   and cannot drive the degradation curve.
2. **The `availability_only` probe is reported alongside every baseline**, so the
   shortcut's size is always visible rather than folded into a "multimodal" number.
3. **Modality dropout during training** (25 %/modality) partially decorrelates presence
   from label, but does not remove the confound from the training labels themselves.

What it does **not** yet do, and what you should decide before the paper: training still
sees the confounded availability pattern. The clean fix is to restrict the corpus to the
7 speech tasks (`Counting1/2/3, Math, Reading, Speaking, Stroop`), where audio is always
present and availability carries zero label information — at the cost of a more skewed
binary label balance (~0.68 positive) and losing the natural missing-modality cases.
Set `tasks` in `src/config.py` to run it that way.

## Design notes worth knowing

- **Physio normalisation is per subject, not per recording.** Stats are pooled over each
  subject's own recordings. Per-recording z-scoring would erase the between-task EDA/HR
  level shifts that carry most of the stress signal.
- **The window grid is defined on physio** (the only always-present modality) and the
  boundaries are converted to seconds, so audio and video windows are time-aligned to it.
- **Attention never sees a fully-masked key set.** `CrossModalBlock` carries a learned
  always-valid `null` key/value, and `FusionHead` un-masks the row if every modality is
  dropped. This is what makes 2-missing inference return a number instead of NaN.
- **Missing-modality evaluation is restricted to recordings where all three modalities
  naturally exist**, otherwise `no_audio` would be scored on recordings that never had
  audio and the degradation curve would be meaningless.
- **Every headline number carries a bootstrap 95 % CI** over (seed × fold) runs, and
  temporal-vs-static comparisons use a paired bootstrap p-value.

## Outputs

Written to `results/small/`:

| file | content |
|---|---|
| `baselines.csv` | E1 — classical baselines, GroupKFold vs. leaky random KFold |
| `e1_leakage_inflation.csv` | E1 — how much subject leakage inflates each baseline |
| `predictions.csv` | raw per-recording predictions × 7 modality conditions × seed × fold |
| `per_run_metrics.csv` | metrics per (variant, condition, seed, fold) |
| `e2_temporal_vs_static.csv` | E2/E4 — paired comparison at full modality |
| `e5e6_missing_modality.csv` | E5/E6 — per-condition F1 / accuracy / ECE with CIs |
| `e8_per_class_recall.csv` | E8 — 3-class per-class recall |
| `e12_degradation.csv`, `e12_paired_tests.csv` | E12 — the headline degradation curves |
| `e13_calibration.csv`, `e13_reliability_bins.csv` | E13 — ECE and reliability bins |
| `figures.png` | degradation curve, ECE vs. #missing, reliability diagram |

## What the first full run actually found

24 runs (3 seeds × 4 folds × {temporal, static}), 40 epochs each, ~20 min on a Quadro P1000.

**The shortcut audit is the result.** The same full-modality predictions, scored two ways:

| scope | variant | n | pos rate | macro F1 | majority-class F1 | margin |
|---|---|---|---|---|---|---|
| all | static | 96 | 0.53 | 0.621 | 0.347 | **+0.274** |
| all | temporal | 96 | 0.53 | 0.705 | 0.347 | **+0.358** |
| complete | static | 64 | 0.73 | 0.452 | 0.423 | +0.028 |
| complete | temporal | 64 | 0.73 | 0.417 | 0.423 | **−0.007** |

Scored on all test recordings, this looks like a clean result — temporal beats static by
0.084 macro F1, exactly the story E2 is supposed to tell. Scored on the 64 recordings
where all three modalities are present, **both variants sit at the majority-class
reference**. The apparent margin was the availability shortcut, not stress signal.

Everything downstream follows from that, and none of it should be read as a finding:

- **E2** temporal − static = −0.035 [−0.105, +0.046], p = 0.375. No effect.
- **E12** degradation curves are flat-to-*rising* — temporal macro F1 goes 0.417 → 0.440 →
  0.447 as modalities are removed. A model whose score improves when you delete its
  inputs is not using those inputs.
- **E8** 3-class recall is 0.08 / 0.29 / 0.45 — the minority-class collapse the PDF's O6
  describes, in its most severe form.
- **E13** ECE is the one number that behaves sensibly: the temporal model is markedly
  better calibrated than the static one (0.127 vs 0.233) and stays flat under missing
  modalities. With the accuracy at chance, this mostly says it is confidently uncertain.

This is the expected outcome at this scale, not a bug: 96 recordings, ~57 training
recordings per fold, 1.2 M parameters, encoders trained from scratch. Stage −1
(multi-dataset pretraining) exists in the plan precisely because StressID is too small to
train these encoders cold. The pipeline is doing its job — it is correctly reporting that
there is no signal yet, instead of laundering the shortcut into a headline number.

Before scaling up, decide the confound question above. If availability stays correlated
with the label, a larger run will produce *better-looking* numbers with the same problem.

## Caveat on the numbers

16 subjects × 6 tasks = 96 recordings, 4 folds → ~24 test recordings per fold. Confidence
intervals will be wide and fold-to-fold variance large. This subset run is for validating
that the pipeline is correct and leakage-free, **not** for drawing the paper's conclusions.
Scale `n_subjects` to 64 and `tasks` to all 11 for numbers worth reporting.
