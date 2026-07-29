# StressID research pipeline

Runnable implementation of the pipeline in `StressID_Paper_Pipeline_Objectives (1).pdf`,
at two scales. The small subset exists to validate the pipeline in minutes; the full
corpus is the one to report from.

```
python run_full.py             # ENTIRE corpus: 64 subjects, 11 tasks, 700 recordings
python run_full.py --stage evaluate
python run_full.py --workers 8 --stage preprocess

python run_small.py            # 16-subject / 6-task subset
python run_small.py --smoke    # ~2 min sanity run (1 seed, 2 folds, 6 epochs)
```

Artefacts are namespaced by scale, so both runs coexist:

| | small | full |
|---|---|---|
| manifest | `data/manifest_small.csv` | `data/manifest_full.csv` |
| splits | `data/splits_small.json` | `data/splits_full.json` |
| window cache | `data/cache_small/` | `data/cache_full/` |
| results | `results/small/` | `results/full/` |

The window cache **must** be per-scale: physio is z-scored per subject with stats
pooled over that subject's recordings *in the manifest*, so the same `subject_task`
window sequence genuinely differs between the 6-task subset and the 11-task corpus.

Full-scale cost on a Quadro P1000 / 12-core CPU: preprocessing 16 min (6 workers),
baselines 1.5 min, training 3 h (30 fold-models), evaluation 4 s.

## Full corpus

| | |
|---|---|
| subjects | 64 (all of them) |
| tasks | all 11 |
| recordings | 700 |
| binary-stress balance | 0.526 |
| affect3 | 253 / 202 / 245 |
| modality coverage | physio 100%, audio 54%, video 83% |
| protocol | 5-fold subject GroupKFold, 41 train / 10 val / 13 test subjects |

`src/config.py:full_config()` is the preset. Nothing about the model or the stage
code differs from the small run — only `n_subjects`, `tasks`, `n_folds` and the
`require_*` filters.

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
   are 60 s for the four speech tasks (Reading, Math, Stroop, Speaking), 150 s for Relax
   and 180 s for Breathing. 30 s / 50 % overlap yields only 3 windows for a 60 s task —
   not enough of a sequence to test temporal modelling. 10 s / 5 s hop gives 11 windows.
   Set `window_sec`/`hop_sec` in `src/config.py` to change this.
   Note `max_windows = 16` truncates Breathing/Relax to their first ~85 s. This does not
   affect E5/E6/E12, which score only the 60 s speech tasks.
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
stressful ones. Over the full 700-recording corpus:

```
P(stress | audio file exists) = 0.71
P(stress | no audio file)     = 0.31
```

So "is there an audio file?" is itself a strong stress cue. The `availability_only`
row in `baselines.csv` makes the cost concrete — it is a classifier fed **only the
three modality-presence bits and no signal content whatsoever**:

| baseline (5-fold GroupKFold, 700 recordings) | accuracy | macro F1 |
|---|---|---|
| physio (logreg / RF) | 0.569 / 0.546 | 0.565 / 0.540 |
| audio (logreg / RF) | 0.689 / 0.681 | 0.686 / 0.678 |
| video (logreg / RF) | 0.601 / 0.626 | 0.590 / 0.622 |
| feature fusion (RF) | 0.691 | 0.685 |
| decision fusion (RF) | 0.690 | 0.688 |
| **availability_only** | **0.700 / 0.698** | **0.697 / 0.696** |

Three modality-presence bits beat every real multimodal baseline. Any model that can
observe which modalities are present — which is exactly what a missing-modality
architecture does by construction — can score ~0.78 without learning anything about
stress. This matters directly for the paper's headline claim (E12): a
missing-modality robustness result on StressID is **not interpretable** unless
availability is decorrelated from the label.

What this implementation does about it:

1. **E5/E6/E12 are evaluated only on recordings where all three modalities naturally
   exist** (364 of 700 at full scale; 64 of 96 in the subset), so availability is
   constant across the compared conditions and cannot drive the degradation curve.
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

Written to `results/full/` (or `results/small/` for the subset run):

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
| `e12_vs_full_tests.csv` | E12b — each condition vs its own full-modality run, Bonferroni-corrected |
| `shortcut_audit.csv` | full-modality predictions scored on all vs complete-modality recordings |
| `e13_calibration.csv`, `e13_reliability_bins.csv` | E13 — ECE and reliability bins |
| `figures.png` | degradation curve, ECE vs. #missing, reliability diagram |

## What the FULL-CORPUS run found

30 runs (3 seeds × 5 folds × {temporal, static}), 40 epochs each, ~3 h on a Quadro P1000.
700 recordings, 64 subjects, 11 tasks. Results in `results/full/`.

**The shortcut audit is still the result — but the signal is no longer zero.**

| scope | variant | n | pos rate | macro F1 [95% CI] | majority-class F1 | margin |
|---|---|---|---|---|---|---|
| all | static | 700 | 0.53 | 0.654 [0.634, 0.671] | 0.345 | +0.309 |
| all | temporal | 700 | 0.53 | 0.671 [0.648, 0.692] | 0.345 | +0.326 |
| complete | static | 364 | 0.72 | 0.476 [0.452, 0.504] | 0.418 | +0.058 |
| complete | temporal | 364 | 0.72 | 0.487 [0.451, 0.525] | 0.418 | +0.069 |

Read the two scopes together. On all 700 recordings the model looks strong (+0.31 over
the majority reference) — but it scores **below the `availability_only` probe**
(0.671 vs 0.697 macro F1). A 1.2 M-parameter multimodal transformer losing to three
presence bits is the cleanest possible statement that the "all" column is measuring
availability, not stress.

On the 364 naturally-complete recordings, where availability is constant and cannot be
read, the margin drops to +0.058 / +0.069. That is small, but unlike the 16-subject
subset (where it was **−0.007 / +0.028**, i.e. at or below chance) it is now reliably
positive: both CI lower bounds (0.452, 0.451) sit above the 0.418 majority reference.
Scaling from 96 to 700 recordings turned "no measurable signal" into "a small,
statistically resolvable one". That is the honest headline.

Everything else is null, and should be reported as null:

- **E2 / E4** temporal − static = **+0.011 [−0.041, +0.063], p = 0.677** (n = 15 paired
  runs). No temporal effect. The subset run gave −0.035, p = 0.375; at 7× the data the
  estimate has moved to ~0 with a tighter interval, so this is now a reasonably
  well-powered null rather than an underpowered one.
- **E12** degradation curves are flat to slightly *rising*: static 0.476 → 0.484 → 0.486
  and temporal 0.487 → 0.483 → 0.489 as 0 → 1 → 2 modalities are removed.
- **E12b** (`e12_vs_full_tests.csv`) tests that directly — does removing a modality cost
  anything? Of 12 paired comparisons, exactly one is nominally significant
  (static / `no_audio`, +0.053 [+0.006, +0.098], p = 0.022) and it does **not** survive
  Bonferroni correction for 12 tests. One nominal hit in twelve is what chance produces.
  The defensible reading is that no modality is contributing measurably — and the sign
  on `no_audio` and `physio_only` (both positive in both variants) hints that audio is
  if anything a distractor once availability is held constant.
- **E8** 3-class recall is 0.05 / 0.39 / 0.60 (static) and 0.05 / 0.37 / 0.71 (temporal),
  macro F1 0.318 for both — the minority-class collapse of the PDF's O6, essentially
  unchanged by 7× more data.
- **E13** ECE 0.109–0.146, and now the *static* model is the better-calibrated one
  (0.109 vs 0.146 at full modality), reversing the subset result. With accuracy this
  close to the reference, ECE differences of this size are not worth interpreting.

**Training behaviour worth knowing.** Train BCE falls to ~0.09 while validation F1 peaks
early and decays — the model memorises the ~450 training recordings. Best-val epoch
across the 30 runs is scattered from 1 to 22 with nothing selected past 22, so the back
half of the 40-epoch OneCycle schedule is dead weight, and the spread looks more like
val-F1 noise on 10 validation subjects than a stable early-stopping point. Best-checkpoint
selection makes the reported numbers valid, but a shorter schedule with explicit early
stopping would be the honest configuration.

## What the earlier 16-subject subset run found

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

This was the expected outcome at that scale, not a bug: 96 recordings, ~57 training
recordings per fold, 1.2 M parameters, encoders trained from scratch. The pipeline was
doing its job — correctly reporting that there was no signal yet, instead of laundering
the shortcut into a headline number.

**The full-corpus run confirmed the prediction made here.** Scaling to 700 recordings did
produce better-looking numbers with the same confound intact (+0.31 margin on "all",
+0.06 on "complete"), exactly as warned. What it also produced is a small but reliably
non-zero margin on the complete subset, which the subset run could not resolve.

## Where this leaves the paper

1. **The confound is the strongest result in the repo.** A classifier fed three
   presence bits reaches 0.697 macro F1 and beats the trained multimodal model
   (0.671) on the same recordings, under a leakage-free protocol on the full corpus.
   That is a methodological finding about StressID that stands on its own, and it
   invalidates any missing-modality robustness claim on this dataset that does not
   control for availability.
2. **The architecture contribution (E2/E12) is not supported by the data.** Temporal
   vs static is +0.011, p = 0.677; no modality removal costs measurable F1 after
   correction. Reporting these as positive results would not survive review.
3. **The remaining lever is Stage −1** (multi-dataset pretraining, WESAD / SWELL-KW /
   K-EmoCon). 448 training recordings is not enough to train three encoders and a
   fusion transformer cold — which the memorisation behaviour above shows directly.
4. **The clean-corpus variant is one config change away.** Setting `tasks` to the 7
   speech tasks makes availability carry zero label information (audio always present),
   at the cost of a skewed 0.72 positive rate. That run answers "is there stress signal
   at all, with the shortcut fully removed?" — note the complete-modality subset already
   *is* those 7 tasks, so the +0.058 / +0.069 margins above are effectively that answer
   at inference time, but with training still exposed to the confound.
