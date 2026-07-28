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

## Caveat on the numbers

16 subjects × 6 tasks = 96 recordings, 4 folds → ~24 test recordings per fold. Confidence
intervals will be wide and fold-to-fold variance large. This subset run is for validating
that the pipeline is correct and leakage-free, **not** for drawing the paper's conclusions.
Scale `n_subjects` to 64 and `tasks` to all 11 for numbers worth reporting.
