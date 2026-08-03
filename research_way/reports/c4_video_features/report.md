# Run report — `c4_video_features`

**Generated:** 2026-08-03 15:06:40  
**Primary metric (complete364_macro_f1):** **0.4979**  
**Majority-class reference:** 0.4180 (margin +0.0799)  
**Best previous:** `c1_nested_ensemble` at 0.5327 → **no improvement** (-0.0348)  
**Duration:** 549 s (9.1 min)  

> Primary metric is macro F1 on the 364 all-modality recordings
> under subject GroupKFold. Availability is constant there, so the
> modality-availability shortcut carries no signal (see §10.5).

## What changed

Iteration 4 (pre-registered in 15.5). Video is the strongest single modality but was described by a global pixel mean/std plus a 64-element diff-energy vector. C4 adds 248 features from the same 32x32 crop cache: regional temporal dynamics on a 4x4 grid (motion energy, peak motion, texture spread per region) and per-region uniform LBP histograms. OpenFace AUs would be better but need the raw video re-decoded at full resolution. Ensemble size fixed a priori at k=3 to match c1; C3 showed adaptive k is worse.

## Headline metrics

| metric | value |
|---|---|
| complete364_macro_f1 | 0.4979 |
| complete364_macro_f1_single_ref | 0.5116 |
| mean_chosen_k | 3.4000 |
| complete364_weighted_f1 | 0.5912 |
| complete364_balanced_acc | 0.5249 |
| complete364_accuracy | 0.5994 |
| n_folds | 5 |
| subject_relative | False |

## Per-fold outer results

| fold | n_test | chosen_k | ens_macro_f1 | ens_weighted_f1 | ens_balanced_acc | ens_accuracy | single_macro_f1 | single_weighted_f1 | single_balanced_acc | single_accuracy |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 70 | 5 | 0.4647 | 0.5779 | 0.4796 | 0.6143 | 0.4562 | 0.5680 | 0.4694 | 0.6000 |
| 1 | 77 | 1 | 0.6404 | 0.6855 | 0.6562 | 0.6753 | 0.6404 | 0.6855 | 0.6562 | 0.6753 |
| 2 | 77 | 5 | 0.4435 | 0.5847 | 0.4426 | 0.5584 | 0.4515 | 0.5947 | 0.4510 | 0.5714 |
| 3 | 84 | 1 | 0.4219 | 0.5199 | 0.4708 | 0.5952 | 0.4219 | 0.5199 | 0.4708 | 0.5952 |
| 4 | 56 | 5 | 0.5191 | 0.5881 | 0.5751 | 0.5536 | 0.5882 | 0.6355 | 0.6905 | 0.6071 |

## Inner-CV model ranking (mean over folds)

| features | model | inner_macro_f1 |
|---|---|---|
| video | logreg_c01 | 0.5646 |
| video | logreg | 0.5405 |
| video | hgb | 0.5350 |
| videofeat+video | logreg_c01 | 0.5298 |
| videofeat+video | logreg | 0.5280 |
| videofeat | logreg_c01 | 0.5219 |
| videofeat | logreg | 0.5169 |
| video | extratrees | 0.5165 |
| videofeat | hgb | 0.5148 |
| physfeat+video | logreg | 0.4990 |
| videofeat+audio | logreg | 0.4986 |
| all+videofeat | logreg | 0.4986 |
| videofeat+video | hgb | 0.4973 |
| videofeat+physfeat+audio | logreg | 0.4970 |
| videofeat | svc_lin | 0.4957 |

## Per-fold selection

| fold | chosen_k | inner_score_at_k | k_grid_scores | picked |
|---|---|---|---|---|
| 0 | 5 | 0.6190 | 1:0.601 2:0.594 3:0.594 5:0.619 8:0.538 | video/hgb | video/logreg_c01 | physfeat+video/hgb | video/logreg | video/extratrees |
| 1 | 1 | 0.5713 | 1:0.571 2:0.538 3:0.531 5:0.508 8:0.496 | video/logreg_c01 |
| 2 | 5 | 0.6235 | 1:0.601 2:0.595 3:0.601 5:0.624 8:0.604 | videofeat/logreg_c01 | videofeat/logreg | videofeat+video/logreg_c01 | video/logreg_c01 | video/logreg |
| 3 | 1 | 0.5899 | 1:0.590 2:0.566 3:0.571 5:0.580 8:0.577 | video/extratrees |
| 4 | 5 | 0.5521 | 1:0.539 2:0.546 3:0.548 5:0.552 8:0.520 | video/logreg_c01 | physfeat+audio+video/logreg | video/logreg | physfeat+audio+video/svc_lin | physfeat+video/logreg |

## Config

```json
{
  "subject_relative": false,
  "ensemble_top_k": 3,
  "feature_sets": [
    "videofeat",
    "videofeat+video",
    "videofeat+audio",
    "videofeat+physfeat+audio",
    "video",
    "audio",
    "physfeat",
    "physfeat+video",
    "audio+video",
    "physfeat+audio+video",
    "all",
    "all+videofeat"
  ],
  "models": [
    "logreg",
    "logreg_c01",
    "svc_rbf",
    "svc_lin",
    "rf",
    "extratrees",
    "hgb"
  ],
  "n_folds": 5,
  "data_tag": "full",
  "selection": "nested GroupKFold (inner selection on train subjects only)"
}
```

## Environment

- Windows-10-10.0.26200-SP0
- Python 3.11.9
