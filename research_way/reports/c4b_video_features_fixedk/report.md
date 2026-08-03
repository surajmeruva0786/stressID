# Run report — `c4b_video_features_fixedk`

**Generated:** 2026-08-03 15:16:02  
**Primary metric (complete364_macro_f1):** **0.5100**  
**Majority-class reference:** 0.4180 (margin +0.0920)  
**Best previous:** `c1_nested_ensemble` at 0.5327 → **no improvement** (-0.0228)  
**Duration:** 531 s (8.8 min)  

> Primary metric is macro F1 on the 364 all-modality recordings
> under subject GroupKFold. Availability is constant there, so the
> modality-availability shortcut carries no signal (see §10.5).

## What changed

Iteration 4b - CORRECTED. The c4 run was mislabelled: its notes claimed k fixed at 3, but the code still used C3's inner-CV-selected k (mean 3.4), confounding the video-feature change with a mechanism §15.6 measured at -0.043. This re-run pins k=3 a priori, making it directly comparable to c1 (0.5327). Same 248 video features: 4x4 regional dynamics + per-region uniform LBP.

## Headline metrics

| metric | value |
|---|---|
| complete364_macro_f1 | 0.5100 |
| complete364_macro_f1_single_ref | 0.5116 |
| mean_chosen_k | 3.0000 |
| fixed_k | True |
| complete364_weighted_f1 | 0.5950 |
| complete364_balanced_acc | 0.5300 |
| complete364_accuracy | 0.5922 |
| n_folds | 5 |
| subject_relative | False |

## Per-fold outer results

| fold | n_test | chosen_k | ens_macro_f1 | ens_weighted_f1 | ens_balanced_acc | ens_accuracy | single_macro_f1 | single_weighted_f1 | single_balanced_acc | single_accuracy |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 70 | 3 | 0.5044 | 0.6092 | 0.5136 | 0.6429 | 0.4562 | 0.5680 | 0.4694 | 0.6000 |
| 1 | 77 | 3 | 0.6629 | 0.7087 | 0.6747 | 0.7013 | 0.6404 | 0.6855 | 0.6562 | 0.6753 |
| 2 | 77 | 3 | 0.4183 | 0.5702 | 0.4132 | 0.5455 | 0.4515 | 0.5947 | 0.4510 | 0.5714 |
| 3 | 84 | 3 | 0.4510 | 0.5177 | 0.4580 | 0.5357 | 0.4219 | 0.5199 | 0.4708 | 0.5952 |
| 4 | 56 | 3 | 0.5134 | 0.5692 | 0.5903 | 0.5357 | 0.5882 | 0.6355 | 0.6905 | 0.6071 |

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
| 0 | 3 | 0.5939 | 1:0.601 2:0.594 3:0.594 5:0.619 8:0.538 | video/hgb | video/logreg_c01 | physfeat+video/hgb |
| 1 | 3 | 0.5307 | 1:0.571 2:0.538 3:0.531 5:0.508 8:0.496 | video/logreg_c01 | video/logreg | videofeat+video/logreg_c01 |
| 2 | 3 | 0.6010 | 1:0.601 2:0.595 3:0.601 5:0.624 8:0.604 | videofeat/logreg_c01 | videofeat/logreg | videofeat+video/logreg_c01 |
| 3 | 3 | 0.5713 | 1:0.590 2:0.566 3:0.571 5:0.580 8:0.577 | video/extratrees | audio/logreg | audio+video/logreg |
| 4 | 3 | 0.5484 | 1:0.539 2:0.546 3:0.548 5:0.552 8:0.520 | video/logreg_c01 | physfeat+audio+video/logreg | video/logreg |

## Config

```json
{
  "subject_relative": false,
  "ensemble_top_k": 3,
  "fixed_k": true,
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
