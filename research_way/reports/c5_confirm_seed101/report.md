# Run report — `c5_confirm_seed101`

**Generated:** 2026-08-03 15:21:32  
**Primary metric (complete364_macro_f1):** **0.5035**  
**Majority-class reference:** 0.4180 (margin +0.0855)  
**Best previous:** `c1_nested_ensemble` at 0.5327 → **no improvement** (-0.0292)  
**Duration:** 281 s (4.7 min)  

> Primary metric is macro F1 on the 364 all-modality recordings
> under subject GroupKFold. Availability is constant there, so the
> modality-availability shortcut carries no signal (see §10.5).

## What changed

Iteration 5 - HELD-OUT CONFIRMATION of the c1 protocol (frozen: 7 original feature sets, 7 models, k=3 fixed a priori, no subject-relative). Evaluated on FRESH subject partitions (seed 101) never used during the search, because all five stored outer folds were reused by every iteration and the campaign maximum over them is optimistically biased (15.6). This is the number that may be reported, not the search maximum.

## Headline metrics

| metric | value |
|---|---|
| complete364_macro_f1 | 0.5035 |
| complete364_macro_f1_single_ref | 0.5155 |
| mean_chosen_k | 3.0000 |
| fixed_k | True |
| splits_seed | 101 |
| complete364_weighted_f1 | 0.5953 |
| complete364_balanced_acc | 0.5047 |
| complete364_accuracy | 0.5892 |
| n_folds | 5 |
| subject_relative | False |

## Per-fold outer results

| fold | n_test | chosen_k | ens_macro_f1 | ens_weighted_f1 | ens_balanced_acc | ens_accuracy | single_macro_f1 | single_weighted_f1 | single_balanced_acc | single_accuracy |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 84 | 3 | 0.5510 | 0.6380 | 0.5500 | 0.6429 | 0.5654 | 0.6142 | 0.5917 | 0.5952 |
| 1 | 77 | 3 | 0.4870 | 0.5477 | 0.4869 | 0.5455 | 0.5260 | 0.5454 | 0.5604 | 0.5325 |
| 2 | 63 | 3 | 0.4763 | 0.6179 | 0.4838 | 0.5873 | 0.5358 | 0.6397 | 0.5792 | 0.6032 |
| 3 | 77 | 3 | 0.4463 | 0.5221 | 0.4458 | 0.5195 | 0.4011 | 0.4685 | 0.3986 | 0.4545 |
| 4 | 63 | 3 | 0.5569 | 0.6508 | 0.5569 | 0.6508 | 0.5492 | 0.5953 | 0.5953 | 0.5714 |

## Inner-CV model ranking (mean over folds)

| features | model | inner_macro_f1 |
|---|---|---|
| video | logreg_c01 | 0.5790 |
| video | logreg | 0.5657 |
| video | hgb | 0.5336 |
| audio+video | logreg_c01 | 0.5308 |
| audio+video | hgb | 0.5245 |
| physfeat+video | logreg | 0.5230 |
| all | logreg_c01 | 0.5208 |
| physfeat+video | logreg_c01 | 0.5197 |
| all | logreg | 0.5174 |
| physfeat+audio+video | logreg | 0.5161 |
| audio+video | logreg | 0.5144 |
| video | extratrees | 0.5106 |
| physfeat+audio+video | logreg_c01 | 0.4990 |
| audio | logreg_c01 | 0.4975 |
| physfeat+video | hgb | 0.4899 |

## Per-fold selection

| fold | chosen_k | inner_score_at_k | k_grid_scores | picked |
|---|---|---|---|---|
| 0 | 3 | 0.5550 | 1:0.563 2:0.550 3:0.555 5:0.544 8:0.544 | video/logreg | video/logreg_c01 | video/hgb |
| 1 | 3 | 0.6490 | 1:0.630 2:0.633 3:0.649 5:0.641 8:0.645 | video/logreg_c01 | all/logreg_c01 | audio+video/logreg_c01 |
| 2 | 3 | 0.5540 | 1:0.569 2:0.556 3:0.554 5:0.540 8:0.509 | video/logreg_c01 | video/logreg | all/svc_lin |
| 3 | 3 | 0.6129 | 1:0.613 2:0.612 3:0.613 5:0.566 8:0.537 | video/logreg | video/logreg_c01 | video/hgb |
| 4 | 3 | 0.5201 | 1:0.527 2:0.525 3:0.520 5:0.507 8:0.519 | video/logreg_c01 | audio+video/logreg_c01 | audio+video/hgb |

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
