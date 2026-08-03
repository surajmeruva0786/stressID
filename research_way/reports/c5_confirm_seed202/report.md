# Run report — `c5_confirm_seed202`

**Generated:** 2026-08-03 15:26:17  
**Primary metric (complete364_macro_f1):** **0.5499**  
**Majority-class reference:** 0.4180 (margin +0.1319)  
**Best previous:** `c1_nested_ensemble` at 0.5327 → **NEW BEST** (+0.0171)  
**Duration:** 278 s (4.6 min)  

> Primary metric is macro F1 on the 364 all-modality recordings
> under subject GroupKFold. Availability is constant there, so the
> modality-availability shortcut carries no signal (see §10.5).

## What changed

Iteration 5 - HELD-OUT CONFIRMATION of the c1 protocol (frozen: 7 original feature sets, 7 models, k=3 fixed a priori, no subject-relative). Evaluated on FRESH subject partitions (seed 202) never used during the search, because all five stored outer folds were reused by every iteration and the campaign maximum over them is optimistically biased (15.6). This is the number that may be reported, not the search maximum.

## Headline metrics

| metric | value |
|---|---|
| complete364_macro_f1 | 0.5499 |
| complete364_macro_f1_single_ref | 0.5368 |
| mean_chosen_k | 3.0000 |
| fixed_k | True |
| splits_seed | 202 |
| complete364_weighted_f1 | 0.6380 |
| complete364_balanced_acc | 0.5659 |
| complete364_accuracy | 0.6247 |
| n_folds | 5 |
| subject_relative | False |

## Per-fold outer results

| fold | n_test | chosen_k | ens_macro_f1 | ens_weighted_f1 | ens_balanced_acc | ens_accuracy | single_macro_f1 | single_weighted_f1 | single_balanced_acc | single_accuracy |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 77 | 3 | 0.4676 | 0.5867 | 0.4678 | 0.5714 | 0.4949 | 0.5647 | 0.5304 | 0.5325 |
| 1 | 70 | 3 | 0.5761 | 0.5870 | 0.5929 | 0.6143 | 0.5358 | 0.5444 | 0.5428 | 0.5571 |
| 2 | 77 | 3 | 0.4639 | 0.6879 | 0.4975 | 0.6234 | 0.4083 | 0.6733 | 0.3938 | 0.6104 |
| 3 | 70 | 3 | 0.6500 | 0.7186 | 0.6553 | 0.7143 | 0.6599 | 0.7221 | 0.6718 | 0.7143 |
| 4 | 70 | 3 | 0.5917 | 0.6100 | 0.6159 | 0.6000 | 0.5850 | 0.5905 | 0.6350 | 0.5857 |

## Inner-CV model ranking (mean over folds)

| features | model | inner_macro_f1 |
|---|---|---|
| video | logreg_c01 | 0.5527 |
| video | logreg | 0.5352 |
| audio+video | logreg_c01 | 0.5229 |
| physfeat+audio+video | hgb | 0.5089 |
| audio | logreg | 0.5058 |
| audio+video | hgb | 0.5050 |
| video | hgb | 0.5042 |
| audio | logreg_c01 | 0.5040 |
| all | hgb | 0.5024 |
| audio+video | logreg | 0.5015 |
| physfeat+video | logreg_c01 | 0.5002 |
| all | logreg_c01 | 0.4986 |
| physfeat+video | hgb | 0.4962 |
| all | logreg | 0.4951 |
| physfeat+video | logreg | 0.4948 |

## Per-fold selection

| fold | chosen_k | inner_score_at_k | k_grid_scores | picked |
|---|---|---|---|---|
| 0 | 3 | 0.5844 | 1:0.600 2:0.595 3:0.584 5:0.592 8:0.607 | audio+video/logreg_c01 | audio/logreg | video/hgb |
| 1 | 3 | 0.6022 | 1:0.589 2:0.600 3:0.602 5:0.524 8:0.517 | video/logreg | video/logreg_c01 | physfeat/hgb |
| 2 | 3 | 0.5476 | 1:0.547 2:0.550 3:0.548 5:0.561 8:0.554 | audio/logreg | audio/logreg_c01 | video/logreg_c01 |
| 3 | 3 | 0.5153 | 1:0.541 2:0.527 3:0.515 5:0.538 8:0.514 | video/logreg_c01 | video/logreg | physfeat+video/logreg_c01 |
| 4 | 3 | 0.5435 | 1:0.535 2:0.540 3:0.543 5:0.553 8:0.580 | video/logreg_c01 | video/logreg | audio/logreg_c01 |

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
