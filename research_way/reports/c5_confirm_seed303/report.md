# Run report — `c5_confirm_seed303`

**Generated:** 2026-08-03 15:31:47  
**Primary metric (complete364_macro_f1):** **0.5043**  
**Majority-class reference:** 0.4180 (margin +0.0863)  
**Best previous:** `c5_confirm_seed202` at 0.5499 → **no improvement** (-0.0456)  
**Duration:** 287 s (4.8 min)  

> Primary metric is macro F1 on the 364 all-modality recordings
> under subject GroupKFold. Availability is constant there, so the
> modality-availability shortcut carries no signal (see §10.5).

## What changed

Iteration 5 - HELD-OUT CONFIRMATION of the c1 protocol (frozen: 7 original feature sets, 7 models, k=3 fixed a priori, no subject-relative). Evaluated on FRESH subject partitions (seed 303) never used during the search, because all five stored outer folds were reused by every iteration and the campaign maximum over them is optimistically biased (15.6). This is the number that may be reported, not the search maximum.

## Headline metrics

| metric | value |
|---|---|
| complete364_macro_f1 | 0.5043 |
| complete364_macro_f1_single_ref | 0.5111 |
| mean_chosen_k | 3.0000 |
| fixed_k | True |
| splits_seed | 303 |
| complete364_weighted_f1 | 0.5970 |
| complete364_balanced_acc | 0.5029 |
| complete364_accuracy | 0.5852 |
| n_folds | 5 |
| subject_relative | False |

## Per-fold outer results

| fold | n_test | chosen_k | ens_macro_f1 | ens_weighted_f1 | ens_balanced_acc | ens_accuracy | single_macro_f1 | single_weighted_f1 | single_balanced_acc | single_accuracy |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 84 | 3 | 0.4430 | 0.5137 | 0.4464 | 0.5238 | 0.4286 | 0.5238 | 0.4554 | 0.5714 |
| 1 | 70 | 3 | 0.5356 | 0.5966 | 0.5408 | 0.5857 | 0.5238 | 0.6000 | 0.5238 | 0.6000 |
| 2 | 84 | 3 | 0.6571 | 0.7571 | 0.6429 | 0.7738 | 0.6408 | 0.7220 | 0.6508 | 0.7143 |
| 3 | 77 | 3 | 0.5140 | 0.5472 | 0.5292 | 0.5325 | 0.5905 | 0.6091 | 0.6292 | 0.5974 |
| 4 | 49 | 3 | 0.3718 | 0.5704 | 0.3552 | 0.5102 | 0.3718 | 0.5704 | 0.3552 | 0.5102 |

## Inner-CV model ranking (mean over folds)

| features | model | inner_macro_f1 |
|---|---|---|
| video | logreg_c01 | 0.5617 |
| video | logreg | 0.5530 |
| audio+video | logreg | 0.5368 |
| audio+video | logreg_c01 | 0.5344 |
| video | hgb | 0.5214 |
| video | extratrees | 0.5175 |
| video | rf | 0.5138 |
| physfeat+video | logreg | 0.5092 |
| all | logreg_c01 | 0.5091 |
| physfeat+video | hgb | 0.5018 |
| physfeat+audio+video | logreg | 0.5016 |
| physfeat+video | logreg_c01 | 0.5003 |
| audio+video | hgb | 0.4994 |
| physfeat+audio+video | logreg_c01 | 0.4988 |
| physfeat+audio+video | hgb | 0.4973 |

## Per-fold selection

| fold | chosen_k | inner_score_at_k | k_grid_scores | picked |
|---|---|---|---|---|
| 0 | 3 | 0.6827 | 1:0.654 2:0.660 3:0.683 5:0.639 8:0.666 | video/rf | video/logreg_c01 | video/extratrees |
| 1 | 3 | 0.5657 | 1:0.554 2:0.589 3:0.566 5:0.556 8:0.578 | audio+video/logreg_c01 | video/logreg_c01 | audio+video/logreg |
| 2 | 3 | 0.4848 | 1:0.510 2:0.495 3:0.485 5:0.520 8:0.538 | video/logreg_c01 | video/logreg | video/hgb |
| 3 | 3 | 0.5605 | 1:0.560 2:0.548 3:0.561 5:0.560 8:0.571 | video/logreg_c01 | audio+video/logreg | video/logreg |
| 4 | 3 | 0.5738 | 1:0.584 2:0.557 3:0.574 5:0.524 8:0.538 | video/logreg | physfeat+video/logreg | video/logreg_c01 |

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
