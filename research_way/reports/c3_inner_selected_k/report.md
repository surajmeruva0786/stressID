# Run report — `c3_inner_selected_k`

**Generated:** 2026-08-03 14:28:17  
**Primary metric (complete364_macro_f1):** **0.4898**  
**Majority-class reference:** 0.4180 (margin +0.0718)  
**Best previous:** `c1_nested_ensemble` at 0.5327 → **no improvement** (-0.0430)  
**Duration:** 298 s (5.0 min)  

> Primary metric is macro F1 on the 364 all-modality recordings
> under subject GroupKFold. Availability is constant there, so the
> modality-availability shortcut carries no signal (see §10.5).

## What changed

Iteration 3 - CORRECTNESS FIX. c1 favoured the top-3 ensemble and c2 the single model; picking between them after seeing outer scores is the same test-set fishing that inflated video+SVC to 0.544. Now the ensemble size k is chosen per fold from inner out-of-fold predictions over train subjects only (K_GRID 1/2/3/5/8). The single-model column is reference only and is NOT selectable. No subject-relative features - directly comparable to c1.

## Headline metrics

| metric | value |
|---|---|
| complete364_macro_f1 | 0.4898 |
| complete364_macro_f1_single_ref | 0.5088 |
| mean_chosen_k | 2.8000 |
| complete364_weighted_f1 | 0.5808 |
| complete364_balanced_acc | 0.5166 |
| complete364_accuracy | 0.5864 |
| n_folds | 5 |
| subject_relative | False |

## Per-fold outer results

| fold | n_test | chosen_k | ens_macro_f1 | ens_weighted_f1 | ens_balanced_acc | ens_accuracy | single_macro_f1 | single_weighted_f1 | single_balanced_acc | single_accuracy |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 70 | 5 | 0.4647 | 0.5779 | 0.4796 | 0.6143 | 0.4562 | 0.5680 | 0.4694 | 0.6000 |
| 1 | 77 | 1 | 0.6404 | 0.6855 | 0.6562 | 0.6753 | 0.6404 | 0.6855 | 0.6562 | 0.6753 |
| 2 | 77 | 2 | 0.4029 | 0.5328 | 0.4010 | 0.4935 | 0.4373 | 0.5475 | 0.4515 | 0.5065 |
| 3 | 84 | 1 | 0.4219 | 0.5199 | 0.4708 | 0.5952 | 0.4219 | 0.5199 | 0.4708 | 0.5952 |
| 4 | 56 | 5 | 0.5191 | 0.5881 | 0.5751 | 0.5536 | 0.5882 | 0.6355 | 0.6905 | 0.6071 |

## Inner-CV model ranking (mean over folds)

| features | model | inner_macro_f1 |
|---|---|---|
| video | logreg_c01 | 0.5646 |
| video | logreg | 0.5405 |
| video | hgb | 0.5350 |
| video | extratrees | 0.5165 |
| physfeat+video | logreg | 0.4990 |
| physfeat+video | hgb | 0.4923 |
| audio+video | logreg | 0.4911 |
| audio+video | hgb | 0.4899 |
| audio+video | logreg_c01 | 0.4872 |
| physfeat+audio+video | hgb | 0.4865 |
| physfeat+audio+video | svc_lin | 0.4853 |
| physfeat+video | logreg_c01 | 0.4836 |
| all | logreg | 0.4833 |
| audio+video | svc_lin | 0.4829 |
| video | rf | 0.4813 |

## Per-fold selection

| fold | chosen_k | inner_score_at_k | k_grid_scores | picked |
|---|---|---|---|---|
| 0 | 5 | 0.6190 | 1:0.601 2:0.594 3:0.594 5:0.619 8:0.533 | video/hgb | video/logreg_c01 | physfeat+video/hgb | video/logreg | video/extratrees |
| 1 | 1 | 0.5713 | 1:0.571 2:0.538 3:0.541 5:0.511 8:0.505 | video/logreg_c01 |
| 2 | 2 | 0.6033 | 1:0.593 2:0.603 3:0.577 5:0.557 8:0.516 | video/logreg_c01 | video/logreg |
| 3 | 1 | 0.5899 | 1:0.590 2:0.566 3:0.571 5:0.571 8:0.581 | video/extratrees |
| 4 | 5 | 0.5521 | 1:0.539 2:0.546 3:0.548 5:0.552 8:0.520 | video/logreg_c01 | physfeat+audio+video/logreg | video/logreg | physfeat+audio+video/svc_lin | physfeat+video/logreg |

## Config

```json
{
  "subject_relative": false,
  "ensemble_top_k": 3,
  "feature_sets": [
    "video",
    "audio",
    "physfeat",
    "physfeat+video",
    "audio+video",
    "physfeat+audio+video",
    "all"
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
