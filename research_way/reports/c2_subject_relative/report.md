# Run report — `c2_subject_relative`

**Generated:** 2026-08-03 14:20:37  
**Primary metric (complete364_macro_f1):** **0.5199**  
**Majority-class reference:** 0.4180 (margin +0.1019)  
**Best previous:** `c1_nested_ensemble` at 0.5327 → **no improvement** (-0.0128)  
**Duration:** 294 s (4.9 min)  

> Primary metric is macro F1 on the 364 all-modality recordings
> under subject GroupKFold. Availability is constant there, so the
> modality-availability shortcut carries no signal (see §10.5).

## What changed

Iteration 2. B1 subject-adaptive calibration: every feature expressed relative to that subject's own Relax/Breathing baseline. Those tasks have no audio so they are outside the 364-recording evaluation subset - no evaluated sample is touched and no label is read. This is calibration, not leakage, but it is a PERSONALISED setting and must be reported separately from c1.

## Headline metrics

| metric | value |
|---|---|
| complete364_macro_f1 | 0.5199 |
| complete364_macro_f1_single | 0.5371 |
| complete364_weighted_f1 | 0.6378 |
| complete364_balanced_acc | 0.5457 |
| complete364_accuracy | 0.6838 |
| n_folds | 5 |
| subject_relative | True |

## Per-fold outer results

| fold | n_test | ens_macro_f1 | ens_weighted_f1 | ens_balanced_acc | ens_accuracy | single_macro_f1 | single_weighted_f1 | single_balanced_acc | single_accuracy |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 70 | 0.4750 | 0.5650 | 0.4762 | 0.5714 | 0.6259 | 0.6857 | 0.6259 | 0.6857 |
| 1 | 77 | 0.4571 | 0.6075 | 0.5217 | 0.7143 | 0.6280 | 0.6883 | 0.6280 | 0.6883 |
| 2 | 77 | 0.5635 | 0.7186 | 0.5593 | 0.7403 | 0.4897 | 0.7043 | 0.5211 | 0.7792 |
| 3 | 84 | 0.4045 | 0.5120 | 0.4718 | 0.6071 | 0.4218 | 0.5091 | 0.4517 | 0.5595 |
| 4 | 56 | 0.6995 | 0.7857 | 0.6995 | 0.7857 | 0.5204 | 0.5663 | 0.6172 | 0.5357 |

## Inner-CV model ranking (mean over folds)

| features | model | inner_macro_f1 |
|---|---|---|
| video | svc_rbf | 0.5904 |
| video | logreg_c01 | 0.5796 |
| video | svc_lin | 0.5646 |
| video | logreg | 0.5608 |
| physfeat+video | logreg | 0.5572 |
| physfeat+video | logreg_c01 | 0.5562 |
| physfeat+video | svc_lin | 0.5537 |
| physfeat+video | svc_rbf | 0.5515 |
| video | hgb | 0.5484 |
| video | extratrees | 0.5439 |
| physfeat | svc_lin | 0.5334 |
| video | rf | 0.5321 |
| physfeat | logreg | 0.5318 |
| physfeat | logreg_c01 | 0.5306 |
| physfeat+video | rf | 0.5213 |

## Per-fold selection

| fold | picked | inner_best |
|---|---|---|
| 0 | video/svc_rbf | physfeat+video/svc_rbf | physfeat/logreg | 0.5547 |
| 1 | video/svc_rbf | video/logreg_c01 | video/svc_lin | 0.6089 |
| 2 | video/rf | video/extratrees | video/logreg | 0.5877 |
| 3 | video/svc_rbf | physfeat+video/logreg | all/svc_lin | 0.6477 |
| 4 | video/svc_rbf | video/svc_lin | video/logreg_c01 | 0.5925 |

## Config

```json
{
  "subject_relative": true,
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
