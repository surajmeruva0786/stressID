# Run report — `c1_nested_ensemble`

**Generated:** 2026-08-03 14:15:22  
**Primary metric (complete364_macro_f1):** **0.5327**  
**Majority-class reference:** 0.4180 (margin +0.1147)  
**Best previous:** `none` at nan → **FIRST RUN** (+nan)  
**Duration:** 291 s (4.9 min)  

> Primary metric is macro F1 on the 364 all-modality recordings
> under subject GroupKFold. Availability is constant there, so the
> modality-availability shortcut carries no signal (see §10.5).

## What changed

Iteration 1. Deep model abandoned as primary: on complete-364 seven classical configs beat it (video+SVC 0.544 vs 0.485). Nested GroupKFold model selection (inner CV on train subjects only) over 7 feature sets x 7 models, soft-vote ensemble of inner top-3. No subject-relative features yet - this is the honest non-personalised baseline.

## Headline metrics

| metric | value |
|---|---|
| complete364_macro_f1 | 0.5327 |
| complete364_macro_f1_single | 0.4870 |
| complete364_weighted_f1 | 0.6429 |
| complete364_balanced_acc | 0.5470 |
| complete364_accuracy | 0.6793 |
| n_folds | 5 |
| subject_relative | False |

## Per-fold outer results

| fold | n_test | ens_macro_f1 | ens_weighted_f1 | ens_balanced_acc | ens_accuracy | single_macro_f1 | single_weighted_f1 | single_balanced_acc | single_accuracy |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 70 | 0.5139 | 0.6194 | 0.5238 | 0.6571 | 0.4562 | 0.5680 | 0.4694 | 0.6000 |
| 1 | 77 | 0.6980 | 0.7659 | 0.6771 | 0.7922 | 0.6404 | 0.6855 | 0.6562 | 0.6753 |
| 2 | 77 | 0.3955 | 0.5401 | 0.3882 | 0.5065 | 0.3947 | 0.5220 | 0.3926 | 0.4805 |
| 3 | 84 | 0.3957 | 0.5182 | 0.5000 | 0.6548 | 0.4305 | 0.5019 | 0.4408 | 0.5238 |
| 4 | 56 | 0.6606 | 0.7710 | 0.6458 | 0.7857 | 0.5134 | 0.5692 | 0.5903 | 0.5357 |

## Inner-CV model ranking (mean over folds)

| features | model | inner_macro_f1 |
|---|---|---|
| video | logreg_c01 | 0.5627 |
| video | svc_lin | 0.5432 |
| video | logreg | 0.5390 |
| video | svc_rbf | 0.5350 |
| video | hgb | 0.5279 |
| video | extratrees | 0.5147 |
| physfeat+video | logreg | 0.4940 |
| physfeat+video | svc_lin | 0.4933 |
| physfeat+video | svc_rbf | 0.4927 |
| all | svc_lin | 0.4889 |
| physfeat+video | hgb | 0.4871 |
| audio+video | logreg | 0.4867 |
| audio+video | hgb | 0.4848 |
| audio+video | logreg_c01 | 0.4836 |
| physfeat+audio+video | hgb | 0.4830 |

## Per-fold selection

| fold | picked | inner_best |
|---|---|---|
| 0 | video/hgb | video/logreg_c01 | video/svc_lin | 0.5974 |
| 1 | video/logreg_c01 | video/svc_lin | video/svc_rbf | 0.5685 |
| 2 | video/logreg | video/logreg_c01 | video/svc_lin | 0.5856 |
| 3 | audio/svc_lin | video/extratrees | audio+video/svc_lin | 0.6129 |
| 4 | physfeat+video/svc_lin | video/logreg_c01 | video/svc_rbf | 0.5567 |

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
