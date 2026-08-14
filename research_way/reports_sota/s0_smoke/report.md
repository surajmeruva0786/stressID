# SOTA run — `s0_smoke`

**Generated:** 2026-08-14 23:43:24  
**Primary (all700_macro_f1):** **nan**  
**Best previous:** `none` at nan → **FIRST RUN** (+nan)  
**Duration:** 255 s (4.2 min)  

> **Protocol.** Subject-shared repeated stratified CV — the split rule
> the StressID origin paper uses (random 80/20 + SMOTE). Subjects appear
> on both sides. Numbers here are comparable to published ones and are
> **not** comparable to the GroupKFold track in `reports/`, which holds
> subjects out. Origin paper's best reported weighted F1: 0.72.

## What changed

smoke: env-based GPU detect, no torch in workers

## Headline metrics

| metric | value |
|---|---|
| c364_macro_f1 | 0.6723 |
| c364_macro_f1_std | 0.0623 |
| c364_weighted_f1 | 0.7474 |
| c364_weighted_f1_std | 0.0478 |
| c364_balanced_acc | 0.6587 |
| c364_balanced_acc_std | 0.0542 |
| c364_accuracy | 0.7637 |
| c364_accuracy_std | 0.0488 |
| c364_roc_auc | 0.7359 |
| c364_roc_auc_std | 0.0476 |
| c364_single_macro_f1 | 0.6530 |
| c364_n_eval_folds | 5 |
| c364_n_recordings | 364 |

## Per-fold — c364

| fold | n_test | thr | n_members | macro_f1 | weighted_f1 | balanced_acc | accuracy | roc_auc | single_macro_f1 | single_weighted_f1 | single_balanced_acc | single_accuracy | single_roc_auc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 73 | 0.6900 | 5 | 0.6087 | 0.6935 | 0.6057 | 0.6986 | 0.7142 | 0.6087 | 0.6935 | 0.6057 | 0.6986 | 0.6651 |
| 1 | 73 | 0.4600 | 6 | 0.6398 | 0.7257 | 0.6282 | 0.7534 | 0.7921 | 0.6868 | 0.7415 | 0.6896 | 0.7397 | 0.7610 |
| 2 | 73 | 0.5500 | 8 | 0.7700 | 0.8221 | 0.7427 | 0.8356 | 0.7720 | 0.7376 | 0.7895 | 0.7280 | 0.7945 | 0.7372 |
| 3 | 73 | 0.6200 | 7 | 0.6923 | 0.7567 | 0.6804 | 0.7671 | 0.6722 | 0.5868 | 0.6661 | 0.5847 | 0.6712 | 0.6062 |
| 4 | 72 | 0.5200 | 5 | 0.6506 | 0.7390 | 0.6365 | 0.7639 | 0.7288 | 0.6453 | 0.7417 | 0.6308 | 0.7778 | 0.8067 |
| 0 |  | 0.6900 |  |  |  |  |  |  |  |  |  |  |  |
| 1 |  | 0.4600 |  |  |  |  |  |  |  |  |  |  |  |
| 2 |  | 0.5500 |  |  |  |  |  |  |  |  |  |  |  |
| 3 |  | 0.6200 |  |  |  |  |  |  |  |  |  |  |  |
| 4 |  | 0.5200 |  |  |  |  |  |  |  |  |  |  |  |

## Inner-CV candidate ranking — c364

| candidate | inner_macro_f1 |
|---|---|
| video|raw|logreg | 0.6516 |
| video|raw|lgbm | 0.6492 |
| audio|raw|lgbm | 0.6377 |
| video|raw|xgb | 0.6281 |
| audio|raw|xgb | 0.6243 |
| audio|raw|logreg | 0.5853 |
| audio|raw|extratrees | 0.5843 |
| phys|raw|lgbm | 0.5632 |
| video|raw|extratrees | 0.5593 |
| phys|raw|logreg | 0.5428 |
| phys|raw|xgb | 0.5242 |
| phys|raw|extratrees | 0.5066 |

## Config

```json
{
  "protocol": "subject-shared RepeatedStratifiedKFold (paper-style)",
  "n_folds": 5,
  "repeats": 1,
  "seed": 42,
  "views": [
    "raw"
  ],
  "feature_sets": [
    "phys",
    "audio",
    "video",
    "audio+video",
    "phys+video",
    "phys+audio",
    "all",
    "all+avail"
  ],
  "models": [
    "logreg",
    "extratrees",
    "lgbm",
    "xgb"
  ],
  "greedy_max_size": 5,
  "greedy_bags": 3,
  "feature_version": 3,
  "scopes": [
    "c364"
  ],
  "fast": true,
  "n_par": 3,
  "jobs": 6,
  "selection": "bagged greedy w/ replacement on inner OOF; threshold tuned on inner OOF"
}
```

## Environment

- Windows-10-10.0.26200-SP0
- Python 3.11.9
