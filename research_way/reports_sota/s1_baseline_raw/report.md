# SOTA run — `s1_baseline_raw`

**Generated:** 2026-08-15 01:09:00  
**Primary (all700_macro_f1):** **0.7419**  
**Best previous:** `none` at nan → **FIRST RUN** (+nan)  
**Duration:** 3594 s (59.9 min)  

> **Protocol.** Subject-shared repeated stratified CV — the split rule
> the StressID origin paper uses (random 80/20 + SMOTE). Subjects appear
> on both sides. Numbers here are comparable to published ones and are
> **not** comparable to the GroupKFold track in `reports/`, which holds
> subjects out. Origin paper's best reported weighted F1: 0.72.

## What changed

R1 baseline: raw view only, no subject-referenced normalisation. 8 feature sets x 8 models = 64 candidates, bagged greedy ensemble + inner-OOF threshold. Serial sweep, 3 inner folds. The number every later round must beat.

## Headline metrics

| metric | value |
|---|---|
| all700_macro_f1 | 0.7419 |
| all700_macro_f1_std | 0.0338 |
| all700_weighted_f1 | 0.7426 |
| all700_weighted_f1_std | 0.0338 |
| all700_balanced_acc | 0.7420 |
| all700_balanced_acc_std | 0.0339 |
| all700_accuracy | 0.7429 |
| all700_accuracy_std | 0.0339 |
| all700_roc_auc | 0.8091 |
| all700_roc_auc_std | 0.0558 |
| all700_single_macro_f1 | 0.7429 |
| all700_n_eval_folds | 5 |
| all700_n_recordings | 700 |
| c364_macro_f1 | 0.6518 |
| c364_macro_f1_std | 0.0608 |
| c364_weighted_f1 | 0.7302 |
| c364_weighted_f1_std | 0.0478 |
| c364_balanced_acc | 0.6444 |
| c364_balanced_acc_std | 0.0567 |
| c364_accuracy | 0.7473 |
| c364_accuracy_std | 0.0526 |
| c364_roc_auc | 0.6999 |
| c364_roc_auc_std | 0.0367 |
| c364_single_macro_f1 | 0.6481 |
| c364_n_eval_folds | 5 |
| c364_n_recordings | 364 |

## Per-fold — all700

| fold | n_test | thr | n_members | macro_f1 | weighted_f1 | balanced_acc | accuracy | roc_auc | single_macro_f1 | single_weighted_f1 | single_balanced_acc | single_accuracy | single_roc_auc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 140 | 0.5400 | 38 | 0.6851 | 0.6857 | 0.6851 | 0.6857 | 0.7228 | 0.7203 | 0.7210 | 0.7200 | 0.7214 | 0.7229 |
| 1 | 140 | 0.5200 | 53 | 0.7500 | 0.7499 | 0.7517 | 0.7500 | 0.8027 | 0.7285 | 0.7287 | 0.7293 | 0.7286 | 0.7919 |
| 2 | 140 | 0.5000 | 46 | 0.7703 | 0.7712 | 0.7699 | 0.7714 | 0.8741 | 0.7916 | 0.7925 | 0.7910 | 0.7929 | 0.8554 |
| 3 | 140 | 0.4800 | 37 | 0.7637 | 0.7644 | 0.7639 | 0.7643 | 0.8096 | 0.7266 | 0.7279 | 0.7260 | 0.7286 | 0.7922 |
| 4 | 140 | 0.4800 | 51 | 0.7403 | 0.7417 | 0.7396 | 0.7429 | 0.8364 | 0.7478 | 0.7492 | 0.7471 | 0.7500 | 0.8098 |
| 0 |  | 0.5400 |  |  |  |  |  |  |  |  |  |  |  |
| 1 |  | 0.5200 |  |  |  |  |  |  |  |  |  |  |  |
| 2 |  | 0.5000 |  |  |  |  |  |  |  |  |  |  |  |
| 3 |  | 0.4800 |  |  |  |  |  |  |  |  |  |  |  |
| 4 |  | 0.4800 |  |  |  |  |  |  |  |  |  |  |  |

## Inner-CV candidate ranking — all700

| candidate | inner_macro_f1 |
|---|---|
| all+avail|raw|extratrees | 0.7336 |
| all|raw|extratrees | 0.7331 |
| all+avail|raw|lgbm | 0.7329 |
| all|raw|rf | 0.7297 |
| phys+audio|raw|extratrees | 0.7293 |
| all+avail|raw|rf | 0.7280 |
| all|raw|lgbm | 0.7247 |
| audio+video|raw|extratrees | 0.7241 |
| all+avail|raw|xgb | 0.7210 |
| audio+video|raw|rf | 0.7206 |
| phys+audio|raw|rf | 0.7200 |
| audio+video|raw|lgbm | 0.7192 |
| all|raw|xgb | 0.7181 |
| audio|raw|extratrees | 0.7163 |
| audio+video|raw|svc_rbf | 0.7157 |
| phys+audio|raw|lgbm | 0.7156 |
| all|raw|svc_rbf | 0.7148 |
| all+avail|raw|svc_rbf | 0.7148 |
| audio+video|raw|xgb | 0.7131 |
| audio|raw|rf | 0.7128 |
| audio|raw|lgbm | 0.7111 |
| phys+video|raw|lgbm | 0.7105 |
| audio|raw|logreg_l2w | 0.7094 |
| all|raw|mlp | 0.7085 |
| phys+video|raw|extratrees | 0.7083 |

## Per-fold — c364

| fold | n_test | thr | n_members | macro_f1 | weighted_f1 | balanced_acc | accuracy | roc_auc | single_macro_f1 | single_weighted_f1 | single_balanced_acc | single_accuracy | single_roc_auc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 73 | 0.5200 | 44 | 0.6166 | 0.7092 | 0.6090 | 0.7260 | 0.7009 | 0.6323 | 0.7029 | 0.6368 | 0.6986 | 0.7028 |
| 1 | 73 | 0.4500 | 45 | 0.6256 | 0.7280 | 0.6190 | 0.7808 | 0.6731 | 0.6644 | 0.7483 | 0.6474 | 0.7808 | 0.6612 |
| 2 | 73 | 0.6200 | 44 | 0.7510 | 0.8017 | 0.7376 | 0.8082 | 0.7436 | 0.7376 | 0.7895 | 0.7280 | 0.7945 | 0.7372 |
| 3 | 73 | 0.6700 | 43 | 0.5989 | 0.6712 | 0.5989 | 0.6712 | 0.6548 | 0.6054 | 0.6926 | 0.5994 | 0.7123 | 0.6822 |
| 4 | 72 | 0.6600 | 49 | 0.6667 | 0.7407 | 0.6577 | 0.7500 | 0.7269 | 0.6010 | 0.7094 | 0.5962 | 0.7500 | 0.7510 |
| 0 |  | 0.5200 |  |  |  |  |  |  |  |  |  |  |  |
| 1 |  | 0.4500 |  |  |  |  |  |  |  |  |  |  |  |
| 2 |  | 0.6200 |  |  |  |  |  |  |  |  |  |  |  |
| 3 |  | 0.6700 |  |  |  |  |  |  |  |  |  |  |  |
| 4 |  | 0.6600 |  |  |  |  |  |  |  |  |  |  |  |

## Inner-CV candidate ranking — c364

| candidate | inner_macro_f1 |
|---|---|
| video|raw|lgbm | 0.6581 |
| audio+video|raw|logreg_l2w | 0.6558 |
| audio+video|raw|lgbm | 0.6550 |
| all|raw|svc_rbf | 0.6492 |
| all+avail|raw|svc_rbf | 0.6492 |
| video|raw|logreg_l2w | 0.6491 |
| audio+video|raw|logreg | 0.6489 |
| video|raw|logreg | 0.6489 |
| phys+video|raw|lgbm | 0.6429 |
| all|raw|logreg_l2w | 0.6397 |
| all+avail|raw|logreg_l2w | 0.6397 |
| audio|raw|lgbm | 0.6381 |
| phys+video|raw|svc_rbf | 0.6371 |
| all|raw|lgbm | 0.6353 |
| all+avail|raw|lgbm | 0.6353 |
| video|raw|svc_rbf | 0.6348 |
| audio+video|raw|svc_rbf | 0.6343 |
| phys+video|raw|logreg_l2w | 0.6340 |
| all|raw|logreg | 0.6332 |
| all+avail|raw|logreg | 0.6332 |
| audio+video|raw|xgb | 0.6289 |
| video|raw|xgb | 0.6269 |
| all+avail|raw|mlp | 0.6266 |
| all|raw|xgb | 0.6235 |
| all+avail|raw|xgb | 0.6219 |

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
    "logreg_l2w",
    "svc_rbf",
    "rf",
    "extratrees",
    "mlp",
    "lgbm",
    "xgb"
  ],
  "greedy_max_size": 25,
  "greedy_bags": 12,
  "feature_version": 3,
  "scopes": [
    "all700",
    "c364"
  ],
  "fast": true,
  "n_par": 1,
  "jobs": 6,
  "window_views": [],
  "torch_archs": [],
  "inner_folds": 3,
  "selection": "bagged greedy w/ replacement on inner OOF; threshold tuned on inner OOF"
}
```

## Environment

- Windows-10-10.0.26200-SP0
- Python 3.11.9
