# SOTA run — `t1_affect3`

**Generated:** 2026-08-15 18:51:15  
**Primary (all700_macro_f1):** **nan**  
**Best previous:** `s5_final` at 0.7604 → **no improvement** (+nan)  
**Duration:** 12361 s (206.0 min)  

> **Protocol.** Subject-shared repeated stratified CV — the split rule
> the StressID origin paper uses (random 80/20 + SMOTE). Subjects appear
> on both sides. Numbers here are comparable to published ones and are
> **not** comparable to the GroupKFold track in `reports/`, which holds
> subjects out. Origin paper's best reported weighted F1: 0.72.

## What changed

T1: 3-class affect (negative / neutral / positive-arousal), 253/202/245. Same recipe as the binary campaign - raw+rel views, window-level candidates, bagged greedy on inner OOF, 90% pruning. Greedy maximises macro F1 over the argmax.

## Headline metrics

| metric | value |
|---|---|
| affect3_macro_f1 | 0.6214 |
| affect3_macro_f1_std | 0.0142 |
| affect3_weighted_f1 | 0.6334 |
| affect3_weighted_f1_std | 0.0141 |
| affect3_balanced_acc | 0.6290 |
| affect3_balanced_acc_std | 0.0154 |
| affect3_accuracy | 0.6471 |
| affect3_accuracy_std | 0.0179 |
| affect3_single_macro_f1 | 0.6141 |
| affect3_n_folds | 5 |
| affect3_n_recordings | 700 |

## Per-fold — affect3

| fold | n_test | n_members | macro_f1 | weighted_f1 | balanced_acc | accuracy | single_macro_f1 | single_weighted_f1 | single_balanced_acc | single_accuracy |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 140 | 42 | 0.6349 | 0.6465 | 0.6402 | 0.6571 | 0.6308 | 0.6387 | 0.6320 | 0.6429 |
| 1 | 140 | 37 | 0.6246 | 0.6407 | 0.6398 | 0.6643 | 0.6110 | 0.6264 | 0.6217 | 0.6429 |
| 2 | 140 | 35 | 0.6275 | 0.6401 | 0.6361 | 0.6571 | 0.6269 | 0.6337 | 0.6287 | 0.6357 |
| 3 | 140 | 45 | 0.6227 | 0.6283 | 0.6252 | 0.6357 | 0.6148 | 0.6205 | 0.6149 | 0.6214 |
| 4 | 140 | 42 | 0.5974 | 0.6111 | 0.6038 | 0.6214 | 0.5867 | 0.5979 | 0.5866 | 0.6000 |

## Selected members — affect3

| fold | n_members | top_members | weight_window |
|---|---|---|---|
| 0 | 42 | phys|raw|extratrees:0.07 | phys+video|raw|svc_rbf:0.07 | phys+video|rel|xgb:0.07 | phys+video|rel|lgbm:0.07 | phys+video|rel|extratrees:0.06 | 0.0300 |
| 1 | 37 | phys|raw|extratrees:0.07 | phys|raw|svc_rbf:0.07 | all+avail|rel|lgbm:0.06 | phys|rel|svc_rbf:0.06 | phys+video|rel|rf:0.05 | 0.0590 |
| 2 | 35 | phys+audio|rel|svc_rbf:0.11 | phys|rel|svc_rbf:0.08 | win-raw|mean|xgb:0.07 | win-raw|mean|lgbm:0.07 | phys|raw|svc_rbf:0.05 | 0.1330 |
| 3 | 45 | win-raw|mean|xgb:0.08 | phys|raw|rf:0.06 | phys+video|rel|svc_rbf:0.06 | phys+video|rel|extratrees:0.06 | audio|raw|logreg:0.05 | 0.1110 |
| 4 | 42 | win-raw|mean|extratrees:0.16 | phys+video|rel|svc_rbf:0.08 | video|rel|lgbm:0.07 | win-raw|mean|xgb:0.06 | video|rel|rf:0.05 | 0.2360 |

## Inner-CV ranking — affect3

| candidate | inner_score |
|---|---|
| win-raw|mean|xgb | 0.5921 |
| win-raw|mean|lgbm | 0.5878 |
| all|rel|lgbm | 0.5813 |
| win-raw|mean|extratrees | 0.5803 |
| all+avail|rel|lgbm | 0.5793 |
| phys+video|rel|svc_rbf | 0.5765 |
| all|rel|xgb | 0.5762 |
| all+avail|rel|xgb | 0.5762 |
| all|raw|lgbm | 0.5760 |
| phys+audio|rel|extratrees | 0.5749 |
| phys+video|rel|lgbm | 0.5721 |
| phys+video|rel|xgb | 0.5713 |
| all+avail|raw|lgbm | 0.5693 |
| phys+audio|raw|extratrees | 0.5683 |
| all|rel|svc_rbf | 0.5674 |
| all|raw|xgb | 0.5673 |
| phys+audio|rel|lgbm | 0.5661 |
| all+avail|rel|svc_rbf | 0.5661 |
| phys+audio|rel|svc_rbf | 0.5660 |
| all+avail|raw|xgb | 0.5645 |

## Config

```json
{
  "task": "affect3",
  "protocol": "subject-shared repeated K-fold (paper-style)",
  "n_folds": 5,
  "repeats": 1,
  "seed": 101,
  "views": [
    "raw",
    "rel"
  ],
  "window_views": [
    "raw",
    "rel"
  ],
  "models": [
    "logreg",
    "svc_rbf",
    "rf",
    "extratrees",
    "mlp",
    "lgbm",
    "xgb"
  ],
  "cum_keep": 0.9,
  "inner_folds": 3,
  "feature_version": 3,
  "selection": "bagged greedy w/ replacement on inner OOF"
}
```

## Environment

- Windows-10-10.0.26200-SP0
- Python 3.11.9
