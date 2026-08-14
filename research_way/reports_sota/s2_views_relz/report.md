# SOTA run — `s2_views_relz`

**Generated:** 2026-08-15 02:48:38  
**Primary (all700_macro_f1):** **0.7517**  
**Best previous:** `s1_baseline_raw` at 0.7419 → **NEW BEST** (+0.0099)  
**Duration:** 5915 s (98.6 min)  

> **Protocol.** Subject-shared repeated stratified CV — the split rule
> the StressID origin paper uses (random 80/20 + SMOTE). Subjects appear
> on both sides. Numbers here are comparable to published ones and are
> **not** comparable to the GroupKFold track in `reports/`, which holds
> subjects out. Origin paper's best reported weighted F1: 0.72.

## What changed

R2: adds the subject-referenced views. Same features, same models, same protocol as R1 -- the only change is that each feature block is also offered centred on the participant own low-stress baseline (rel) and z-scored within participant (z). all700 only, to isolate the view effect before spending a round on both scopes.

## Headline metrics

| metric | value |
|---|---|
| all700_macro_f1 | 0.7517 |
| all700_macro_f1_std | 0.0324 |
| all700_weighted_f1 | 0.7525 |
| all700_weighted_f1_std | 0.0327 |
| all700_balanced_acc | 0.7517 |
| all700_balanced_acc_std | 0.0319 |
| all700_accuracy | 0.7529 |
| all700_accuracy_std | 0.0334 |
| all700_roc_auc | 0.8133 |
| all700_roc_auc_std | 0.0498 |
| all700_single_macro_f1 | 0.7337 |
| all700_n_eval_folds | 5 |
| all700_n_recordings | 700 |

## Per-fold — all700

| fold | n_test | thr | n_members | macro_f1 | weighted_f1 | balanced_acc | accuracy | roc_auc | single_macro_f1 | single_weighted_f1 | single_balanced_acc | single_accuracy | single_roc_auc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 140 | 0.5100 | 63 | 0.7064 | 0.7070 | 0.7063 | 0.7071 | 0.7354 | 0.7064 | 0.7070 | 0.7063 | 0.7071 | 0.7281 |
| 1 | 140 | 0.4600 | 87 | 0.7563 | 0.7569 | 0.7561 | 0.7571 | 0.8033 | 0.7356 | 0.7354 | 0.7380 | 0.7357 | 0.8144 |
| 2 | 140 | 0.4900 | 76 | 0.7966 | 0.7981 | 0.7952 | 0.8000 | 0.8706 | 0.7974 | 0.7987 | 0.7961 | 0.8000 | 0.8654 |
| 3 | 140 | 0.5100 | 80 | 0.7426 | 0.7431 | 0.7437 | 0.7429 | 0.8268 | 0.7266 | 0.7279 | 0.7260 | 0.7286 | 0.7922 |
| 4 | 140 | 0.5200 | 92 | 0.7567 | 0.7573 | 0.7572 | 0.7571 | 0.8305 | 0.7024 | 0.7058 | 0.7052 | 0.7143 | 0.8139 |
| 0 |  | 0.5100 |  |  |  |  |  |  |  |  |  |  |  |
| 1 |  | 0.4600 |  |  |  |  |  |  |  |  |  |  |  |
| 2 |  | 0.4900 |  |  |  |  |  |  |  |  |  |  |  |
| 3 |  | 0.5100 |  |  |  |  |  |  |  |  |  |  |  |
| 4 |  | 0.5200 |  |  |  |  |  |  |  |  |  |  |  |

## Inner-CV candidate ranking — all700

| candidate | inner_macro_f1 |
|---|---|
| all|rel|extratrees | 0.7503 |
| all|rel|lgbm | 0.7474 |
| all+avail|rel|extratrees | 0.7471 |
| all+avail|rel|lgbm | 0.7457 |
| all|rel|rf | 0.7434 |
| all+avail|rel|rf | 0.7412 |
| all+avail|rel|svc_rbf | 0.7402 |
| all|rel|svc_rbf | 0.7400 |
| phys+audio|rel|extratrees | 0.7382 |
| all|rel|xgb | 0.7377 |
| all+avail|rel|xgb | 0.7370 |
| phys+video|rel|extratrees | 0.7368 |
| phys+video|rel|lgbm | 0.7367 |
| phys+audio|rel|lgbm | 0.7356 |
| phys+video|rel|svc_rbf | 0.7343 |
| phys+video|rel|xgb | 0.7338 |
| all+avail|raw|extratrees | 0.7336 |
| all|raw|extratrees | 0.7331 |
| all+avail|raw|lgbm | 0.7329 |
| phys+audio|rel|svc_rbf | 0.7313 |
| phys+video|rel|rf | 0.7301 |
| all|raw|rf | 0.7297 |
| phys+audio|raw|extratrees | 0.7293 |
| all+avail|raw|rf | 0.7280 |
| audio+video|rel|svc_rbf | 0.7274 |

## Config

```json
{
  "protocol": "subject-shared RepeatedStratifiedKFold (paper-style)",
  "n_folds": 5,
  "repeats": 1,
  "seed": 42,
  "views": [
    "raw",
    "rel",
    "z"
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
    "all700"
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
