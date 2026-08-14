# SOTA run — `s3_prune`

**Generated:** 2026-08-15 03:58:09  
**Primary (all700_macro_f1):** **0.7505**  
**Best previous:** `s2_views_relz` at 0.7517 → **no improvement** (-0.0012)  
**Duration:** 4044 s (67.4 min)  

> **Protocol.** Subject-shared repeated stratified CV — the split rule
> the StressID origin paper uses (random 80/20 + SMOTE). Subjects appear
> on both sides. Numbers here are comparable to published ones and are
> **not** comparable to the GroupKFold track in `reports/`, which holds
> subjects out. Origin paper's best reported weighted F1: 0.72.

## What changed

R3: ensemble pruning. Members covering only 90 percent of cumulative greedy weight are kept; the UNPRUNED blend is scored in the same run on the same folds, so the pruning effect is attributable without comparing across runs. The z view is dropped on R2 evidence (zero of the top 25 inner candidates used it), which also returns a third of the sweep budget.

## Headline metrics

| metric | value |
|---|---|
| all700_macro_f1 | 0.7505 |
| all700_macro_f1_std | 0.0353 |
| all700_weighted_f1 | 0.7512 |
| all700_weighted_f1_std | 0.0355 |
| all700_balanced_acc | 0.7507 |
| all700_balanced_acc_std | 0.0348 |
| all700_accuracy | 0.7514 |
| all700_accuracy_std | 0.0359 |
| all700_roc_auc | 0.8154 |
| all700_roc_auc_std | 0.0504 |
| all700_single_macro_f1 | 0.7337 |
| all700_unpruned_macro_f1 | 0.7464 |
| all700_mean_members | 40.4000 |
| all700_mean_members_unpruned | 66.0000 |
| all700_n_eval_folds | 5 |
| all700_n_recordings | 700 |

## Per-fold — all700

| fold | n_test | thr | n_members | n_members_full | macro_f1 | weighted_f1 | balanced_acc | accuracy | roc_auc | unpruned_macro_f1 | unpruned_weighted_f1 | unpruned_balanced_acc | unpruned_accuracy | unpruned_roc_auc | single_macro_f1 | single_weighted_f1 | single_balanced_acc | single_accuracy | single_roc_auc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 140 | 0.5100 | 38 | 63 | 0.7133 | 0.7141 | 0.7131 | 0.7143 | 0.7350 | 0.6784 | 0.6787 | 0.6789 | 0.6786 | 0.7377 | 0.7064 | 0.7070 | 0.7063 | 0.7071 | 0.7281 |
| 1 | 140 | 0.4800 | 35 | 61 | 0.7642 | 0.7644 | 0.7648 | 0.7643 | 0.8060 | 0.7642 | 0.7644 | 0.7648 | 0.7643 | 0.8074 | 0.7356 | 0.7354 | 0.7380 | 0.7357 | 0.8144 |
| 2 | 140 | 0.5000 | 34 | 61 | 0.8049 | 0.8061 | 0.8036 | 0.8071 | 0.8698 | 0.8118 | 0.8131 | 0.8104 | 0.8143 | 0.8716 | 0.7974 | 0.7987 | 0.7961 | 0.8000 | 0.8654 |
| 3 | 140 | 0.5400 | 45 | 70 | 0.7357 | 0.7358 | 0.7377 | 0.7357 | 0.8329 | 0.7428 | 0.7430 | 0.7445 | 0.7429 | 0.8331 | 0.7266 | 0.7279 | 0.7260 | 0.7286 | 0.7922 |
| 4 | 140 | 0.5100 | 50 | 75 | 0.7346 | 0.7356 | 0.7344 | 0.7357 | 0.8335 | 0.7346 | 0.7356 | 0.7344 | 0.7357 | 0.8329 | 0.7024 | 0.7058 | 0.7052 | 0.7143 | 0.8139 |
| 0 |  | 0.5100 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 1 |  | 0.4800 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 2 |  | 0.5000 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 3 |  | 0.5400 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 4 |  | 0.5100 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

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
    "rel"
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
  "ensemble_cum_keep": 0.9,
  "selection": "bagged greedy w/ replacement on inner OOF; threshold tuned on inner OOF"
}
```

## Environment

- Windows-10-10.0.26200-SP0
- Python 3.11.9
