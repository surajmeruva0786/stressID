# SOTA run — `s4_windows`

**Generated:** 2026-08-15 05:27:44  
**Primary (all700_macro_f1):** **0.7572**  
**Best previous:** `s2_views_relz` at 0.7517 → **NEW BEST** (+0.0054)  
**Duration:** 5062 s (84.4 min)  

> **Protocol.** Subject-shared repeated stratified CV — the split rule
> the StressID origin paper uses (random 80/20 + SMOTE). Subjects appear
> on both sides. Numbers here are comparable to published ones and are
> **not** comparable to the GroupKFold track in `reports/`, which holds
> subjects out. Origin paper's best reported weighted F1: 0.72.

## What changed

R4: adds window-level and GPU sequence candidates. Window candidates fit tree learners on ~9000 window rows instead of 560 recording rows and average the window probabilities back up; sequence candidates are a small masked GRU/attention encoder on the GPU. Recording-level candidates are unchanged and still present, so the new candidate types have to earn selection against them. Pruning at 0.90 retained from R3.

## Headline metrics

| metric | value |
|---|---|
| all700_macro_f1 | 0.7572 |
| all700_macro_f1_std | 0.0455 |
| all700_weighted_f1 | 0.7580 |
| all700_weighted_f1_std | 0.0456 |
| all700_balanced_acc | 0.7570 |
| all700_balanced_acc_std | 0.0450 |
| all700_accuracy | 0.7586 |
| all700_accuracy_std | 0.0458 |
| all700_roc_auc | 0.8228 |
| all700_roc_auc_std | 0.0490 |
| all700_single_macro_f1 | 0.7337 |
| all700_unpruned_macro_f1 | 0.7540 |
| all700_mean_members | 49.2000 |
| all700_mean_members_unpruned | 75.4000 |
| all700_n_eval_folds | 5 |
| all700_n_recordings | 700 |

## Per-fold — all700

| fold | n_test | thr | n_members | n_members_full | macro_f1 | weighted_f1 | balanced_acc | accuracy | roc_auc | unpruned_macro_f1 | unpruned_weighted_f1 | unpruned_balanced_acc | unpruned_accuracy | unpruned_roc_auc | single_macro_f1 | single_weighted_f1 | single_balanced_acc | single_accuracy | single_roc_auc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 140 | 0.5600 | 46 | 69 | 0.6994 | 0.7000 | 0.6994 | 0.7000 | 0.7422 | 0.6855 | 0.6858 | 0.6857 | 0.6857 | 0.7438 | 0.7064 | 0.7070 | 0.7063 | 0.7071 | 0.7281 |
| 1 | 140 | 0.5000 | 50 | 78 | 0.7571 | 0.7572 | 0.7579 | 0.7571 | 0.8197 | 0.7640 | 0.7643 | 0.7642 | 0.7643 | 0.8193 | 0.7356 | 0.7354 | 0.7380 | 0.7357 | 0.8144 |
| 2 | 140 | 0.5100 | 45 | 69 | 0.8268 | 0.8278 | 0.8256 | 0.8286 | 0.8733 | 0.8118 | 0.8131 | 0.8104 | 0.8143 | 0.8757 | 0.7974 | 0.7987 | 0.7961 | 0.8000 | 0.8654 |
| 3 | 140 | 0.5100 | 55 | 81 | 0.7494 | 0.7501 | 0.7496 | 0.7500 | 0.8403 | 0.7559 | 0.7569 | 0.7555 | 0.7571 | 0.8395 | 0.7266 | 0.7279 | 0.7260 | 0.7286 | 0.8138 |
| 4 | 140 | 0.4400 | 50 | 80 | 0.7531 | 0.7549 | 0.7523 | 0.7571 | 0.8387 | 0.7531 | 0.7549 | 0.7523 | 0.7571 | 0.8372 | 0.7024 | 0.7058 | 0.7052 | 0.7143 | 0.8139 |
| 0 |  | 0.5600 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 1 |  | 0.5000 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 2 |  | 0.5100 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 3 |  | 0.5100 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 4 |  | 0.4400 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## Inner-CV candidate ranking — all700

| candidate | inner_macro_f1 |
|---|---|
| all|rel|extratrees | 0.7503 |
| all|rel|lgbm | 0.7474 |
| all+avail|rel|extratrees | 0.7471 |
| all+avail|rel|lgbm | 0.7457 |
| all|rel|rf | 0.7434 |
| win-raw|mean|extratrees | 0.7416 |
| all+avail|rel|rf | 0.7412 |
| win-raw|trimmed|extratrees | 0.7409 |
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
  "window_views": [
    "raw",
    "rel"
  ],
  "torch_archs": [
    "gru",
    "attn"
  ],
  "inner_folds": 3,
  "ensemble_cum_keep": 0.9,
  "selection": "bagged greedy w/ replacement on inner OOF; threshold tuned on inner OOF"
}
```

## Environment

- Windows-10-10.0.26200-SP0
- Python 3.11.9
