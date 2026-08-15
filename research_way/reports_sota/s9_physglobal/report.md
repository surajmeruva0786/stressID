# SOTA run — `s9_physglobal`

**Generated:** 2026-08-15 18:42:22  
**Primary (all700_macro_f1):** **0.7542**  
**Best previous:** `s5_final` at 0.7604 → **no improvement** (-0.0062)  
**Duration:** 10085 s (168.1 min)  

> **Protocol.** Subject-shared repeated stratified CV — the split rule
> the StressID origin paper uses (random 80/20 + SMOTE). Subjects appear
> on both sides. Numbers here are comparable to published ones and are
> **not** comparable to the GroupKFold track in `reports/`, which holds
> subjects out. Origin paper's best reported weighted F1: 0.72.

## What changed

R9: adds the physglobal block (frequency-domain HRV LF/HF/LF-HF, whole-recording time-domain HRV, EDA tonic-phasic, respiration) to the three physio-bearing feature sets. Everything else identical to R8a including the seed, so folds pair exactly. Duration-dependent counts were removed from this block first -- duration alone scores 0.5923 macro F1 on StressID and is a task label in disguise.

## Headline metrics

| metric | value |
|---|---|
| all700_macro_f1 | 0.7542 |
| all700_macro_f1_std | 0.0434 |
| all700_weighted_f1 | 0.7553 |
| all700_weighted_f1_std | 0.0424 |
| all700_balanced_acc | 0.7552 |
| all700_balanced_acc_std | 0.0430 |
| all700_accuracy | 0.7571 |
| all700_accuracy_std | 0.0401 |
| all700_roc_auc | 0.8155 |
| all700_roc_auc_std | 0.0452 |
| all700_single_macro_f1 | 0.7245 |
| all700_unpruned_macro_f1 | 0.7472 |
| all700_mean_members | 56.4000 |
| all700_mean_members_unpruned | 85.0000 |
| all700_n_eval_folds | 5 |
| all700_n_recordings | 700 |

## Per-fold — all700

| fold | n_test | thr | n_members | n_members_full | macro_f1 | weighted_f1 | balanced_acc | accuracy | roc_auc | unpruned_macro_f1 | unpruned_weighted_f1 | unpruned_balanced_acc | unpruned_accuracy | unpruned_roc_auc | single_macro_f1 | single_weighted_f1 | single_balanced_acc | single_accuracy | single_roc_auc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 140 | 0.4700 | 58 | 87 | 0.7780 | 0.7785 | 0.7779 | 0.7786 | 0.8530 | 0.7780 | 0.7785 | 0.7779 | 0.7786 | 0.8542 | 0.7281 | 0.7286 | 0.7281 | 0.7286 | 0.8323 |
| 1 | 140 | 0.5000 | 42 | 67 | 0.7521 | 0.7536 | 0.7524 | 0.7571 | 0.7970 | 0.7375 | 0.7391 | 0.7381 | 0.7429 | 0.7970 | 0.7114 | 0.7126 | 0.7113 | 0.7143 | 0.7856 |
| 2 | 140 | 0.5000 | 56 | 85 | 0.7996 | 0.8001 | 0.8002 | 0.8000 | 0.8464 | 0.7996 | 0.8001 | 0.8002 | 0.8000 | 0.8473 | 0.7780 | 0.7787 | 0.7783 | 0.7786 | 0.8315 |
| 3 | 140 | 0.4900 | 60 | 90 | 0.6843 | 0.6873 | 0.6857 | 0.6929 | 0.7447 | 0.6778 | 0.6807 | 0.6790 | 0.6857 | 0.7457 | 0.6550 | 0.6582 | 0.6570 | 0.6643 | 0.7460 |
| 4 | 140 | 0.5100 | 66 | 96 | 0.7571 | 0.7571 | 0.7596 | 0.7571 | 0.8364 | 0.7429 | 0.7429 | 0.7453 | 0.7429 | 0.8362 | 0.7500 | 0.7501 | 0.7520 | 0.7500 | 0.8279 |

## Selected ensemble members — all700

| fold | thr | inner_blend_f1 | top_members | weight_recording | weight_window | weight_sequence |
|---|---|---|---|---|---|---|
| 0 | 0.4700 | 0.7567 | all+avail|rel|lgbm:0.10 | all|rel|svc_rbf:0.06 | all+avail|rel|svc_rbf:0.04 | all|raw|svc_rbf:0.03 | all+avail|rel|rf:0.03 | audio+video|rel|xgb:0.03 | 0.9410 | 0.0590 | 0 |
| 1 | 0.5000 | 0.7697 | phys|rel|extratrees:0.08 | phys+video|rel|extratrees:0.07 | phys+audio|rel|extratrees:0.06 | all|rel|extratrees:0.06 | phys|z|rf:0.05 | win-raw|trimmed|extratrees:0.05 | 0.8600 | 0.1330 | 0.0070 |
| 2 | 0.5000 | 0.7634 | all+avail|rel|svc_rbf:0.04 | phys+audio|rel|svc_rbf:0.04 | phys|rel|svc_rbf:0.04 | all+avail|rel|lgbm:0.04 | phys+video|rel|rf:0.04 | all+avail|rel|extratrees:0.04 | 1.0000 | 0 | 0 |
| 3 | 0.4900 | 0.7972 | all|rel|rf:0.07 | all|rel|svc_rbf:0.06 | all+avail|rel|svc_rbf:0.06 | all+avail|rel|rf:0.04 | phys|rel|extratrees:0.04 | phys+audio|rel|extratrees:0.04 | 0.9260 | 0.0740 | 0 |
| 4 | 0.5100 | 0.7585 | phys+audio|rel|extratrees:0.15 | all|raw|rf:0.03 | phys+video|rel|svc_rbf:0.03 | phys+video|raw|xgb:0.03 | phys+audio|raw|svc_rbf:0.03 | win-rel|mean|lgbm:0.03 | 0.8560 | 0.1410 | 0.0040 |

## Inner-CV candidate ranking — all700

| candidate | inner_macro_f1 |
|---|---|
| all|rel|lgbm | 0.7464 |
| all+avail|rel|lgbm | 0.7455 |
| all|rel|extratrees | 0.7441 |
| all+avail|rel|rf | 0.7439 |
| phys+audio|rel|extratrees | 0.7434 |
| all|rel|rf | 0.7427 |
| all|rel|svc_rbf | 0.7414 |
| all+avail|rel|svc_rbf | 0.7407 |
| win-raw|mean|extratrees | 0.7385 |
| all+avail|rel|extratrees | 0.7379 |
| win-raw|trimmed|extratrees | 0.7371 |
| all+avail|rel|xgb | 0.7371 |
| phys+audio|rel|lgbm | 0.7358 |
| all|rel|xgb | 0.7349 |
| phys+audio|rel|rf | 0.7347 |
| phys+audio|rel|svc_rbf | 0.7333 |
| phys+video|rel|lgbm | 0.7321 |
| phys+audio|raw|extratrees | 0.7317 |
| phys+video|rel|svc_rbf | 0.7315 |
| phys+video|rel|extratrees | 0.7296 |
| phys|rel|extratrees | 0.7287 |
| phys+video|rel|xgb | 0.7287 |
| phys+audio|rel|xgb | 0.7284 |
| all|raw|extratrees | 0.7268 |
| phys+video|rel|rf | 0.7241 |

## Config

```json
{
  "protocol": "subject-shared RepeatedStratifiedKFold (paper-style)",
  "n_folds": 5,
  "repeats": 1,
  "seed": 101,
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
    "all+avail",
    "physglobal",
    "phys_window_only"
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
  "jobs": 4,
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
