# SOTA run — `s7_baseline_seed101`

**Generated:** 2026-08-15 10:49:23  
**Primary (all700_macro_f1):** **0.7272**  
**Best previous:** `s5_final` at 0.7604 → **no improvement** (-0.0332)  
**Duration:** 1821 s (30.3 min)  

> **Protocol.** Subject-shared repeated stratified CV — the split rule
> the StressID origin paper uses (random 80/20 + SMOTE). Subjects appear
> on both sides. Numbers here are comparable to published ones and are
> **not** comparable to the GroupKFold track in `reports/`, which holds
> subjects out. Origin paper's best reported weighted F1: 0.72.

## What changed

R7: the R1 BASELINE configuration re-run on the same unseen seed-101 partitions as R6. R6 alone cannot establish that the campaign improved anything, because 0.7467 on seed 101 and 0.7419 on seed 42 are different partitions. This run supplies the missing term so the improvement can be quoted as final-minus-baseline on identical folds.

## Headline metrics

| metric | value |
|---|---|
| all700_macro_f1 | 0.7272 |
| all700_macro_f1_std | 0.0248 |
| all700_weighted_f1 | 0.7283 |
| all700_weighted_f1_std | 0.0239 |
| all700_balanced_acc | 0.7283 |
| all700_balanced_acc_std | 0.0246 |
| all700_accuracy | 0.7300 |
| all700_accuracy_std | 0.0222 |
| all700_roc_auc | 0.7965 |
| all700_roc_auc_std | 0.0421 |
| all700_single_macro_f1 | 0.6899 |
| all700_unpruned_macro_f1 | 0.7272 |
| all700_mean_members | 44.4000 |
| all700_mean_members_unpruned | 44.4000 |
| all700_n_eval_folds | 5 |
| all700_n_recordings | 700 |
| c364_macro_f1 | 0.6605 |
| c364_macro_f1_std | 0.0337 |
| c364_weighted_f1 | 0.7276 |
| c364_weighted_f1_std | 0.0352 |
| c364_balanced_acc | 0.6585 |
| c364_balanced_acc_std | 0.0281 |
| c364_accuracy | 0.7333 |
| c364_accuracy_std | 0.0470 |
| c364_roc_auc | 0.7435 |
| c364_roc_auc_std | 0.0330 |
| c364_single_macro_f1 | 0.6362 |
| c364_unpruned_macro_f1 | 0.6605 |
| c364_mean_members | 47.6000 |
| c364_mean_members_unpruned | 47.6000 |
| c364_n_eval_folds | 5 |
| c364_n_recordings | 364 |

## Per-fold — all700

| fold | n_test | thr | n_members | n_members_full | macro_f1 | weighted_f1 | balanced_acc | accuracy | roc_auc | unpruned_macro_f1 | unpruned_weighted_f1 | unpruned_balanced_acc | unpruned_accuracy | unpruned_roc_auc | single_macro_f1 | single_weighted_f1 | single_balanced_acc | single_accuracy | single_roc_auc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 140 | 0.5100 | 45 | 45 | 0.7424 | 0.7429 | 0.7424 | 0.7429 | 0.8467 | 0.7424 | 0.7429 | 0.7424 | 0.7429 | 0.8467 | 0.7277 | 0.7283 | 0.7275 | 0.7286 | 0.8315 |
| 1 | 140 | 0.4700 | 51 | 51 | 0.7453 | 0.7468 | 0.7456 | 0.7500 | 0.7870 | 0.7453 | 0.7468 | 0.7456 | 0.7500 | 0.7870 | 0.6893 | 0.6907 | 0.6895 | 0.6929 | 0.7557 |
| 2 | 140 | 0.5400 | 40 | 40 | 0.7357 | 0.7356 | 0.7385 | 0.7357 | 0.8092 | 0.7357 | 0.7356 | 0.7385 | 0.7357 | 0.8092 | 0.6417 | 0.6405 | 0.6482 | 0.6429 | 0.7126 |
| 3 | 140 | 0.5100 | 49 | 49 | 0.6843 | 0.6873 | 0.6857 | 0.6929 | 0.7318 | 0.6843 | 0.6873 | 0.6857 | 0.6929 | 0.7318 | 0.6909 | 0.6939 | 0.6925 | 0.7000 | 0.7310 |
| 4 | 140 | 0.4800 | 37 | 37 | 0.7283 | 0.7288 | 0.7293 | 0.7286 | 0.8077 | 0.7283 | 0.7288 | 0.7293 | 0.7286 | 0.8077 | 0.6999 | 0.7002 | 0.7015 | 0.7000 | 0.7833 |

## Selected ensemble members — all700

| fold | thr | inner_blend_f1 | top_members | weight_recording | weight_window | weight_sequence |
|---|---|---|---|---|---|---|
| 0 | 0.5100 | 0.7419 | all+avail|raw|xgb:0.20 | audio+video|raw|rf:0.05 | video|raw|rf:0.05 | all|raw|lgbm:0.05 | all|raw|rf:0.05 | phys+video|raw|rf:0.04 | 1.0000 | 0 | 0 |
| 1 | 0.4700 | 0.7379 | video|raw|extratrees:0.08 | phys+audio|raw|rf:0.07 | phys+video|raw|rf:0.05 | phys+audio|raw|extratrees:0.05 | phys|raw|rf:0.04 | phys+audio|raw|svc_rbf:0.04 | 1.0000 | 0 | 0 |
| 2 | 0.5400 | 0.7393 | audio+video|raw|logreg_l2w:0.13 | phys+audio|raw|svc_rbf:0.10 | phys+audio|raw|mlp:0.06 | phys|raw|rf:0.05 | phys+audio|raw|extratrees:0.05 | all+avail|raw|svc_rbf:0.04 | 1.0000 | 0 | 0 |
| 3 | 0.5100 | 0.7709 | phys+video|raw|extratrees:0.09 | video|raw|extratrees:0.08 | phys|raw|rf:0.07 | phys+audio|raw|extratrees:0.05 | phys+video|raw|rf:0.05 | all|raw|xgb:0.04 | 1.0000 | 0 | 0 |
| 4 | 0.4800 | 0.7504 | all+avail|raw|rf:0.15 | phys+audio|raw|extratrees:0.14 | phys+audio|raw|rf:0.08 | all|raw|extratrees:0.07 | all+avail|raw|lgbm:0.05 | all+avail|raw|extratrees:0.05 | 1.0000 | 0 | 0 |

## Inner-CV candidate ranking — all700

| candidate | inner_macro_f1 |
|---|---|
| phys+audio|raw|extratrees | 0.7341 |
| all+avail|raw|extratrees | 0.7284 |
| all|raw|extratrees | 0.7242 |
| all+avail|raw|rf | 0.7236 |
| all|raw|lgbm | 0.7229 |
| all|raw|rf | 0.7229 |
| all+avail|raw|lgbm | 0.7189 |
| all|raw|svc_rbf | 0.7187 |
| phys+audio|raw|rf | 0.7186 |
| all+avail|raw|svc_rbf | 0.7183 |
| audio+video|raw|extratrees | 0.7165 |
| all|raw|xgb | 0.7162 |
| all+avail|raw|xgb | 0.7142 |
| audio+video|raw|svc_rbf | 0.7134 |
| phys+audio|raw|lgbm | 0.7124 |
| audio|raw|rf | 0.7121 |
| audio+video|raw|rf | 0.7119 |
| audio|raw|lgbm | 0.7118 |
| audio|raw|logreg_l2w | 0.7117 |
| audio|raw|xgb | 0.7094 |
| audio+video|raw|lgbm | 0.7090 |
| audio|raw|extratrees | 0.7074 |
| audio|raw|svc_rbf | 0.7069 |
| phys+audio|raw|svc_rbf | 0.7059 |
| phys+audio|raw|xgb | 0.7047 |

## Per-fold — c364

| fold | n_test | thr | n_members | n_members_full | macro_f1 | weighted_f1 | balanced_acc | accuracy | roc_auc | unpruned_macro_f1 | unpruned_weighted_f1 | unpruned_balanced_acc | unpruned_accuracy | unpruned_roc_auc | single_macro_f1 | single_weighted_f1 | single_balanced_acc | single_accuracy | single_roc_auc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 73 | 0.6100 | 54 | 54 | 0.6806 | 0.7557 | 0.6684 | 0.7671 | 0.7840 | 0.6806 | 0.7557 | 0.6684 | 0.7671 | 0.7840 | 0.6557 | 0.7260 | 0.6557 | 0.7260 | 0.7038 |
| 1 | 73 | 0.5100 | 42 | 42 | 0.6519 | 0.7369 | 0.6378 | 0.7671 | 0.7015 | 0.6519 | 0.7369 | 0.6378 | 0.7671 | 0.7015 | 0.6398 | 0.7257 | 0.6282 | 0.7534 | 0.6777 |
| 2 | 73 | 0.6100 | 50 | 50 | 0.6868 | 0.7415 | 0.6896 | 0.7397 | 0.7656 | 0.6868 | 0.7415 | 0.6896 | 0.7397 | 0.7656 | 0.6422 | 0.7025 | 0.6465 | 0.6986 | 0.7372 |
| 3 | 73 | 0.5300 | 45 | 45 | 0.6778 | 0.7378 | 0.6754 | 0.7397 | 0.7445 | 0.6778 | 0.7378 | 0.6754 | 0.7397 | 0.7445 | 0.6798 | 0.7562 | 0.6616 | 0.7808 | 0.7628 |
| 4 | 72 | 0.6700 | 47 | 47 | 0.6052 | 0.6661 | 0.6212 | 0.6528 | 0.7221 | 0.6052 | 0.6661 | 0.6212 | 0.6528 | 0.7221 | 0.5636 | 0.6364 | 0.5712 | 0.6250 | 0.6760 |

## Selected ensemble members — c364

| fold | thr | inner_blend_f1 | top_members | weight_recording | weight_window | weight_sequence |
|---|---|---|---|---|---|---|
| 0 | 0.6100 | 0.6935 | video|raw|logreg:0.07 | video|raw|mlp:0.07 | all+avail|raw|logreg_l2w:0.05 | video|raw|logreg_l2w:0.05 | phys+video|raw|logreg_l2w:0.05 | audio+video|raw|logreg_l2w:0.04 | 1.0000 | 0 | 0 |
| 1 | 0.5100 | 0.6856 | video|raw|svc_rbf:0.13 | audio+video|raw|svc_rbf:0.13 | all|raw|svc_rbf:0.12 | audio|raw|logreg_l2w:0.08 | all+avail|raw|svc_rbf:0.05 | all|raw|logreg_l2w:0.05 | 1.0000 | 0 | 0 |
| 2 | 0.6100 | 0.7078 | audio+video|raw|logreg_l2w:0.21 | audio+video|raw|logreg:0.11 | video|raw|logreg:0.10 | audio+video|raw|mlp:0.04 | video|raw|mlp:0.04 | all+avail|raw|svc_rbf:0.04 | 1.0000 | 0 | 0 |
| 3 | 0.5300 | 0.7062 | audio+video|raw|logreg_l2w:0.12 | audio|raw|logreg_l2w:0.12 | audio|raw|extratrees:0.11 | audio|raw|logreg:0.08 | video|raw|lgbm:0.05 | audio+video|raw|logreg:0.04 | 1.0000 | 0 | 0 |
| 4 | 0.6700 | 0.6756 | audio+video|raw|logreg:0.22 | video|raw|logreg:0.12 | video|raw|svc_rbf:0.06 | audio+video|raw|logreg_l2w:0.05 | audio|raw|logreg_l2w:0.05 | video|raw|xgb:0.04 | 1.0000 | 0 | 0 |

## Inner-CV candidate ranking — c364

| candidate | inner_macro_f1 |
|---|---|
| audio+video|raw|logreg | 0.6601 |
| audio+video|raw|logreg_l2w | 0.6599 |
| video|raw|logreg | 0.6516 |
| all|raw|logreg_l2w | 0.6479 |
| all+avail|raw|logreg_l2w | 0.6479 |
| video|raw|logreg_l2w | 0.6459 |
| all|raw|svc_rbf | 0.6454 |
| all+avail|raw|svc_rbf | 0.6454 |
| phys+video|raw|logreg_l2w | 0.6448 |
| audio+video|raw|lgbm | 0.6441 |
| audio|raw|lgbm | 0.6434 |
| phys+video|raw|logreg | 0.6404 |
| phys+video|raw|svc_rbf | 0.6386 |
| video|raw|lgbm | 0.6383 |
| video|raw|svc_rbf | 0.6374 |
| all|raw|logreg | 0.6365 |
| all+avail|raw|logreg | 0.6365 |
| audio+video|raw|svc_rbf | 0.6331 |
| all|raw|lgbm | 0.6320 |
| all+avail|raw|lgbm | 0.6320 |
| video|raw|xgb | 0.6295 |
| audio+video|raw|xgb | 0.6286 |
| phys+video|raw|xgb | 0.6253 |
| phys+video|raw|lgbm | 0.6243 |
| video|raw|mlp | 0.6187 |

## Config

```json
{
  "protocol": "subject-shared RepeatedStratifiedKFold (paper-style)",
  "n_folds": 5,
  "repeats": 1,
  "seed": 101,
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
  "ensemble_cum_keep": 1.0,
  "selection": "bagged greedy w/ replacement on inner OOF; threshold tuned on inner OOF"
}
```

## Environment

- Windows-10-10.0.26200-SP0
- Python 3.11.9
