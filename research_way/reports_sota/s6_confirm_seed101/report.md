# SOTA run — `s6_confirm_seed101`

**Generated:** 2026-08-15 10:18:45  
**Primary (all700_macro_f1):** **0.7467**  
**Best previous:** `s5_final` at 0.7604 → **no improvement** (-0.0137)  
**Duration:** 5961 s (99.3 min)  

> **Protocol.** Subject-shared repeated stratified CV — the split rule
> the StressID origin paper uses (random 80/20 + SMOTE). Subjects appear
> on both sides. Numbers here are comparable to published ones and are
> **not** comparable to the GroupKFold track in `reports/`, which holds
> subjects out. Origin paper's best reported weighted F1: 0.72.

## What changed

R6 confirmation: the R5 configuration, frozen, re-run on outer folds the search never saw (seed 101 instead of 42). All five rounds selected against the seed-42 partition, so the campaign maximum over it is optimistically biased. This run changes nothing but the partition, and its number -- not R5 -- is the one that belongs in a paper.

## Headline metrics

| metric | value |
|---|---|
| all700_macro_f1 | 0.7467 |
| all700_macro_f1_std | 0.0456 |
| all700_weighted_f1 | 0.7479 |
| all700_weighted_f1_std | 0.0445 |
| all700_balanced_acc | 0.7478 |
| all700_balanced_acc_std | 0.0452 |
| all700_accuracy | 0.7500 |
| all700_accuracy_std | 0.0423 |
| all700_roc_auc | 0.8156 |
| all700_roc_auc_std | 0.0447 |
| all700_single_macro_f1 | 0.7288 |
| all700_unpruned_macro_f1 | 0.7495 |
| all700_mean_members | 57.2000 |
| all700_mean_members_unpruned | 86.6000 |
| all700_n_eval_folds | 5 |
| all700_n_recordings | 700 |
| c364_macro_f1 | 0.6721 |
| c364_macro_f1_std | 0.0362 |
| c364_weighted_f1 | 0.7358 |
| c364_weighted_f1_std | 0.0314 |
| c364_balanced_acc | 0.6710 |
| c364_balanced_acc_std | 0.0381 |
| c364_accuracy | 0.7389 |
| c364_accuracy_std | 0.0344 |
| c364_roc_auc | 0.7295 |
| c364_roc_auc_std | 0.0216 |
| c364_single_macro_f1 | 0.6782 |
| c364_unpruned_macro_f1 | 0.6813 |
| c364_mean_members | 41.6000 |
| c364_mean_members_unpruned | 70.6000 |
| c364_n_eval_folds | 5 |
| c364_n_recordings | 364 |

## Per-fold — all700

| fold | n_test | thr | n_members | n_members_full | macro_f1 | weighted_f1 | balanced_acc | accuracy | roc_auc | unpruned_macro_f1 | unpruned_weighted_f1 | unpruned_balanced_acc | unpruned_accuracy | unpruned_roc_auc | single_macro_f1 | single_weighted_f1 | single_balanced_acc | single_accuracy | single_roc_auc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 140 | 0.4900 | 52 | 82 | 0.7712 | 0.7715 | 0.7716 | 0.7714 | 0.8571 | 0.7783 | 0.7786 | 0.7785 | 0.7786 | 0.8567 | 0.7569 | 0.7572 | 0.7573 | 0.7571 | 0.8411 |
| 1 | 140 | 0.4600 | 42 | 69 | 0.7351 | 0.7370 | 0.7369 | 0.7429 | 0.7947 | 0.7351 | 0.7370 | 0.7369 | 0.7429 | 0.7962 | 0.7173 | 0.7187 | 0.7175 | 0.7214 | 0.7907 |
| 2 | 140 | 0.5100 | 70 | 100 | 0.7996 | 0.8001 | 0.8002 | 0.8000 | 0.8442 | 0.7996 | 0.8001 | 0.8002 | 0.8000 | 0.8438 | 0.8069 | 0.8073 | 0.8077 | 0.8071 | 0.8309 |
| 3 | 140 | 0.5000 | 59 | 89 | 0.6778 | 0.6807 | 0.6790 | 0.6857 | 0.7473 | 0.6778 | 0.6807 | 0.6790 | 0.6857 | 0.7469 | 0.6631 | 0.6661 | 0.6646 | 0.6714 | 0.7383 |
| 4 | 140 | 0.5100 | 63 | 93 | 0.7499 | 0.7502 | 0.7512 | 0.7500 | 0.8348 | 0.7569 | 0.7573 | 0.7580 | 0.7571 | 0.8321 | 0.6999 | 0.7002 | 0.7015 | 0.7000 | 0.7833 |

## Selected ensemble members — all700

| fold | thr | inner_blend_f1 | top_members | weight_recording | weight_window | weight_sequence |
|---|---|---|---|---|---|---|
| 0 | 0.4900 | 0.7717 | all+avail|rel|svc_rbf:0.08 | phys|rel|svc_rbf:0.07 | all|rel|svc_rbf:0.06 | phys+video|rel|svc_rbf:0.05 | all|raw|svc_rbf:0.04 | phys|raw|svc_rbf:0.04 | 0.9110 | 0.0810 | 0.0070 |
| 1 | 0.4600 | 0.7701 | all+avail|rel|rf:0.08 | phys|z|extratrees:0.06 | win-raw|trimmed|extratrees:0.06 | phys|z|rf:0.05 | phys|rel|rf:0.05 | phys+video|rel|extratrees:0.04 | 0.8080 | 0.1620 | 0.0300 |
| 2 | 0.5100 | 0.7635 | phys|rel|svc_rbf:0.07 | phys|rel|extratrees:0.06 | all|rel|xgb:0.06 | phys+video|rel|xgb:0.04 | audio+video|raw|logreg_l2w:0.03 | phys|z|extratrees:0.03 | 0.9890 | 0.0110 | 0 |
| 3 | 0.5000 | 0.7937 | all|rel|svc_rbf:0.08 | phys+audio|rel|extratrees:0.05 | all|rel|rf:0.04 | all+avail|rel|svc_rbf:0.04 | all+avail|rel|lgbm:0.03 | audio+video|rel|xgb:0.03 | 0.9740 | 0.0260 | 0 |
| 4 | 0.5100 | 0.7565 | phys+audio|rel|extratrees:0.14 | phys+audio|raw|extratrees:0.06 | all+avail|rel|svc_rbf:0.04 | all|raw|rf:0.03 | all+avail|raw|extratrees:0.03 | all|rel|svc_rbf:0.03 | 0.8960 | 0.1040 | 0 |

## Inner-CV candidate ranking — all700

| candidate | inner_macro_f1 |
|---|---|
| all+avail|rel|lgbm | 0.7439 |
| all+avail|rel|svc_rbf | 0.7420 |
| all|rel|rf | 0.7418 |
| all|rel|extratrees | 0.7418 |
| all|rel|lgbm | 0.7416 |
| all|rel|svc_rbf | 0.7406 |
| all+avail|rel|xgb | 0.7401 |
| all+avail|rel|rf | 0.7399 |
| phys+audio|rel|extratrees | 0.7395 |
| all+avail|rel|extratrees | 0.7390 |
| win-raw|mean|extratrees | 0.7385 |
| all|rel|xgb | 0.7383 |
| phys+video|rel|lgbm | 0.7371 |
| win-raw|trimmed|extratrees | 0.7371 |
| phys+audio|raw|extratrees | 0.7341 |
| phys+audio|rel|rf | 0.7316 |
| phys+audio|rel|lgbm | 0.7310 |
| phys+audio|rel|svc_rbf | 0.7309 |
| all+avail|raw|extratrees | 0.7284 |
| phys+video|rel|svc_rbf | 0.7282 |
| phys+video|rel|xgb | 0.7262 |
| all|raw|extratrees | 0.7242 |
| all+avail|raw|rf | 0.7236 |
| phys|rel|extratrees | 0.7230 |
| all|raw|lgbm | 0.7229 |

## Per-fold — c364

| fold | n_test | thr | n_members | n_members_full | macro_f1 | weighted_f1 | balanced_acc | accuracy | roc_auc | unpruned_macro_f1 | unpruned_weighted_f1 | unpruned_balanced_acc | unpruned_accuracy | unpruned_roc_auc | single_macro_f1 | single_weighted_f1 | single_balanced_acc | single_accuracy | single_roc_auc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 73 | 0.5400 | 50 | 80 | 0.7181 | 0.7845 | 0.7028 | 0.7945 | 0.7368 | 0.7181 | 0.7845 | 0.7028 | 0.7945 | 0.7387 | 0.7245 | 0.7808 | 0.7245 | 0.7808 | 0.7660 |
| 1 | 73 | 0.6100 | 47 | 77 | 0.6326 | 0.7053 | 0.6277 | 0.7123 | 0.6914 | 0.6326 | 0.7053 | 0.6277 | 0.7123 | 0.6923 | 0.6676 | 0.7334 | 0.6612 | 0.7397 | 0.6419 |
| 2 | 73 | 0.4900 | 45 | 75 | 0.6561 | 0.7281 | 0.6470 | 0.7397 | 0.7353 | 0.6561 | 0.7281 | 0.6470 | 0.7397 | 0.7353 | 0.6422 | 0.7025 | 0.6465 | 0.6986 | 0.7372 |
| 3 | 73 | 0.5600 | 31 | 58 | 0.7019 | 0.7470 | 0.7179 | 0.7397 | 0.7454 | 0.7019 | 0.7470 | 0.7179 | 0.7397 | 0.7454 | 0.6806 | 0.7512 | 0.6662 | 0.7671 | 0.7179 |
| 4 | 72 | 0.5800 | 35 | 63 | 0.6519 | 0.7142 | 0.6596 | 0.7083 | 0.7385 | 0.6975 | 0.7535 | 0.7038 | 0.7500 | 0.7394 | 0.6761 | 0.7380 | 0.6788 | 0.7361 | 0.7144 |

## Selected ensemble members — c364

| fold | thr | inner_blend_f1 | top_members | weight_recording | weight_window | weight_sequence |
|---|---|---|---|---|---|---|
| 0 | 0.5400 | 0.6894 | audio|z|logreg_l2w:0.09 | video|rel|logreg:0.07 | video|rel|logreg_l2w:0.07 | seq-raw|attn|torch:0.07 | video|raw|logreg:0.06 | seq-rel|attn|torch:0.04 | 0.8300 | 0.0070 | 0.1630 |
| 1 | 0.6100 | 0.6890 | video|rel|logreg_l2w:0.18 | video|rel|svc_rbf:0.16 | video|rel|logreg:0.10 | audio+video|rel|logreg_l2w:0.03 | audio+video|raw|svc_rbf:0.03 | seq-raw|gru|torch:0.02 | 0.9300 | 0 | 0.0700 |
| 2 | 0.4900 | 0.7196 | seq-rel|gru|torch:0.13 | seq-raw|gru|torch:0.12 | audio+video|raw|logreg_l2w:0.08 | seq-raw|attn|torch:0.06 | video|raw|logreg_l2w:0.05 | seq-rel|attn|torch:0.04 | 0.6410 | 0 | 0.3590 |
| 3 | 0.5600 | 0.7358 | win-raw|trimmed|xgb:0.13 | audio|raw|logreg_l2w:0.11 | audio|z|logreg_l2w:0.10 | video|rel|logreg_l2w:0.09 | seq-rel|gru|torch:0.06 | seq-rel|attn|torch:0.04 | 0.6570 | 0.2210 | 0.1220 |
| 4 | 0.5800 | 0.6809 | audio+video|rel|logreg_l2w:0.21 | seq-rel|gru|torch:0.12 | seq-rel|attn|torch:0.09 | video|rel|logreg:0.07 | video|rel|logreg_l2w:0.06 | video|raw|logreg:0.05 | 0.7080 | 0.0700 | 0.2210 |

## Inner-CV candidate ranking — c364

| candidate | inner_macro_f1 |
|---|---|
| video|rel|logreg_l2w | 0.6857 |
| video|rel|logreg | 0.6765 |
| audio+video|rel|logreg | 0.6696 |
| audio+video|rel|logreg_l2w | 0.6682 |
| win-raw|trimmed|xgb | 0.6667 |
| win-raw|trimmed|lgbm | 0.6645 |
| phys+video|rel|svc_rbf | 0.6632 |
| win-raw|mean|lgbm | 0.6632 |
| all|rel|svc_rbf | 0.6620 |
| all+avail|rel|svc_rbf | 0.6620 |
| audio+video|raw|logreg | 0.6601 |
| win-raw|mean|xgb | 0.6601 |
| audio+video|raw|logreg_l2w | 0.6599 |
| video|raw|logreg | 0.6516 |
| phys+video|rel|logreg | 0.6502 |
| phys+video|rel|logreg_l2w | 0.6502 |
| audio+video|rel|svc_rbf | 0.6493 |
| all|raw|logreg_l2w | 0.6479 |
| all+avail|raw|logreg_l2w | 0.6479 |
| video|raw|logreg_l2w | 0.6459 |
| all|raw|svc_rbf | 0.6454 |
| all+avail|raw|svc_rbf | 0.6454 |
| phys+video|raw|logreg_l2w | 0.6448 |
| audio+video|raw|lgbm | 0.6441 |
| audio|raw|lgbm | 0.6434 |

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
