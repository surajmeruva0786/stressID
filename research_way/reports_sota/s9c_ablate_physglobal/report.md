# SOTA run — `s9c_ablate_physglobal`

**Generated:** 2026-08-16 00:03:21  
**Primary (all700_macro_f1):** **0.7484**  
**Best previous:** `s5_final` at 0.7604 → **no improvement** (-0.0120)  
**Duration:** 9903 s (165.1 min)  

> **Protocol.** Subject-shared repeated stratified CV — the split rule
> the StressID origin paper uses (random 80/20 + SMOTE). Subjects appear
> on both sides. Numbers here are comparable to published ones and are
> **not** comparable to the GroupKFold track in `reports/`, which holds
> subjects out. Origin paper's best reported weighted F1: 0.72.

## What changed

R9c: the physglobal ABLATION arm. Byte-identical to R9b -- same seed, same folds, same feature-set names, same thread budget -- except the physglobal block is dropped from every feature set. Exists because the earlier R9-vs-R8a comparison differed in thread count as well as features, and thread count alone changes booster predictions.

## Headline metrics

| metric | value |
|---|---|
| all700_macro_f1 | 0.7484 |
| all700_macro_f1_std | 0.0331 |
| all700_weighted_f1 | 0.7494 |
| all700_weighted_f1_std | 0.0324 |
| all700_balanced_acc | 0.7490 |
| all700_balanced_acc_std | 0.0331 |
| all700_accuracy | 0.7505 |
| all700_accuracy_std | 0.0313 |
| all700_roc_auc | 0.8194 |
| all700_roc_auc_std | 0.0377 |
| all700_single_macro_f1 | 0.7401 |
| all700_unpruned_macro_f1 | 0.7440 |
| all700_mean_members | 54.7333 |
| all700_mean_members_unpruned | 84.0667 |
| all700_n_eval_folds | 15 |
| all700_n_recordings | 700 |

## Per-fold — all700

| fold | n_test | thr | n_members | n_members_full | macro_f1 | weighted_f1 | balanced_acc | accuracy | roc_auc | unpruned_macro_f1 | unpruned_weighted_f1 | unpruned_balanced_acc | unpruned_accuracy | unpruned_roc_auc | single_macro_f1 | single_weighted_f1 | single_balanced_acc | single_accuracy | single_roc_auc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 140 | 0.4900 | 52 | 82 | 0.7712 | 0.7715 | 0.7716 | 0.7714 | 0.8571 | 0.7783 | 0.7786 | 0.7785 | 0.7786 | 0.8567 | 0.7569 | 0.7572 | 0.7573 | 0.7571 | 0.8411 |
| 1 | 140 | 0.4600 | 42 | 69 | 0.7351 | 0.7370 | 0.7369 | 0.7429 | 0.7947 | 0.7351 | 0.7370 | 0.7369 | 0.7429 | 0.7962 | 0.7173 | 0.7187 | 0.7175 | 0.7214 | 0.7907 |
| 2 | 140 | 0.5100 | 70 | 100 | 0.7996 | 0.8001 | 0.8002 | 0.8000 | 0.8442 | 0.7996 | 0.8001 | 0.8002 | 0.8000 | 0.8438 | 0.8069 | 0.8073 | 0.8077 | 0.8071 | 0.8309 |
| 3 | 140 | 0.5000 | 59 | 89 | 0.6778 | 0.6807 | 0.6790 | 0.6857 | 0.7473 | 0.6778 | 0.6807 | 0.6790 | 0.6857 | 0.7469 | 0.6631 | 0.6661 | 0.6646 | 0.6714 | 0.7383 |
| 4 | 140 | 0.5100 | 63 | 93 | 0.7499 | 0.7502 | 0.7512 | 0.7500 | 0.8348 | 0.7569 | 0.7573 | 0.7580 | 0.7571 | 0.8321 | 0.6999 | 0.7002 | 0.7015 | 0.7000 | 0.7833 |
| 5 | 140 | 0.5100 | 45 | 75 | 0.7426 | 0.7430 | 0.7430 | 0.7429 | 0.8113 | 0.7497 | 0.7501 | 0.7498 | 0.7500 | 0.8113 | 0.7559 | 0.7566 | 0.7555 | 0.7571 | 0.7883 |
| 6 | 140 | 0.5100 | 56 | 86 | 0.7710 | 0.7714 | 0.7710 | 0.7714 | 0.8303 | 0.7710 | 0.7714 | 0.7710 | 0.7714 | 0.8293 | 0.7707 | 0.7712 | 0.7704 | 0.7714 | 0.8141 |
| 7 | 140 | 0.5200 | 62 | 92 | 0.7341 | 0.7353 | 0.7336 | 0.7357 | 0.7740 | 0.7258 | 0.7274 | 0.7252 | 0.7286 | 0.7725 | 0.7182 | 0.7199 | 0.7176 | 0.7214 | 0.7545 |
| 8 | 140 | 0.4800 | 52 | 80 | 0.7571 | 0.7573 | 0.7588 | 0.7571 | 0.8452 | 0.7428 | 0.7426 | 0.7461 | 0.7429 | 0.8485 | 0.7286 | 0.7286 | 0.7310 | 0.7286 | 0.8347 |
| 9 | 140 | 0.4900 | 48 | 77 | 0.7296 | 0.7319 | 0.7295 | 0.7357 | 0.8305 | 0.7296 | 0.7319 | 0.7295 | 0.7357 | 0.8278 | 0.7296 | 0.7319 | 0.7295 | 0.7357 | 0.8137 |
| 10 | 140 | 0.5400 | 61 | 91 | 0.8000 | 0.8000 | 0.8015 | 0.8000 | 0.8743 | 0.7710 | 0.7714 | 0.7710 | 0.7714 | 0.8743 | 0.8067 | 0.8071 | 0.8065 | 0.8071 | 0.8301 |
| 11 | 140 | 0.5100 | 48 | 75 | 0.7197 | 0.7206 | 0.7194 | 0.7214 | 0.7964 | 0.7162 | 0.7179 | 0.7169 | 0.7214 | 0.7964 | 0.6893 | 0.6907 | 0.6895 | 0.6929 | 0.7769 |
| 12 | 140 | 0.5100 | 47 | 77 | 0.7211 | 0.7216 | 0.7217 | 0.7214 | 0.8262 | 0.7211 | 0.7216 | 0.7217 | 0.7214 | 0.8270 | 0.7203 | 0.7213 | 0.7201 | 0.7214 | 0.8190 |
| 13 | 140 | 0.4800 | 52 | 81 | 0.7850 | 0.7857 | 0.7850 | 0.7857 | 0.8630 | 0.7920 | 0.7928 | 0.7918 | 0.7929 | 0.8634 | 0.8129 | 0.8138 | 0.8120 | 0.8143 | 0.8387 |
| 14 | 140 | 0.4900 | 64 | 94 | 0.7326 | 0.7343 | 0.7320 | 0.7357 | 0.7619 | 0.6925 | 0.6931 | 0.6931 | 0.6929 | 0.7635 | 0.7258 | 0.7274 | 0.7252 | 0.7286 | 0.7709 |

## Selected ensemble members — all700

| fold | thr | inner_blend_f1 | top_members | weight_recording | weight_window | weight_sequence |
|---|---|---|---|---|---|---|
| 0 | 0.4900 | 0.7717 | all+avail|rel|svc_rbf:0.08 | phys|rel|svc_rbf:0.07 | all|rel|svc_rbf:0.06 | phys+video|rel|svc_rbf:0.05 | all|raw|svc_rbf:0.04 | phys|raw|svc_rbf:0.04 | 0.9110 | 0.0810 | 0.0070 |
| 1 | 0.4600 | 0.7701 | all+avail|rel|rf:0.08 | phys|z|extratrees:0.06 | win-raw|trimmed|extratrees:0.06 | phys|z|rf:0.05 | phys|rel|rf:0.05 | phys+video|rel|extratrees:0.04 | 0.8080 | 0.1620 | 0.0300 |
| 2 | 0.5100 | 0.7635 | phys|rel|svc_rbf:0.07 | phys|rel|extratrees:0.06 | all|rel|xgb:0.06 | phys+video|rel|xgb:0.04 | audio+video|raw|logreg_l2w:0.03 | phys|z|extratrees:0.03 | 0.9890 | 0.0110 | 0 |
| 3 | 0.5000 | 0.7937 | all|rel|svc_rbf:0.08 | phys+audio|rel|extratrees:0.05 | all|rel|rf:0.04 | all+avail|rel|svc_rbf:0.04 | all+avail|rel|lgbm:0.03 | audio+video|rel|xgb:0.03 | 0.9740 | 0.0260 | 0 |
| 4 | 0.5100 | 0.7565 | phys+audio|rel|extratrees:0.14 | phys+audio|raw|extratrees:0.06 | all+avail|rel|svc_rbf:0.04 | all|raw|rf:0.03 | all+avail|raw|extratrees:0.03 | all|rel|svc_rbf:0.03 | 0.8960 | 0.1040 | 0 |
| 5 | 0.5100 | 0.7649 | win-raw|mean|extratrees:0.11 | win-raw|trimmed|extratrees:0.11 | all|rel|extratrees:0.07 | all+avail|raw|extratrees:0.04 | all|raw|extratrees:0.03 | phys+video|rel|extratrees:0.03 | 0.7370 | 0.2630 | 0 |
| 6 | 0.5100 | 0.7675 | audio+video|raw|lgbm:0.07 | all+avail|rel|lgbm:0.05 | all+avail|rel|rf:0.04 | phys+video|raw|mlp:0.04 | phys+video|rel|extratrees:0.04 | video|rel|lgbm:0.03 | 0.9560 | 0.0260 | 0.0190 |
| 7 | 0.5200 | 0.7745 | phys+video|raw|extratrees:0.07 | win-raw|mean|extratrees:0.06 | phys|raw|extratrees:0.05 | all|raw|xgb:0.04 | audio+video|rel|extratrees:0.04 | audio+video|raw|extratrees:0.03 | 0.8560 | 0.1370 | 0.0070 |
| 8 | 0.4800 | 0.7742 | all+avail|rel|extratrees:0.08 | audio+video|rel|xgb:0.08 | all|rel|svc_rbf:0.07 | video|raw|rf:0.06 | all+avail|rel|svc_rbf:0.05 | video|rel|extratrees:0.04 | 0.9850 | 0.0150 | 0 |
| 9 | 0.4900 | 0.7700 | all|rel|rf:0.14 | all+avail|rel|extratrees:0.09 | phys+audio|rel|rf:0.04 | phys|raw|extratrees:0.04 | win-raw|mean|extratrees:0.04 | all+avail|rel|rf:0.03 | 0.8930 | 0.1070 | 0 |
| 10 | 0.5400 | 0.7601 | video|raw|extratrees:0.05 | video|rel|extratrees:0.04 | all+avail|rel|xgb:0.04 | phys|rel|svc_rbf:0.04 | phys+video|rel|extratrees:0.04 | all+avail|rel|lgbm:0.04 | 0.9070 | 0.0930 | 0 |
| 11 | 0.5100 | 0.7777 | phys|rel|extratrees:0.11 | phys+audio|rel|rf:0.06 | seq-raw|attn|torch:0.06 | phys|z|svc_rbf:0.04 | win-raw|trimmed|xgb:0.04 | all|rel|extratrees:0.04 | 0.8340 | 0.0960 | 0.0700 |
| 12 | 0.5100 | 0.7733 | all+avail|rel|extratrees:0.06 | win-raw|trimmed|extratrees:0.05 | win-raw|mean|extratrees:0.04 | win-raw|mean|lgbm:0.04 | phys+video|rel|extratrees:0.04 | all+avail|rel|rf:0.04 | 0.7590 | 0.2220 | 0.0190 |
| 13 | 0.4800 | 0.7555 | all|rel|extratrees:0.08 | phys+audio|rel|extratrees:0.05 | video|rel|lgbm:0.05 | all|rel|rf:0.04 | audio+video|rel|lgbm:0.04 | all+avail|rel|rf:0.04 | 0.8970 | 0.0960 | 0.0070 |
| 14 | 0.4900 | 0.7721 | all|rel|lgbm:0.11 | phys|rel|mlp:0.08 | phys+video|rel|svc_rbf:0.05 | phys+video|raw|rf:0.04 | phys+audio|rel|extratrees:0.04 | video|rel|extratrees:0.03 | 0.9740 | 0.0220 | 0.0040 |

## Inner-CV candidate ranking — all700

| candidate | inner_macro_f1 |
|---|---|
| all+avail|rel|extratrees | 0.7438 |
| all|rel|extratrees | 0.7433 |
| win-raw|mean|extratrees | 0.7426 |
| all+avail|rel|rf | 0.7419 |
| all+avail|rel|lgbm | 0.7415 |
| win-raw|trimmed|extratrees | 0.7410 |
| all|rel|rf | 0.7408 |
| phys+audio|rel|extratrees | 0.7403 |
| all|rel|lgbm | 0.7399 |
| all+avail|rel|svc_rbf | 0.7384 |
| all|rel|svc_rbf | 0.7372 |
| all+avail|rel|xgb | 0.7371 |
| phys+video|rel|lgbm | 0.7349 |
| all|rel|xgb | 0.7348 |
| all+avail|raw|extratrees | 0.7305 |
| phys+audio|rel|rf | 0.7303 |
| phys+audio|raw|extratrees | 0.7299 |
| all|raw|extratrees | 0.7293 |
| phys+video|rel|svc_rbf | 0.7280 |
| phys+audio|rel|lgbm | 0.7280 |
| all|raw|lgbm | 0.7273 |
| win-raw|trimmed|lgbm | 0.7264 |
| phys+video|rel|extratrees | 0.7260 |
| all|raw|rf | 0.7258 |
| phys+audio|rel|svc_rbf | 0.7257 |

## Config

```json
{
  "protocol": "subject-shared RepeatedStratifiedKFold (paper-style)",
  "n_folds": 5,
  "repeats": 3,
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
  "n_par": 3,
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
  "exclude_blocks": [
    "physglobal"
  ],
  "selection": "bagged greedy w/ replacement on inner OOF; threshold tuned on inner OOF"
}
```

## Environment

- Windows-10-10.0.26200-SP0
- Python 3.11.9
