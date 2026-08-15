# SOTA run — `s9b_physglobal_r3`

**Generated:** 2026-08-15 23:34:57  
**Primary (all700_macro_f1):** **0.7507**  
**Best previous:** `s5_final` at 0.7604 → **no improvement** (-0.0097)  
**Duration:** 10196 s (169.9 min)  

> **Protocol.** Subject-shared repeated stratified CV — the split rule
> the StressID origin paper uses (random 80/20 + SMOTE). Subjects appear
> on both sides. Numbers here are comparable to published ones and are
> **not** comparable to the GroupKFold track in `reports/`, which holds
> subjects out. Origin paper's best reported weighted F1: 0.72.

## What changed

R9b: physglobal at 3 repeats (15 folds), pairing fold-for-fold against s8a_final_r3. Settles whether frequency-domain HRV helps, after the 5-fold read came in at +0.0075 with p=0.051 and no fold worse.

## Headline metrics

| metric | value |
|---|---|
| all700_macro_f1 | 0.7507 |
| all700_macro_f1_std | 0.0341 |
| all700_weighted_f1 | 0.7517 |
| all700_weighted_f1_std | 0.0335 |
| all700_balanced_acc | 0.7511 |
| all700_balanced_acc_std | 0.0341 |
| all700_accuracy | 0.7529 |
| all700_accuracy_std | 0.0323 |
| all700_roc_auc | 0.8191 |
| all700_roc_auc_std | 0.0380 |
| all700_single_macro_f1 | 0.7353 |
| all700_unpruned_macro_f1 | 0.7512 |
| all700_mean_members | 53.0000 |
| all700_mean_members_unpruned | 81.3333 |
| all700_n_eval_folds | 15 |
| all700_n_recordings | 700 |

## Per-fold — all700

| fold | n_test | thr | n_members | n_members_full | macro_f1 | weighted_f1 | balanced_acc | accuracy | roc_auc | unpruned_macro_f1 | unpruned_weighted_f1 | unpruned_balanced_acc | unpruned_accuracy | unpruned_roc_auc | single_macro_f1 | single_weighted_f1 | single_balanced_acc | single_accuracy | single_roc_auc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 140 | 0.4700 | 58 | 87 | 0.7780 | 0.7785 | 0.7779 | 0.7786 | 0.8530 | 0.7780 | 0.7785 | 0.7779 | 0.7786 | 0.8542 | 0.7281 | 0.7286 | 0.7281 | 0.7286 | 0.8323 |
| 1 | 140 | 0.5000 | 42 | 67 | 0.7521 | 0.7536 | 0.7524 | 0.7571 | 0.7970 | 0.7375 | 0.7391 | 0.7381 | 0.7429 | 0.7970 | 0.7114 | 0.7126 | 0.7113 | 0.7143 | 0.7856 |
| 2 | 140 | 0.4900 | 56 | 84 | 0.8067 | 0.8072 | 0.8069 | 0.8071 | 0.8475 | 0.7996 | 0.8001 | 0.8002 | 0.8000 | 0.8477 | 0.7780 | 0.7787 | 0.7783 | 0.7786 | 0.8315 |
| 3 | 140 | 0.4900 | 60 | 90 | 0.6843 | 0.6873 | 0.6857 | 0.6929 | 0.7447 | 0.6778 | 0.6807 | 0.6790 | 0.6857 | 0.7457 | 0.6550 | 0.6582 | 0.6570 | 0.6643 | 0.7460 |
| 4 | 140 | 0.5100 | 66 | 96 | 0.7571 | 0.7571 | 0.7596 | 0.7571 | 0.8364 | 0.7429 | 0.7429 | 0.7453 | 0.7429 | 0.8362 | 0.7500 | 0.7501 | 0.7520 | 0.7500 | 0.8279 |
| 5 | 140 | 0.5000 | 43 | 72 | 0.7710 | 0.7714 | 0.7710 | 0.7714 | 0.8158 | 0.7707 | 0.7712 | 0.7704 | 0.7714 | 0.8144 | 0.7559 | 0.7566 | 0.7555 | 0.7571 | 0.7883 |
| 6 | 140 | 0.5200 | 57 | 86 | 0.7640 | 0.7643 | 0.7642 | 0.7643 | 0.8268 | 0.7712 | 0.7715 | 0.7716 | 0.7714 | 0.8266 | 0.7642 | 0.7644 | 0.7648 | 0.7643 | 0.8080 |
| 7 | 140 | 0.5100 | 51 | 81 | 0.7250 | 0.7268 | 0.7244 | 0.7286 | 0.7684 | 0.7173 | 0.7192 | 0.7168 | 0.7214 | 0.7676 | 0.7182 | 0.7199 | 0.7176 | 0.7214 | 0.7545 |
| 8 | 140 | 0.4800 | 50 | 78 | 0.7428 | 0.7430 | 0.7445 | 0.7429 | 0.8444 | 0.7500 | 0.7499 | 0.7529 | 0.7500 | 0.8456 | 0.7346 | 0.7356 | 0.7344 | 0.7357 | 0.8129 |
| 9 | 140 | 0.4700 | 50 | 78 | 0.7217 | 0.7242 | 0.7219 | 0.7286 | 0.8286 | 0.7217 | 0.7242 | 0.7219 | 0.7286 | 0.8272 | 0.7284 | 0.7309 | 0.7287 | 0.7357 | 0.8202 |
| 10 | 140 | 0.5000 | 46 | 76 | 0.8000 | 0.8001 | 0.8009 | 0.8000 | 0.8769 | 0.8071 | 0.8072 | 0.8077 | 0.8071 | 0.8773 | 0.7855 | 0.7858 | 0.7859 | 0.7857 | 0.8409 |
| 11 | 140 | 0.4900 | 52 | 81 | 0.7326 | 0.7339 | 0.7325 | 0.7357 | 0.7929 | 0.7403 | 0.7414 | 0.7399 | 0.7429 | 0.7923 | 0.6825 | 0.6839 | 0.6827 | 0.6857 | 0.7678 |
| 12 | 140 | 0.5000 | 54 | 77 | 0.7211 | 0.7216 | 0.7217 | 0.7214 | 0.8262 | 0.7211 | 0.7216 | 0.7217 | 0.7214 | 0.8270 | 0.7203 | 0.7213 | 0.7201 | 0.7214 | 0.8274 |
| 13 | 140 | 0.4900 | 54 | 83 | 0.7850 | 0.7857 | 0.7850 | 0.7857 | 0.8610 | 0.8059 | 0.8068 | 0.8053 | 0.8071 | 0.8606 | 0.8059 | 0.8068 | 0.8053 | 0.8071 | 0.8371 |
| 14 | 140 | 0.5300 | 56 | 84 | 0.7190 | 0.7205 | 0.7185 | 0.7214 | 0.7666 | 0.7272 | 0.7283 | 0.7269 | 0.7286 | 0.7680 | 0.7114 | 0.7130 | 0.7109 | 0.7143 | 0.7688 |

## Selected ensemble members — all700

| fold | thr | inner_blend_f1 | top_members | weight_recording | weight_window | weight_sequence |
|---|---|---|---|---|---|---|
| 0 | 0.4700 | 0.7567 | all+avail|rel|lgbm:0.10 | all|rel|svc_rbf:0.06 | all+avail|rel|svc_rbf:0.04 | all|raw|svc_rbf:0.03 | all+avail|rel|rf:0.03 | audio+video|rel|xgb:0.03 | 0.9410 | 0.0590 | 0 |
| 1 | 0.5000 | 0.7697 | phys|rel|extratrees:0.08 | phys+video|rel|extratrees:0.07 | phys+audio|rel|extratrees:0.06 | all|rel|extratrees:0.06 | phys|z|rf:0.05 | win-raw|trimmed|extratrees:0.05 | 0.8600 | 0.1330 | 0.0070 |
| 2 | 0.4900 | 0.7650 | phys|rel|svc_rbf:0.05 | all+avail|rel|lgbm:0.05 | all+avail|rel|svc_rbf:0.04 | phys+audio|rel|svc_rbf:0.04 | phys+video|rel|rf:0.04 | all|rel|rf:0.04 | 1.0000 | 0 | 0 |
| 3 | 0.4900 | 0.7972 | all|rel|rf:0.07 | all|rel|svc_rbf:0.06 | all+avail|rel|svc_rbf:0.06 | all+avail|rel|rf:0.04 | phys|rel|extratrees:0.04 | phys+audio|rel|extratrees:0.04 | 0.9260 | 0.0740 | 0 |
| 4 | 0.5100 | 0.7585 | phys+audio|rel|extratrees:0.15 | all|raw|rf:0.03 | phys+video|rel|svc_rbf:0.03 | phys+video|raw|xgb:0.03 | phys+audio|raw|svc_rbf:0.03 | win-rel|mean|lgbm:0.03 | 0.8560 | 0.1410 | 0.0040 |
| 5 | 0.5000 | 0.7664 | win-raw|trimmed|extratrees:0.11 | win-raw|mean|extratrees:0.10 | all+avail|raw|extratrees:0.05 | phys+video|rel|rf:0.04 | all|raw|extratrees:0.04 | phys|z|extratrees:0.03 | 0.6850 | 0.3070 | 0.0070 |
| 6 | 0.5200 | 0.7711 | audio+video|raw|lgbm:0.06 | all|rel|lgbm:0.05 | all+avail|rel|lgbm:0.04 | phys|raw|svc_rbf:0.04 | audio+video|rel|xgb:0.03 | video|rel|lgbm:0.03 | 0.9110 | 0.0550 | 0.0330 |
| 7 | 0.5100 | 0.7741 | win-raw|mean|extratrees:0.08 | all|rel|rf:0.06 | all+avail|rel|extratrees:0.06 | phys|raw|rf:0.05 | phys+video|raw|extratrees:0.05 | phys|raw|extratrees:0.04 | 0.8440 | 0.1560 | 0 |
| 8 | 0.4800 | 0.7740 | all+avail|rel|svc_rbf:0.07 | audio+video|raw|xgb:0.07 | all+avail|rel|extratrees:0.07 | phys+audio|rel|svc_rbf:0.06 | video|rel|extratrees:0.04 | audio+video|rel|xgb:0.04 | 1.0000 | 0 | 0 |
| 9 | 0.4700 | 0.7730 | all+avail|rel|rf:0.07 | phys|rel|rf:0.06 | all|rel|rf:0.06 | all+avail|rel|svc_rbf:0.04 | win-raw|trimmed|extratrees:0.03 | phys+video|raw|svc_rbf:0.03 | 0.9040 | 0.0960 | 0 |
| 10 | 0.5000 | 0.7679 | phys+video|rel|extratrees:0.10 | win-raw|mean|extratrees:0.09 | video|raw|rf:0.05 | video|raw|extratrees:0.04 | all+avail|rel|mlp:0.04 | video|rel|lgbm:0.04 | 0.8260 | 0.1740 | 0 |
| 11 | 0.4900 | 0.7876 | phys|rel|extratrees:0.08 | phys|raw|extratrees:0.06 | phys+video|rel|extratrees:0.05 | seq-raw|attn|torch:0.05 | phys|raw|rf:0.05 | all+avail|rel|rf:0.04 | 0.8270 | 0.1030 | 0.0700 |
| 12 | 0.5000 | 0.7712 | all+avail|rel|extratrees:0.07 | win-raw|mean|extratrees:0.06 | win-raw|trimmed|extratrees:0.05 | audio+video|raw|xgb:0.04 | win-raw|mean|lgbm:0.04 | phys+video|rel|rf:0.04 | 0.7420 | 0.2360 | 0.0220 |
| 13 | 0.4900 | 0.7595 | all+avail|rel|lgbm:0.10 | all+avail|rel|rf:0.07 | all|rel|rf:0.05 | all+avail|rel|extratrees:0.05 | phys+audio|rel|extratrees:0.04 | video|raw|extratrees:0.04 | 0.9410 | 0.0590 | 0 |
| 14 | 0.5300 | 0.7802 | phys+audio|rel|extratrees:0.08 | phys+video|rel|lgbm:0.06 | phys+video|raw|rf:0.06 | phys+video|rel|rf:0.05 | all|rel|lgbm:0.05 | phys|raw|extratrees:0.05 | 0.9780 | 0.0220 | 0 |

## Inner-CV candidate ranking — all700

| candidate | inner_macro_f1 |
|---|---|
| all|rel|extratrees | 0.7448 |
| all|rel|lgbm | 0.7443 |
| all+avail|rel|rf | 0.7440 |
| phys+audio|rel|extratrees | 0.7431 |
| win-raw|mean|extratrees | 0.7426 |
| all+avail|rel|lgbm | 0.7425 |
| all+avail|rel|extratrees | 0.7417 |
| all|rel|rf | 0.7417 |
| win-raw|trimmed|extratrees | 0.7410 |
| all|rel|svc_rbf | 0.7386 |
| all+avail|rel|svc_rbf | 0.7386 |
| all+avail|rel|xgb | 0.7365 |
| all|rel|xgb | 0.7354 |
| phys+audio|rel|rf | 0.7345 |
| phys+audio|rel|lgbm | 0.7332 |
| phys+video|rel|lgbm | 0.7329 |
| phys+audio|raw|extratrees | 0.7313 |
| phys+video|rel|svc_rbf | 0.7308 |
| phys+video|rel|extratrees | 0.7299 |
| all|raw|extratrees | 0.7297 |
| phys+video|rel|xgb | 0.7295 |
| all+avail|raw|extratrees | 0.7288 |
| all|raw|lgbm | 0.7270 |
| win-raw|trimmed|lgbm | 0.7264 |
| phys+audio|rel|svc_rbf | 0.7264 |

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
  "selection": "bagged greedy w/ replacement on inner OOF; threshold tuned on inner OOF"
}
```

## Environment

- Windows-10-10.0.26200-SP0
- Python 3.11.9
