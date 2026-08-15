# SOTA run — `s5_final`

**Generated:** 2026-08-15 08:38:10  
**Primary (all700_macro_f1):** **0.7604**  
**Best previous:** `s4_windows` at 0.7572 → **NEW BEST** (+0.0032)  
**Duration:** 11206 s (186.8 min)  

> **Protocol.** Subject-shared repeated stratified CV — the split rule
> the StressID origin paper uses (random 80/20 + SMOTE). Subjects appear
> on both sides. Numbers here are comparable to published ones and are
> **not** comparable to the GroupKFold track in `reports/`, which holds
> subjects out. Origin paper's best reported weighted F1: 0.72.

## What changed

R5 final: everything the campaign learned, combined. Views raw+rel+z (z restored -- R3 showed removing it cost 0.0053 as ensemble diversity even though it never ranked individually), window-level tree candidates on raw+rel, GPU sequence candidates (gru and attn), ensemble pruned to 0.90 cumulative weight. Both scopes. This is the configuration reported as the headline result.

## Headline metrics

| metric | value |
|---|---|
| all700_macro_f1 | 0.7604 |
| all700_macro_f1_std | 0.0374 |
| all700_weighted_f1 | 0.7612 |
| all700_weighted_f1_std | 0.0376 |
| all700_balanced_acc | 0.7602 |
| all700_balanced_acc_std | 0.0370 |
| all700_accuracy | 0.7614 |
| all700_accuracy_std | 0.0380 |
| all700_roc_auc | 0.8226 |
| all700_roc_auc_std | 0.0495 |
| all700_single_macro_f1 | 0.7337 |
| all700_unpruned_macro_f1 | 0.7575 |
| all700_mean_members | 54.4000 |
| all700_mean_members_unpruned | 83.4000 |
| all700_n_eval_folds | 5 |
| all700_n_recordings | 700 |
| c364_macro_f1 | 0.6435 |
| c364_macro_f1_std | 0.0564 |
| c364_weighted_f1 | 0.7163 |
| c364_weighted_f1_std | 0.0474 |
| c364_balanced_acc | 0.6385 |
| c364_balanced_acc_std | 0.0537 |
| c364_accuracy | 0.7226 |
| c364_accuracy_std | 0.0504 |
| c364_roc_auc | 0.6964 |
| c364_roc_auc_std | 0.0509 |
| c364_single_macro_f1 | 0.6340 |
| c364_unpruned_macro_f1 | 0.6460 |
| c364_mean_members | 50.0000 |
| c364_mean_members_unpruned | 79.0000 |
| c364_n_eval_folds | 5 |
| c364_n_recordings | 364 |

## Per-fold — all700

| fold | n_test | thr | n_members | n_members_full | macro_f1 | weighted_f1 | balanced_acc | accuracy | roc_auc | unpruned_macro_f1 | unpruned_weighted_f1 | unpruned_balanced_acc | unpruned_accuracy | unpruned_roc_auc | single_macro_f1 | single_weighted_f1 | single_balanced_acc | single_accuracy | single_roc_auc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 140 | 0.5100 | 53 | 82 | 0.7064 | 0.7070 | 0.7063 | 0.7071 | 0.7446 | 0.6925 | 0.6929 | 0.6926 | 0.6929 | 0.7457 | 0.7064 | 0.7070 | 0.7063 | 0.7071 | 0.7281 |
| 1 | 140 | 0.5100 | 51 | 81 | 0.7642 | 0.7644 | 0.7648 | 0.7643 | 0.8131 | 0.7633 | 0.7640 | 0.7629 | 0.7643 | 0.8111 | 0.7356 | 0.7354 | 0.7380 | 0.7357 | 0.8144 |
| 2 | 140 | 0.4900 | 53 | 79 | 0.8118 | 0.8131 | 0.8104 | 0.8143 | 0.8786 | 0.8118 | 0.8131 | 0.8104 | 0.8143 | 0.8780 | 0.7974 | 0.7987 | 0.7961 | 0.8000 | 0.8654 |
| 3 | 140 | 0.5100 | 55 | 85 | 0.7637 | 0.7644 | 0.7639 | 0.7643 | 0.8382 | 0.7569 | 0.7573 | 0.7580 | 0.7571 | 0.8372 | 0.7266 | 0.7279 | 0.7260 | 0.7286 | 0.8138 |
| 4 | 140 | 0.5100 | 60 | 90 | 0.7559 | 0.7569 | 0.7555 | 0.7571 | 0.8382 | 0.7628 | 0.7639 | 0.7623 | 0.7643 | 0.8376 | 0.7024 | 0.7058 | 0.7052 | 0.7143 | 0.8139 |

## Selected ensemble members — all700

| fold | thr | inner_blend_f1 | top_members | weight_recording | weight_window | weight_sequence |
|---|---|---|---|---|---|---|
| 0 | 0.5100 | 0.7724 | all|rel|rf:0.13 | all|rel|extratrees:0.07 | all+avail|rel|extratrees:0.05 | phys+video|rel|extratrees:0.05 | all+avail|rel|lgbm:0.04 | phys+video|rel|rf:0.04 | 0.9370 | 0.0560 | 0.0070 |
| 1 | 0.5100 | 0.7486 | phys|rel|extratrees:0.16 | phys+video|rel|extratrees:0.05 | win-raw|trimmed|extratrees:0.05 | phys+video|raw|rf:0.04 | phys+video|rel|rf:0.04 | phys+video|raw|extratrees:0.03 | 0.8960 | 0.1040 | 0 |
| 2 | 0.4900 | 0.7602 | all+avail|rel|extratrees:0.06 | all|rel|rf:0.06 | phys+video|rel|extratrees:0.05 | video|raw|extratrees:0.04 | phys+video|z|svc_rbf:0.03 | phys+video|rel|svc_rbf:0.03 | 0.9190 | 0.0520 | 0.0300 |
| 3 | 0.5100 | 0.7759 | win-raw|mean|extratrees:0.09 | phys+video|rel|extratrees:0.06 | win-raw|mean|xgb:0.04 | win-raw|trimmed|extratrees:0.04 | phys|rel|svc_rbf:0.04 | video|raw|logreg_l2w:0.04 | 0.8070 | 0.1930 | 0 |
| 4 | 0.5100 | 0.7724 | phys|raw|extratrees:0.07 | phys+audio|rel|lgbm:0.07 | all+avail|rel|svc_rbf:0.04 | phys+video|rel|extratrees:0.04 | all|rel|svc_rbf:0.04 | win-raw|mean|extratrees:0.04 | 0.8630 | 0.1370 | 0 |

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

## Per-fold — c364

| fold | n_test | thr | n_members | n_members_full | macro_f1 | weighted_f1 | balanced_acc | accuracy | roc_auc | unpruned_macro_f1 | unpruned_weighted_f1 | unpruned_balanced_acc | unpruned_accuracy | unpruned_roc_auc | single_macro_f1 | single_weighted_f1 | single_balanced_acc | single_accuracy | single_roc_auc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 73 | 0.5400 | 54 | 84 | 0.6054 | 0.6983 | 0.5995 | 0.7123 | 0.6509 | 0.5944 | 0.6874 | 0.5901 | 0.6986 | 0.6547 | 0.6557 | 0.7260 | 0.6557 | 0.7260 | 0.6538 |
| 1 | 73 | 0.6000 | 58 | 88 | 0.6561 | 0.7281 | 0.6470 | 0.7397 | 0.6712 | 0.6682 | 0.7396 | 0.6566 | 0.7534 | 0.6712 | 0.6398 | 0.7257 | 0.6282 | 0.7534 | 0.6694 |
| 2 | 73 | 0.5400 | 42 | 72 | 0.7245 | 0.7774 | 0.7184 | 0.7808 | 0.7821 | 0.7245 | 0.7774 | 0.7184 | 0.7808 | 0.7821 | 0.6747 | 0.7296 | 0.6799 | 0.7260 | 0.7903 |
| 3 | 73 | 0.5600 | 31 | 56 | 0.5771 | 0.6484 | 0.5797 | 0.6438 | 0.6786 | 0.5761 | 0.6550 | 0.5751 | 0.6575 | 0.6822 | 0.5989 | 0.6712 | 0.5989 | 0.6712 | 0.6172 |
| 4 | 72 | 0.5900 | 65 | 95 | 0.6545 | 0.7291 | 0.6481 | 0.7361 | 0.6990 | 0.6667 | 0.7407 | 0.6577 | 0.7500 | 0.7010 | 0.6010 | 0.7094 | 0.5962 | 0.7500 | 0.7510 |

## Selected ensemble members — c364

| fold | thr | inner_blend_f1 | top_members | weight_recording | weight_window | weight_sequence |
|---|---|---|---|---|---|---|
| 0 | 0.5400 | 0.7206 | seq-raw|attn|torch:0.15 | seq-rel|attn|torch:0.13 | video|raw|logreg_l2w:0.07 | seq-rel|gru|torch:0.07 | phys|rel|mlp:0.03 | audio+video|raw|logreg_l2w:0.03 | 0.5630 | 0.0670 | 0.3700 |
| 1 | 0.6000 | 0.7086 | audio+video|rel|xgb:0.11 | seq-raw|gru|torch:0.09 | phys+video|rel|svc_rbf:0.06 | seq-raw|attn|torch:0.05 | all|rel|svc_rbf:0.04 | video|rel|svc_rbf:0.04 | 0.7930 | 0.0150 | 0.1930 |
| 2 | 0.5400 | 0.6991 | seq-raw|gru|torch:0.15 | video|rel|logreg_l2w:0.14 | audio+video|rel|logreg_l2w:0.11 | video|rel|logreg:0.10 | seq-rel|gru|torch:0.06 | seq-rel|attn|torch:0.04 | 0.7110 | 0 | 0.2890 |
| 3 | 0.5600 | 0.7623 | win-raw|mean|lgbm:0.13 | seq-raw|attn|torch:0.10 | seq-rel|gru|torch:0.10 | seq-rel|attn|torch:0.09 | phys+video|rel|svc_rbf:0.07 | seq-raw|gru|torch:0.06 | 0.4130 | 0.2440 | 0.3430 |
| 4 | 0.5900 | 0.7342 | audio+video|raw|lgbm:0.08 | seq-raw|gru|torch:0.07 | audio|z|logreg_l2w:0.07 | video|raw|lgbm:0.06 | audio|raw|logreg:0.06 | audio|rel|logreg:0.04 | 0.8670 | 0.0330 | 0.1000 |

## Inner-CV candidate ranking — c364

| candidate | inner_macro_f1 |
|---|---|
| win-raw|mean|lgbm | 0.6835 |
| win-raw|trimmed|lgbm | 0.6834 |
| phys+video|rel|svc_rbf | 0.6817 |
| win-raw|trimmed|xgb | 0.6768 |
| video|rel|logreg_l2w | 0.6742 |
| win-raw|mean|xgb | 0.6740 |
| all|rel|svc_rbf | 0.6730 |
| all+avail|rel|svc_rbf | 0.6730 |
| audio+video|rel|logreg | 0.6720 |
| video|rel|lgbm | 0.6713 |
| audio+video|rel|logreg_l2w | 0.6702 |
| phys+video|rel|logreg | 0.6698 |
| phys+video|rel|logreg_l2w | 0.6688 |
| all|rel|logreg | 0.6674 |
| all+avail|rel|logreg | 0.6674 |
| phys+video|rel|lgbm | 0.6668 |
| all|rel|logreg_l2w | 0.6667 |
| all+avail|rel|logreg_l2w | 0.6667 |
| all|rel|lgbm | 0.6662 |
| all+avail|rel|lgbm | 0.6662 |
| video|rel|logreg | 0.6640 |
| audio+video|rel|lgbm | 0.6633 |
| audio+video|rel|xgb | 0.6612 |
| video|raw|lgbm | 0.6581 |
| audio+video|raw|logreg_l2w | 0.6558 |

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
