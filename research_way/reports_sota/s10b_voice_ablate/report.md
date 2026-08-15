# SOTA run — `s10b_voice_ablate`

**Generated:** 2026-08-16 02:50:01  
**Primary (all700_macro_f1):** **nan**  
**Best previous:** `s5_final` at 0.7604 → **no improvement** (+nan)  
**Duration:** 4507 s (75.1 min)  

> **Protocol.** Subject-shared repeated stratified CV — the split rule
> the StressID origin paper uses (random 80/20 + SMOTE). Subjects appear
> on both sides. Numbers here are comparable to published ones and are
> **not** comparable to the GroupKFold track in `reports/`, which holds
> subjects out. Origin paper's best reported weighted F1: 0.72.

## What changed

R10b: ablation arm, byte-identical to R10a except --exclude-blocks audioglobal.

## Headline metrics

| metric | value |
|---|---|
| c364_macro_f1 | 0.6725 |
| c364_macro_f1_std | 0.0610 |
| c364_weighted_f1 | 0.7420 |
| c364_weighted_f1_std | 0.0457 |
| c364_balanced_acc | 0.6681 |
| c364_balanced_acc_std | 0.0562 |
| c364_accuracy | 0.7527 |
| c364_accuracy_std | 0.0458 |
| c364_roc_auc | 0.7249 |
| c364_roc_auc_std | 0.0687 |
| c364_single_macro_f1 | 0.6639 |
| c364_unpruned_macro_f1 | 0.6736 |
| c364_mean_members | 45.1333 |
| c364_mean_members_unpruned | 74.4000 |
| c364_n_eval_folds | 15 |
| c364_n_recordings | 364 |

## Per-fold — c364

| fold | n_test | thr | n_members | n_members_full | macro_f1 | weighted_f1 | balanced_acc | accuracy | roc_auc | unpruned_macro_f1 | unpruned_weighted_f1 | unpruned_balanced_acc | unpruned_accuracy | unpruned_roc_auc | single_macro_f1 | single_weighted_f1 | single_balanced_acc | single_accuracy | single_roc_auc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 73 | 0.4800 | 51 | 81 | 0.7198 | 0.7910 | 0.6967 | 0.8082 | 0.7208 | 0.7198 | 0.7910 | 0.6967 | 0.8082 | 0.7255 | 0.7245 | 0.7808 | 0.7245 | 0.7808 | 0.7660 |
| 1 | 73 | 0.5800 | 65 | 94 | 0.6430 | 0.7219 | 0.6328 | 0.7397 | 0.6932 | 0.6430 | 0.7219 | 0.6328 | 0.7397 | 0.6914 | 0.6676 | 0.7334 | 0.6612 | 0.7397 | 0.6419 |
| 2 | 73 | 0.5500 | 49 | 79 | 0.6901 | 0.7496 | 0.6850 | 0.7534 | 0.7564 | 0.6798 | 0.7450 | 0.6708 | 0.7534 | 0.7582 | 0.6422 | 0.7025 | 0.6465 | 0.6986 | 0.7372 |
| 3 | 73 | 0.5700 | 35 | 62 | 0.6775 | 0.7225 | 0.6987 | 0.7123 | 0.7454 | 0.7019 | 0.7470 | 0.7179 | 0.7397 | 0.7463 | 0.6806 | 0.7512 | 0.6662 | 0.7671 | 0.7179 |
| 4 | 72 | 0.6200 | 37 | 67 | 0.6486 | 0.7050 | 0.6654 | 0.6944 | 0.7385 | 0.6606 | 0.7172 | 0.6750 | 0.7083 | 0.7385 | 0.6761 | 0.7380 | 0.6788 | 0.7361 | 0.7144 |
| 5 | 73 | 0.5300 | 38 | 66 | 0.6672 | 0.7496 | 0.6528 | 0.7671 | 0.6557 | 0.6672 | 0.7496 | 0.6528 | 0.7671 | 0.6538 | 0.6561 | 0.7328 | 0.6495 | 0.7397 | 0.6538 |
| 6 | 73 | 0.5300 | 54 | 83 | 0.5944 | 0.6817 | 0.5897 | 0.6986 | 0.6026 | 0.5837 | 0.6709 | 0.5801 | 0.6849 | 0.6053 | 0.5998 | 0.6952 | 0.5948 | 0.7260 | 0.6429 |
| 7 | 73 | 0.4900 | 48 | 76 | 0.7285 | 0.7853 | 0.7138 | 0.7945 | 0.8297 | 0.7510 | 0.8017 | 0.7376 | 0.8082 | 0.8324 | 0.6682 | 0.7396 | 0.6566 | 0.7534 | 0.7454 |
| 8 | 73 | 0.5100 | 44 | 74 | 0.7376 | 0.7895 | 0.7280 | 0.7945 | 0.8434 | 0.7376 | 0.7895 | 0.7280 | 0.7945 | 0.8388 | 0.7051 | 0.7685 | 0.6900 | 0.7808 | 0.8416 |
| 9 | 72 | 0.6200 | 56 | 86 | 0.7140 | 0.7740 | 0.7077 | 0.7778 | 0.7471 | 0.7272 | 0.7861 | 0.7173 | 0.7917 | 0.7442 | 0.6761 | 0.7380 | 0.6788 | 0.7361 | 0.7481 |
| 10 | 73 | 0.5000 | 38 | 68 | 0.5581 | 0.6812 | 0.5623 | 0.7260 | 0.6198 | 0.5944 | 0.6874 | 0.5901 | 0.6986 | 0.6236 | 0.5912 | 0.7026 | 0.5873 | 0.7397 | 0.6585 |
| 11 | 73 | 0.5200 | 33 | 61 | 0.7590 | 0.8052 | 0.7518 | 0.8082 | 0.7756 | 0.6948 | 0.7445 | 0.7038 | 0.7397 | 0.7821 | 0.6992 | 0.7534 | 0.6992 | 0.7534 | 0.7701 |
| 12 | 73 | 0.5600 | 47 | 77 | 0.6276 | 0.6817 | 0.6415 | 0.6712 | 0.7253 | 0.6212 | 0.6939 | 0.6181 | 0.6986 | 0.7271 | 0.6078 | 0.6671 | 0.6177 | 0.6575 | 0.6795 |
| 13 | 73 | 0.5500 | 35 | 65 | 0.7316 | 0.7925 | 0.7092 | 0.8082 | 0.7564 | 0.7316 | 0.7925 | 0.7092 | 0.8082 | 0.7564 | 0.6933 | 0.7629 | 0.6758 | 0.7808 | 0.7701 |
| 14 | 72 | 0.5400 | 47 | 77 | 0.5898 | 0.6987 | 0.5865 | 0.7361 | 0.6644 | 0.5898 | 0.6987 | 0.5865 | 0.7361 | 0.6635 | 0.6714 | 0.7647 | 0.6500 | 0.8056 | 0.6644 |

## Selected ensemble members — c364

| fold | thr | inner_blend_f1 | top_members | weight_recording | weight_window | weight_sequence |
|---|---|---|---|---|---|---|
| 0 | 0.4800 | 0.7096 | phys|rel|mlp:0.07 | seq-raw|attn|torch:0.07 | video|rel|logreg:0.07 | seq-rel|attn|torch:0.06 | audio|z|logreg_l2w:0.06 | seq-raw|gru|torch:0.05 | 0.7890 | 0 | 0.2110 |
| 1 | 0.5800 | 0.6883 | video|rel|logreg_l2w:0.17 | video|rel|svc_rbf:0.11 | video|rel|logreg:0.10 | audio+video|rel|logreg_l2w:0.04 | phys+video|rel|logreg:0.03 | audio|raw|logreg_l2w:0.03 | 0.9520 | 0.0260 | 0.0220 |
| 2 | 0.5500 | 0.7216 | seq-rel|gru|torch:0.19 | seq-raw|gru|torch:0.12 | audio+video|raw|logreg_l2w:0.07 | seq-raw|attn|torch:0.06 | phys+video|rel|mlp:0.05 | video|raw|logreg_l2w:0.04 | 0.6330 | 0 | 0.3670 |
| 3 | 0.5700 | 0.7257 | win-raw|trimmed|xgb:0.13 | video|rel|logreg_l2w:0.11 | audio|raw|logreg_l2w:0.10 | seq-rel|gru|torch:0.08 | audio|z|logreg_l2w:0.07 | audio+video|rel|logreg_l2w:0.04 | 0.6040 | 0.2300 | 0.1670 |
| 4 | 0.6200 | 0.6920 | audio+video|rel|logreg_l2w:0.18 | seq-rel|gru|torch:0.14 | seq-rel|attn|torch:0.10 | video|rel|logreg:0.07 | win-raw|trimmed|lgbm:0.06 | video|rel|logreg_l2w:0.05 | 0.6150 | 0.1150 | 0.2700 |
| 5 | 0.5300 | 0.7474 | seq-rel|gru|torch:0.08 | video|rel|logreg_l2w:0.08 | phys+video|rel|svc_rbf:0.07 | video|rel|logreg:0.06 | seq-raw|gru|torch:0.06 | audio+video|rel|logreg_l2w:0.05 | 0.8080 | 0.0070 | 0.1850 |
| 6 | 0.5300 | 0.7588 | seq-raw|attn|torch:0.13 | seq-raw|gru|torch:0.06 | phys|rel|logreg_l2w:0.05 | seq-rel|gru|torch:0.05 | audio+video|raw|logreg_l2w:0.05 | seq-rel|attn|torch:0.04 | 0.7050 | 0.0180 | 0.2770 |
| 7 | 0.4900 | 0.6785 | seq-rel|attn|torch:0.12 | phys+video|rel|mlp:0.07 | seq-rel|gru|torch:0.05 | video|rel|mlp:0.05 | win-raw|mean|xgb:0.04 | phys+video|rel|svc_rbf:0.04 | 0.7380 | 0.0700 | 0.1920 |
| 8 | 0.5100 | 0.7028 | phys+video|rel|svc_rbf:0.11 | seq-raw|attn|torch:0.11 | seq-rel|attn|torch:0.07 | all+avail|rel|svc_rbf:0.06 | seq-rel|gru|torch:0.06 | all|rel|svc_rbf:0.05 | 0.6590 | 0.0890 | 0.2520 |
| 9 | 0.6200 | 0.6913 | video|rel|logreg_l2w:0.11 | seq-raw|gru|torch:0.07 | video|rel|mlp:0.05 | audio|rel|extratrees:0.04 | seq-rel|gru|torch:0.04 | win-raw|trimmed|lgbm:0.04 | 0.7930 | 0.0930 | 0.1150 |
| 10 | 0.5000 | 0.7200 | audio+video|rel|lgbm:0.23 | audio+video|rel|xgb:0.15 | seq-rel|gru|torch:0.15 | video|rel|lgbm:0.09 | phys+video|rel|mlp:0.03 | video|rel|rf:0.02 | 0.8110 | 0 | 0.1890 |
| 11 | 0.5200 | 0.6972 | audio+video|rel|logreg_l2w:0.14 | seq-raw|gru|torch:0.09 | seq-raw|attn|torch:0.09 | seq-rel|attn|torch:0.08 | seq-rel|gru|torch:0.07 | video|rel|logreg_l2w:0.06 | 0.6570 | 0.0110 | 0.3320 |
| 12 | 0.5600 | 0.7191 | video|rel|logreg_l2w:0.16 | seq-rel|attn|torch:0.06 | seq-raw|gru|torch:0.06 | video|raw|logreg_l2w:0.06 | all|rel|svc_rbf:0.05 | video|raw|logreg:0.03 | 0.8300 | 0.0110 | 0.1590 |
| 13 | 0.5500 | 0.7390 | phys+video|rel|svc_rbf:0.26 | seq-raw|gru|torch:0.15 | phys+video|rel|mlp:0.10 | seq-raw|attn|torch:0.08 | seq-rel|gru|torch:0.05 | audio+video|raw|lgbm:0.04 | 0.6590 | 0.0330 | 0.3070 |
| 14 | 0.5400 | 0.7158 | seq-rel|gru|torch:0.10 | win-raw|mean|xgb:0.08 | seq-rel|attn|torch:0.08 | phys+video|raw|svc_rbf:0.06 | phys+audio|rel|mlp:0.05 | video|rel|lgbm:0.05 | 0.6630 | 0.1410 | 0.1960 |

## Inner-CV candidate ranking — c364

| candidate | inner_macro_f1 |
|---|---|
| video|rel|logreg_l2w | 0.6763 |
| video|rel|logreg | 0.6719 |
| phys+video|rel|svc_rbf | 0.6700 |
| audio+video|rel|logreg_l2w | 0.6673 |
| audio+video|rel|logreg | 0.6669 |
| all|rel|svc_rbf | 0.6624 |
| all+avail|rel|svc_rbf | 0.6624 |
| win-raw|trimmed|xgb | 0.6624 |
| win-raw|mean|lgbm | 0.6617 |
| win-raw|trimmed|lgbm | 0.6611 |
| win-raw|mean|xgb | 0.6600 |
| audio+video|raw|logreg_l2w | 0.6568 |
| audio+video|raw|logreg | 0.6554 |
| phys+video|rel|logreg | 0.6547 |
| phys+video|rel|logreg_l2w | 0.6529 |
| all|rel|logreg | 0.6505 |
| all+avail|rel|logreg | 0.6505 |
| all|rel|logreg_l2w | 0.6488 |
| all+avail|rel|logreg_l2w | 0.6488 |
| audio+video|rel|svc_rbf | 0.6485 |
| video|raw|logreg | 0.6473 |
| video|raw|logreg_l2w | 0.6469 |
| all|rel|lgbm | 0.6448 |
| all+avail|rel|lgbm | 0.6448 |
| video|rel|svc_rbf | 0.6433 |

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
    "audioglobal",
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
    "c364"
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
    "audioglobal"
  ],
  "selection": "bagged greedy w/ replacement on inner OOF; threshold tuned on inner OOF"
}
```

## Environment

- Windows-10-10.0.26200-SP0
- Python 3.11.9
