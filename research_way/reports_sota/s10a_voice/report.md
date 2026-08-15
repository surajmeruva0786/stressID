# SOTA run — `s10a_voice`

**Generated:** 2026-08-16 01:34:47  
**Primary (all700_macro_f1):** **nan**  
**Best previous:** `s5_final` at 0.7604 → **no improvement** (+nan)  
**Duration:** 4542 s (75.7 min)  

> **Protocol.** Subject-shared repeated stratified CV — the split rule
> the StressID origin paper uses (random 80/20 + SMOTE). Subjects appear
> on both sides. Numbers here are comparable to published ones and are
> **not** comparable to the GroupKFold track in `reports/`, which holds
> subjects out. Origin paper's best reported weighted F1: 0.72.

## What changed

R10a: voice-quality arm. audioglobal present. c364 scope, 15 folds, seed 101.

## Headline metrics

| metric | value |
|---|---|
| c364_macro_f1 | 0.6749 |
| c364_macro_f1_std | 0.0560 |
| c364_weighted_f1 | 0.7459 |
| c364_weighted_f1_std | 0.0433 |
| c364_balanced_acc | 0.6664 |
| c364_balanced_acc_std | 0.0511 |
| c364_accuracy | 0.7583 |
| c364_accuracy_std | 0.0437 |
| c364_roc_auc | 0.7278 |
| c364_roc_auc_std | 0.0648 |
| c364_single_macro_f1 | 0.6668 |
| c364_unpruned_macro_f1 | 0.6749 |
| c364_mean_members | 43.6000 |
| c364_mean_members_unpruned | 72.4667 |
| c364_n_eval_folds | 15 |
| c364_n_recordings | 364 |

## Per-fold — c364

| fold | n_test | thr | n_members | n_members_full | macro_f1 | weighted_f1 | balanced_acc | accuracy | roc_auc | unpruned_macro_f1 | unpruned_weighted_f1 | unpruned_balanced_acc | unpruned_accuracy | unpruned_roc_auc | single_macro_f1 | single_weighted_f1 | single_balanced_acc | single_accuracy | single_roc_auc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 73 | 0.4700 | 67 | 96 | 0.7198 | 0.7910 | 0.6967 | 0.8082 | 0.7330 | 0.7198 | 0.7910 | 0.6967 | 0.8082 | 0.7283 | 0.7245 | 0.7808 | 0.7245 | 0.7808 | 0.7660 |
| 1 | 73 | 0.5100 | 63 | 93 | 0.6398 | 0.7257 | 0.6282 | 0.7534 | 0.6932 | 0.6430 | 0.7219 | 0.6328 | 0.7397 | 0.6996 | 0.6676 | 0.7334 | 0.6612 | 0.7397 | 0.6419 |
| 2 | 73 | 0.4700 | 42 | 72 | 0.6806 | 0.7512 | 0.6662 | 0.7671 | 0.7601 | 0.6682 | 0.7396 | 0.6566 | 0.7534 | 0.7647 | 0.6209 | 0.6871 | 0.6227 | 0.6849 | 0.7335 |
| 3 | 73 | 0.5500 | 19 | 43 | 0.6992 | 0.7534 | 0.6992 | 0.7534 | 0.7546 | 0.6826 | 0.7325 | 0.6941 | 0.7260 | 0.7546 | 0.6806 | 0.7512 | 0.6662 | 0.7671 | 0.7179 |
| 4 | 72 | 0.5700 | 47 | 77 | 0.6783 | 0.7458 | 0.6731 | 0.7500 | 0.7452 | 0.6783 | 0.7458 | 0.6731 | 0.7500 | 0.7481 | 0.6792 | 0.7524 | 0.6673 | 0.7639 | 0.7135 |
| 5 | 73 | 0.5500 | 47 | 77 | 0.6430 | 0.7270 | 0.6340 | 0.7397 | 0.6538 | 0.6430 | 0.7270 | 0.6340 | 0.7397 | 0.6509 | 0.6561 | 0.7328 | 0.6495 | 0.7397 | 0.6538 |
| 6 | 73 | 0.5400 | 52 | 82 | 0.6054 | 0.6926 | 0.5994 | 0.7123 | 0.6190 | 0.6398 | 0.7257 | 0.6282 | 0.7534 | 0.6200 | 0.6224 | 0.7169 | 0.6140 | 0.7534 | 0.6557 |
| 7 | 73 | 0.5000 | 48 | 78 | 0.7647 | 0.8140 | 0.7473 | 0.8219 | 0.8269 | 0.7647 | 0.8140 | 0.7473 | 0.8219 | 0.8306 | 0.7528 | 0.7959 | 0.7564 | 0.7945 | 0.8187 |
| 8 | 73 | 0.5000 | 42 | 72 | 0.7154 | 0.7733 | 0.7042 | 0.7808 | 0.8324 | 0.7154 | 0.7733 | 0.7042 | 0.7808 | 0.8352 | 0.7051 | 0.7685 | 0.6900 | 0.7808 | 0.8416 |
| 9 | 72 | 0.6200 | 44 | 74 | 0.7407 | 0.7984 | 0.7269 | 0.8056 | 0.7452 | 0.7498 | 0.8023 | 0.7423 | 0.8056 | 0.7413 | 0.6425 | 0.7175 | 0.6385 | 0.7222 | 0.7423 |
| 10 | 73 | 0.5800 | 36 | 66 | 0.5889 | 0.6907 | 0.5840 | 0.7123 | 0.6198 | 0.5475 | 0.6484 | 0.5462 | 0.6575 | 0.6208 | 0.5788 | 0.7014 | 0.5811 | 0.7534 | 0.6104 |
| 11 | 73 | 0.5200 | 25 | 49 | 0.6992 | 0.7534 | 0.6992 | 0.7534 | 0.7674 | 0.7117 | 0.7654 | 0.7088 | 0.7671 | 0.7711 | 0.6992 | 0.7534 | 0.6992 | 0.7534 | 0.7665 |
| 12 | 73 | 0.5700 | 48 | 78 | 0.6160 | 0.6696 | 0.6319 | 0.6575 | 0.7381 | 0.6276 | 0.6817 | 0.6415 | 0.6712 | 0.7399 | 0.6078 | 0.6671 | 0.6177 | 0.6575 | 0.6795 |
| 13 | 73 | 0.5300 | 28 | 55 | 0.7316 | 0.7925 | 0.7092 | 0.8082 | 0.7555 | 0.7316 | 0.7925 | 0.7092 | 0.8082 | 0.7546 | 0.6933 | 0.7629 | 0.6758 | 0.7808 | 0.7701 |
| 14 | 72 | 0.5200 | 46 | 75 | 0.6010 | 0.7094 | 0.5962 | 0.7500 | 0.6731 | 0.6010 | 0.7094 | 0.5962 | 0.7500 | 0.6673 | 0.6714 | 0.7647 | 0.6500 | 0.8056 | 0.6644 |

## Selected ensemble members — c364

| fold | thr | inner_blend_f1 | top_members | weight_recording | weight_window | weight_sequence |
|---|---|---|---|---|---|---|
| 0 | 0.4700 | 0.6922 | seq-raw|attn|torch:0.09 | video|rel|logreg:0.07 | video|raw|logreg_l2w:0.06 | seq-rel|attn|torch:0.05 | phys|rel|logreg_l2w:0.05 | video|raw|logreg:0.05 | 0.8040 | 0.0040 | 0.1920 |
| 1 | 0.5100 | 0.6865 | video|rel|logreg_l2w:0.15 | video|rel|svc_rbf:0.13 | video|rel|logreg:0.06 | audio|z|mlp:0.05 | audio|rel|logreg_l2w:0.05 | audio|raw|logreg_l2w:0.03 | 0.9480 | 0.0070 | 0.0440 |
| 2 | 0.4700 | 0.7196 | seq-rel|gru|torch:0.13 | seq-raw|gru|torch:0.13 | audio+video|raw|logreg_l2w:0.11 | audio+video|raw|logreg:0.06 | video|raw|logreg_l2w:0.06 | phys+video|rel|mlp:0.04 | 0.6850 | 0 | 0.3150 |
| 3 | 0.5500 | 0.7222 | win-raw|trimmed|xgb:0.24 | video|rel|logreg_l2w:0.15 | audio|raw|logreg_l2w:0.14 | win-raw|mean|xgb:0.14 | audio+video|rel|logreg_l2w:0.05 | audio|z|logreg_l2w:0.04 | 0.5590 | 0.4080 | 0.0330 |
| 4 | 0.5700 | 0.7083 | seq-rel|gru|torch:0.19 | all|rel|mlp:0.18 | seq-rel|attn|torch:0.09 | win-raw|trimmed|xgb:0.04 | phys+audio|rel|svc_rbf:0.03 | audio|raw|logreg_l2w:0.03 | 0.6000 | 0.1220 | 0.2780 |
| 5 | 0.5500 | 0.7519 | video|rel|logreg:0.09 | seq-rel|gru|torch:0.09 | video|rel|logreg_l2w:0.07 | phys+video|rel|svc_rbf:0.05 | seq-raw|gru|torch:0.05 | seq-raw|attn|torch:0.05 | 0.7700 | 0.0040 | 0.2260 |
| 6 | 0.5400 | 0.7555 | seq-raw|attn|torch:0.14 | seq-rel|gru|torch:0.10 | seq-raw|gru|torch:0.06 | audio+video|raw|logreg_l2w:0.05 | all|raw|mlp:0.04 | audio+video|rel|svc_rbf:0.04 | 0.6440 | 0.0410 | 0.3150 |
| 7 | 0.5000 | 0.6860 | seq-rel|attn|torch:0.10 | phys+video|rel|mlp:0.08 | video|rel|mlp:0.06 | all|rel|svc_rbf:0.05 | win-raw|mean|xgb:0.04 | phys+video|rel|svc_rbf:0.04 | 0.7780 | 0.0850 | 0.1370 |
| 8 | 0.5000 | 0.6922 | seq-raw|attn|torch:0.15 | phys+video|rel|svc_rbf:0.13 | all+avail|rel|svc_rbf:0.06 | seq-rel|attn|torch:0.06 | all|rel|svc_rbf:0.06 | seq-rel|gru|torch:0.06 | 0.6220 | 0.0890 | 0.2890 |
| 9 | 0.6200 | 0.7002 | video|rel|mlp:0.10 | video|rel|logreg_l2w:0.10 | video|rel|logreg:0.07 | win-raw|trimmed|lgbm:0.06 | audio|raw|lgbm:0.06 | seq-raw|gru|torch:0.04 | 0.7810 | 0.1300 | 0.0890 |
| 10 | 0.5800 | 0.7212 | seq-rel|gru|torch:0.23 | audio+video|rel|xgb:0.12 | audio+video|rel|lgbm:0.11 | video|rel|lgbm:0.10 | seq-raw|gru|torch:0.06 | video|rel|xgb:0.04 | 0.6630 | 0.0070 | 0.3300 |
| 11 | 0.5200 | 0.7046 | audio+video|rel|logreg_l2w:0.19 | audio+video|rel|logreg:0.11 | seq-raw|gru|torch:0.10 | seq-rel|attn|torch:0.09 | seq-rel|gru|torch:0.07 | video|rel|logreg_l2w:0.06 | 0.6970 | 0 | 0.3030 |
| 12 | 0.5700 | 0.7094 | video|rel|logreg_l2w:0.16 | audio+video|rel|logreg_l2w:0.06 | video|raw|logreg:0.06 | seq-rel|attn|torch:0.05 | seq-raw|gru|torch:0.05 | all|rel|svc_rbf:0.04 | 0.8330 | 0.0150 | 0.1520 |
| 13 | 0.5300 | 0.7361 | phys+video|rel|svc_rbf:0.24 | seq-raw|gru|torch:0.14 | phys+video|rel|mlp:0.10 | seq-rel|gru|torch:0.08 | seq-raw|attn|torch:0.07 | seq-rel|attn|torch:0.04 | 0.6420 | 0.0260 | 0.3320 |
| 14 | 0.5200 | 0.7128 | seq-rel|gru|torch:0.13 | video|raw|svc_rbf:0.07 | win-raw|mean|xgb:0.06 | seq-rel|attn|torch:0.06 | video|rel|lgbm:0.04 | phys+video|raw|svc_rbf:0.04 | 0.7040 | 0.1040 | 0.1930 |

## Inner-CV candidate ranking — c364

| candidate | inner_macro_f1 |
|---|---|
| video|rel|logreg_l2w | 0.6763 |
| video|rel|logreg | 0.6719 |
| phys+video|rel|svc_rbf | 0.6700 |
| audio+video|rel|logreg | 0.6676 |
| audio+video|rel|logreg_l2w | 0.6654 |
| win-raw|trimmed|xgb | 0.6624 |
| win-raw|mean|lgbm | 0.6617 |
| win-raw|trimmed|lgbm | 0.6611 |
| win-raw|mean|xgb | 0.6600 |
| all|rel|svc_rbf | 0.6595 |
| all+avail|rel|svc_rbf | 0.6595 |
| audio+video|raw|logreg_l2w | 0.6551 |
| phys+video|rel|logreg | 0.6547 |
| audio+video|raw|logreg | 0.6544 |
| phys+video|rel|logreg_l2w | 0.6529 |
| all|rel|logreg | 0.6509 |
| all+avail|rel|logreg | 0.6509 |
| all|rel|logreg_l2w | 0.6509 |
| all+avail|rel|logreg_l2w | 0.6509 |
| audio+video|rel|svc_rbf | 0.6479 |
| video|raw|logreg | 0.6473 |
| video|raw|logreg_l2w | 0.6469 |
| all|rel|lgbm | 0.6449 |
| all+avail|rel|lgbm | 0.6449 |
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
  "exclude_blocks": [],
  "selection": "bagged greedy w/ replacement on inner OOF; threshold tuned on inner OOF"
}
```

## Environment

- Windows-10-10.0.26200-SP0
- Python 3.11.9
