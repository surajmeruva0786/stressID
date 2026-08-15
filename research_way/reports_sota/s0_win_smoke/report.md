# SOTA run — `s0_win_smoke`

**Generated:** 2026-08-15 05:30:36  
**Primary (all700_macro_f1):** **nan**  
**Best previous:** `s4_windows` at 0.7572 → **no improvement** (+nan)  
**Duration:** 98 s (1.6 min)  

> **Protocol.** Subject-shared repeated stratified CV — the split rule
> the StressID origin paper uses (random 80/20 + SMOTE). Subjects appear
> on both sides. Numbers here are comparable to published ones and are
> **not** comparable to the GroupKFold track in `reports/`, which holds
> subjects out. Origin paper's best reported weighted F1: 0.72.

## What changed

smoke: verify selected-member table

## Headline metrics

| metric | value |
|---|---|
| c364_macro_f1 | 0.6356 |
| c364_macro_f1_std | 0.0665 |
| c364_weighted_f1 | 0.7031 |
| c364_weighted_f1_std | 0.0710 |
| c364_balanced_acc | 0.6335 |
| c364_balanced_acc_std | 0.0477 |
| c364_accuracy | 0.7062 |
| c364_accuracy_std | 0.0906 |
| c364_roc_auc | 0.6699 |
| c364_roc_auc_std | 0.0584 |
| c364_single_macro_f1 | 0.6200 |
| c364_unpruned_macro_f1 | 0.6453 |
| c364_mean_members | 3.2000 |
| c364_mean_members_unpruned | 3.4000 |
| c364_n_eval_folds | 5 |
| c364_n_recordings | 364 |

## Per-fold — c364

| fold | n_test | thr | n_members | n_members_full | macro_f1 | weighted_f1 | balanced_acc | accuracy | roc_auc | unpruned_macro_f1 | unpruned_weighted_f1 | unpruned_balanced_acc | unpruned_accuracy | unpruned_roc_auc | single_macro_f1 | single_weighted_f1 | single_balanced_acc | single_accuracy | single_roc_auc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 73 | 0.6600 | 3 | 4 | 0.5742 | 0.6348 | 0.5958 | 0.6164 | 0.6226 | 0.6224 | 0.7230 | 0.6123 | 0.7534 | 0.6236 | 0.5450 | 0.6183 | 0.5552 | 0.6027 | 0.5604 |
| 1 | 73 | 0.7200 | 4 | 4 | 0.6306 | 0.6908 | 0.6369 | 0.6849 | 0.6099 | 0.6306 | 0.6908 | 0.6369 | 0.6849 | 0.6099 | 0.6224 | 0.7169 | 0.6140 | 0.7534 | 0.6236 |
| 2 | 73 | 0.5200 | 3 | 3 | 0.7338 | 0.7988 | 0.7047 | 0.8219 | 0.7518 | 0.7338 | 0.7988 | 0.7047 | 0.8219 | 0.7518 | 0.6398 | 0.7257 | 0.6282 | 0.7534 | 0.7454 |
| 3 | 73 | 0.7900 | 2 | 2 | 0.5764 | 0.6405 | 0.5842 | 0.6301 | 0.6630 | 0.5764 | 0.6405 | 0.5842 | 0.6301 | 0.6630 | 0.6166 | 0.7036 | 0.6090 | 0.7260 | 0.6841 |
| 4 | 72 | 0.5300 | 4 | 4 | 0.6632 | 0.7505 | 0.6462 | 0.7778 | 0.7019 | 0.6632 | 0.7505 | 0.6462 | 0.7778 | 0.7019 | 0.6762 | 0.7621 | 0.6558 | 0.7917 | 0.6625 |

## Selected ensemble members — c364

| fold | thr | inner_blend_f1 | top_members | weight_recording | weight_window | weight_sequence |
|---|---|---|---|---|---|---|
| 0 | 0.6600 | 0.6648 | seq-rel|gru|torch:0.59 | phys|rel|lgbm:0.35 | win-rel|mean|lgbm:0.06 | 0.3530 | 0.0590 | 0.5880 |
| 1 | 0.7200 | 0.6601 | win-rel|mean|lgbm:0.33 | phys|rel|lgbm:0.28 | seq-rel|gru|torch:0.22 | win-rel|trimmed|lgbm:0.17 | 0.2780 | 0.5000 | 0.2220 |
| 2 | 0.5200 | 0.6101 | win-rel|trimmed|lgbm:0.33 | phys|rel|lgbm:0.33 | seq-rel|gru|torch:0.33 | 0.3330 | 0.3330 | 0.3330 |
| 3 | 0.7900 | 0.6322 | win-rel|trimmed|lgbm:0.67 | win-rel|mean|lgbm:0.33 | 0 | 1.0000 | 0 |
| 4 | 0.5300 | 0.6224 | win-rel|trimmed|lgbm:0.44 | win-rel|mean|lgbm:0.22 | phys|rel|lgbm:0.22 | seq-rel|gru|torch:0.11 | 0.2220 | 0.6670 | 0.1110 |

## Inner-CV candidate ranking — c364

| candidate | inner_macro_f1 |
|---|---|
| win-rel|mean|lgbm | 0.6002 |
| win-rel|trimmed|lgbm | 0.5937 |
| phys|rel|lgbm | 0.5780 |
| seq-rel|gru|torch | 0.5594 |

## Config

```json
{
  "protocol": "subject-shared RepeatedStratifiedKFold (paper-style)",
  "n_folds": 5,
  "repeats": 1,
  "seed": 42,
  "views": [
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
    "lgbm"
  ],
  "greedy_max_size": 6,
  "greedy_bags": 3,
  "feature_version": 3,
  "scopes": [
    "c364"
  ],
  "fast": true,
  "n_par": 1,
  "jobs": 6,
  "window_views": [
    "rel"
  ],
  "torch_archs": [
    "gru"
  ],
  "inner_folds": 2,
  "ensemble_cum_keep": 0.9,
  "selection": "bagged greedy w/ replacement on inner OOF; threshold tuned on inner OOF"
}
```

## Environment

- Windows-10-10.0.26200-SP0
- Python 3.11.9
