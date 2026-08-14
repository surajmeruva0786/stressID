# SOTA run — `s0_smoke`

**Generated:** 2026-08-14 23:34:47  
**Primary (all700_macro_f1):** **nan**  
**Best previous:** `none` at nan → **FIRST RUN** (+nan)  
**Duration:** 62 s (1.0 min)  

> **Protocol.** Subject-shared repeated stratified CV — the split rule
> the StressID origin paper uses (random 80/20 + SMOTE). Subjects appear
> on both sides. Numbers here are comparable to published ones and are
> **not** comparable to the GroupKFold track in `reports/`, which holds
> subjects out. Origin paper's best reported weighted F1: 0.72.

## What changed

harness smoke after candidate refactor

## Headline metrics

| metric | value |
|---|---|
| c364_macro_f1 | 0.5863 |
| c364_macro_f1_std | 0.0290 |
| c364_weighted_f1 | 0.6860 |
| c364_weighted_f1_std | 0.0230 |
| c364_balanced_acc | 0.5830 |
| c364_balanced_acc_std | 0.0266 |
| c364_accuracy | 0.7142 |
| c364_accuracy_std | 0.0314 |
| c364_roc_auc | 0.6726 |
| c364_roc_auc_std | 0.0571 |
| c364_single_macro_f1 | 0.5645 |
| c364_n_eval_folds | 5 |
| c364_n_recordings | 364 |

## Per-fold — c364

| fold | n_test | thr | n_members | macro_f1 | weighted_f1 | balanced_acc | accuracy | roc_auc | single_macro_f1 | single_weighted_f1 | single_balanced_acc | single_accuracy | single_roc_auc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 73 | 0.5500 | 4 | 0.5805 | 0.6922 | 0.5778 | 0.7260 | 0.7245 | 0.5179 | 0.6164 | 0.5179 | 0.6164 | 0.6377 |
| 1 | 73 | 0.4800 | 4 | 0.6023 | 0.7064 | 0.5998 | 0.7534 | 0.6429 | 0.5512 | 0.6791 | 0.5714 | 0.7534 | 0.7042 |
| 2 | 73 | 0.4300 | 3 | 0.6054 | 0.6926 | 0.5994 | 0.7123 | 0.7381 | 0.6654 | 0.7102 | 0.6891 | 0.6986 | 0.7189 |
| 3 | 73 | 0.5100 | 3 | 0.6054 | 0.6926 | 0.5994 | 0.7123 | 0.6548 | 0.5663 | 0.6370 | 0.5701 | 0.6301 | 0.5897 |
| 4 | 72 | 0.4400 | 4 | 0.5380 | 0.6463 | 0.5385 | 0.6667 | 0.6029 | 0.5214 | 0.6543 | 0.5365 | 0.7083 | 0.6591 |
| 0 |  | 0.5500 |  |  |  |  |  |  |  |  |  |  |  |
| 1 |  | 0.4800 |  |  |  |  |  |  |  |  |  |  |  |
| 2 |  | 0.4300 |  |  |  |  |  |  |  |  |  |  |  |
| 3 |  | 0.5100 |  |  |  |  |  |  |  |  |  |  |  |
| 4 |  | 0.4400 |  |  |  |  |  |  |  |  |  |  |  |

## Inner-CV candidate ranking — c364

| candidate | inner_macro_f1 |
|---|---|
| audio|raw|logreg | 0.5858 |
| audio|raw|extratrees | 0.5843 |
| phys|raw|logreg | 0.5428 |
| phys|raw|extratrees | 0.5066 |

## Config

```json
{
  "protocol": "subject-shared RepeatedStratifiedKFold (paper-style)",
  "n_folds": 5,
  "repeats": 1,
  "seed": 42,
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
    "extratrees"
  ],
  "greedy_max_size": 5,
  "greedy_bags": 3,
  "feature_version": 3,
  "scopes": [
    "c364"
  ],
  "fast": true,
  "n_par": 4,
  "jobs": 6,
  "selection": "bagged greedy w/ replacement on inner OOF; threshold tuned on inner OOF"
}
```

## Environment

- Windows-10-10.0.26200-SP0
- Python 3.11.9
