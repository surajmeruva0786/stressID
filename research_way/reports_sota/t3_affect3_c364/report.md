# SOTA run — `t3_affect3_c364`

**Generated:** 2026-08-21 12:17:02  
**Primary (all700_macro_f1):** **nan**  
**Best previous:** `s5_final` at 0.7604 → **no improvement** (+nan)  
**Duration:** 3423 s (57.1 min)  

> **Protocol.** Subject-shared repeated stratified CV — the split rule
> the StressID origin paper uses (random 80/20 + SMOTE). Subjects appear
> on both sides. Numbers here are comparable to published ones and are
> **not** comparable to the GroupKFold track in `reports/`, which holds
> subjects out. Origin paper's best reported weighted F1: 0.72.

## What changed

T3: 3-class affect restricted to c364, matching the origin paper's 370-task multimodal evaluation scope.

## Headline metrics

| metric | value |
|---|---|
| affect3_macro_f1 | 0.6195 |
| affect3_macro_f1_std | 0.0673 |
| affect3_weighted_f1 | 0.6533 |
| affect3_weighted_f1_std | 0.0526 |
| affect3_balanced_acc | 0.5964 |
| affect3_balanced_acc_std | 0.0599 |
| affect3_accuracy | 0.6732 |
| affect3_accuracy_std | 0.0458 |
| affect3_single_macro_f1 | 0.5928 |
| affect3_n_folds | 5 |
| affect3_n_recordings | 364 |
| affect3_scope | c364 |

## Per-fold — affect3

| fold | n_test | n_members | macro_f1 | weighted_f1 | balanced_acc | accuracy | single_macro_f1 | single_weighted_f1 | single_balanced_acc | single_accuracy |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 73 | 42 | 0.5865 | 0.6295 | 0.5761 | 0.6575 | 0.6433 | 0.6725 | 0.6272 | 0.6849 |
| 1 | 73 | 37 | 0.5186 | 0.5739 | 0.5011 | 0.6027 | 0.4905 | 0.5321 | 0.4762 | 0.5479 |
| 2 | 73 | 39 | 0.6730 | 0.6744 | 0.6383 | 0.6849 | 0.6420 | 0.6494 | 0.6184 | 0.6712 |
| 3 | 73 | 34 | 0.6405 | 0.6805 | 0.6196 | 0.6986 | 0.5671 | 0.5991 | 0.5565 | 0.6027 |
| 4 | 72 | 46 | 0.6792 | 0.7082 | 0.6469 | 0.7222 | 0.6210 | 0.6275 | 0.6136 | 0.6250 |

## Selected members — affect3

| fold | n_members | top_members | weight_window |
|---|---|---|---|
| 0 | 42 | audioglobal|raw|logreg:0.15 | audio+video|raw|lgbm:0.12 | audio+video|raw|svc_rbf:0.06 | audio|raw|svc_rbf:0.06 | audio+video|raw|mlp:0.06 | 0.0260 |
| 1 | 37 | win-raw|mean|extratrees:0.12 | win-raw|mean|xgb:0.10 | audio+video|raw|extratrees:0.06 | video|raw|extratrees:0.06 | win-raw|mean|lgbm:0.06 | 0.2800 |
| 2 | 39 | audio|raw|logreg:0.08 | audioglobal|raw|extratrees:0.07 | win-raw|mean|xgb:0.06 | audio+video|raw|svc_rbf:0.06 | win-raw|mean|extratrees:0.05 | 0.1220 |
| 3 | 34 | win-raw|mean|extratrees:0.16 | win-raw|mean|lgbm:0.12 | phys+audio|raw|extratrees:0.08 | phys|raw|extratrees:0.05 | phys+audio|raw|lgbm:0.04 | 0.3030 |
| 4 | 46 | audio|raw|mlp:0.09 | phys+audio|raw|svc_rbf:0.07 | win-raw|mean|lgbm:0.07 | win-raw|mean|extratrees:0.05 | audio|raw|extratrees:0.05 | 0.1560 |

## Inner-CV ranking — affect3

| candidate | inner_score |
|---|---|
| win-raw|mean|lgbm | 0.5686 |
| win-raw|mean|extratrees | 0.5660 |
| win-raw|mean|xgb | 0.5635 |
| audio+video|raw|lgbm | 0.5424 |
| all|raw|lgbm | 0.5401 |
| all+avail|raw|lgbm | 0.5401 |
| audio+video|raw|svc_rbf | 0.5370 |
| audio|raw|lgbm | 0.5283 |
| phys+audio|raw|extratrees | 0.5277 |
| all|raw|svc_rbf | 0.5272 |
| all+avail|raw|svc_rbf | 0.5272 |
| audio|raw|logreg | 0.5267 |
| all|raw|xgb | 0.5217 |
| audio+video|raw|logreg | 0.5202 |
| phys+audio|raw|lgbm | 0.5199 |
| all|raw|logreg | 0.5181 |
| all+avail|raw|logreg | 0.5181 |
| all|raw|mlp | 0.5162 |
| all+avail|raw|xgb | 0.5155 |
| audio+video|raw|xgb | 0.5127 |

## Config

```json
{
  "task": "affect3",
  "protocol": "subject-shared repeated K-fold (paper-style)",
  "n_folds": 5,
  "repeats": 1,
  "seed": 101,
  "views": [
    "raw",
    "rel"
  ],
  "scope": "c364",
  "window_views": [
    "raw",
    "rel"
  ],
  "models": [
    "logreg",
    "svc_rbf",
    "rf",
    "extratrees",
    "mlp",
    "lgbm",
    "xgb"
  ],
  "cum_keep": 0.9,
  "inner_folds": 3,
  "feature_version": 3,
  "selection": "bagged greedy w/ replacement on inner OOF"
}
```

## Environment

- Windows-10-10.0.26200-SP0
- Python 3.11.9
