# SOTA run — `t2_score`

**Generated:** 2026-08-15 20:24:51  
**Primary (all700_macro_f1):** **nan**  
**Best previous:** `s5_final` at 0.7604 → **no improvement** (+nan)  
**Duration:** 5609 s (93.5 min)  

> **Protocol.** Subject-shared repeated stratified CV — the split rule
> the StressID origin paper uses (random 80/20 + SMOTE). Subjects appear
> on both sides. Numbers here are comparable to published ones and are
> **not** comparable to the GroupKFold track in `reports/`, which holds
> subjects out. Origin paper's best reported weighted F1: 0.72.

## What changed

T2: stress-score regression, 0-10 continuous. Same recipe; greedy maximises negative RMSE. Reported with RMSE, MAE, Pearson and Spearman correlation, and R2.

## Headline metrics

| metric | value |
|---|---|
| regression_rmse | 1.8570 |
| regression_rmse_std | 0.1065 |
| regression_mae | 1.5027 |
| regression_mae_std | 0.1064 |
| regression_pearson_r | 0.6732 |
| regression_pearson_r_std | 0.0298 |
| regression_spearman_r | 0.6598 |
| regression_spearman_r_std | 0.0309 |
| regression_r2 | 0.4395 |
| regression_r2_std | 0.0395 |
| regression_single_rmse | 1.8948 |
| regression_n_folds | 5 |
| regression_n_recordings | 700 |

## Per-fold — regression

| fold | n_test | n_members | rmse | mae | pearson_r | spearman_r | r2 | single_rmse | single_mae | single_pearson_r | single_spearman_r | single_r2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 140 | 10 | 1.7880 | 1.3976 | 0.6621 | 0.6378 | 0.4266 | 1.8247 | 1.4101 | 0.6401 | 0.6238 | 0.4028 |
| 1 | 140 | 15 | 1.9409 | 1.5779 | 0.6688 | 0.6614 | 0.4390 | 1.9539 | 1.5125 | 0.6601 | 0.6590 | 0.4315 |
| 2 | 140 | 12 | 1.8417 | 1.4785 | 0.6818 | 0.6607 | 0.4557 | 1.9891 | 1.5741 | 0.6050 | 0.6078 | 0.3651 |
| 3 | 140 | 18 | 1.7282 | 1.4145 | 0.7173 | 0.7091 | 0.4920 | 1.7317 | 1.3946 | 0.7099 | 0.6985 | 0.4899 |
| 4 | 140 | 17 | 1.9864 | 1.6449 | 0.6358 | 0.6297 | 0.3841 | 1.9746 | 1.6194 | 0.6326 | 0.6316 | 0.3915 |

## Selected members — regression

| fold | n_members | top_members | weight_window |
|---|---|---|---|
| 0 | 10 | win-raw|mean|lgbm:0.32 | win-raw|mean|extratrees:0.15 | all+avail|rel|lgbm:0.12 | physglobal|rel|lgbm:0.11 | win-raw|mean|xgb:0.09 | 0.5540 |
| 1 | 15 | win-raw|mean|xgb:0.15 | win-raw|mean|lgbm:0.14 | phys+video|rel|xgb:0.11 | all+avail|rel|lgbm:0.09 | video|rel|lgbm:0.07 | 0.3300 |
| 2 | 12 | all|raw|lgbm:0.18 | phys+video|rel|xgb:0.17 | win-raw|mean|xgb:0.14 | win-raw|mean|lgbm:0.12 | audio+video|raw|xgb:0.08 | 0.2610 |
| 3 | 18 | phys+video|rel|lgbm:0.20 | win-raw|mean|lgbm:0.14 | physglobal|rel|svr_rbf:0.11 | win-raw|mean|xgb:0.09 | phys+audio|raw|lgbm:0.06 | 0.2810 |
| 4 | 17 | phys+audio|rel|xgb:0.15 | win-raw|mean|extratrees:0.12 | audio+video|raw|lgbm:0.12 | win-raw|mean|lgbm:0.08 | audio+video|raw|xgb:0.08 | 0.2830 |

## Inner-CV ranking — regression

| candidate | inner_score |
|---|---|
| win-raw|mean|extratrees | -1.9555 |
| all+avail|rel|extratrees | -1.9756 |
| win-raw|mean|xgb | -1.9757 |
| win-raw|mean|lgbm | -1.9769 |
| all+avail|rel|lgbm | -1.9780 |
| all|raw|lgbm | -1.9787 |
| all|rel|extratrees | -1.9815 |
| all+avail|raw|xgb | -1.9846 |
| all|raw|xgb | -1.9855 |
| all|rel|xgb | -1.9872 |
| all+avail|rel|xgb | -1.9891 |
| all+avail|raw|lgbm | -1.9893 |
| all+avail|rel|rf | -1.9898 |
| all+avail|rel|svr_rbf | -1.9901 |
| all|rel|svr_rbf | -1.9936 |
| all|rel|lgbm | -1.9940 |
| all|rel|rf | -1.9941 |
| all|raw|extratrees | -1.9953 |
| all+avail|raw|extratrees | -1.9957 |
| phys+audio|rel|extratrees | -1.9968 |

## Config

```json
{
  "task": "regression",
  "protocol": "subject-shared repeated K-fold (paper-style)",
  "n_folds": 5,
  "repeats": 1,
  "seed": 101,
  "views": [
    "raw",
    "rel"
  ],
  "window_views": [
    "raw",
    "rel"
  ],
  "models": [
    "ridge",
    "svr_rbf",
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
