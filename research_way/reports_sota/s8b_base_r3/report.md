# SOTA run — `s8b_base_r3`

**Generated:** 2026-08-15 14:52:04  
**Primary (all700_macro_f1):** **0.7322**  
**Best previous:** `s5_final` at 0.7604 → **no improvement** (-0.0282)  
**Duration:** 3398 s (56.6 min)  

> **Protocol.** Subject-shared repeated stratified CV — the split rule
> the StressID origin paper uses (random 80/20 + SMOTE). Subjects appear
> on both sides. Numbers here are comparable to published ones and are
> **not** comparable to the GroupKFold track in `reports/`, which holds
> subjects out. Origin paper's best reported weighted F1: 0.72.

## What changed

R8b: R1 baseline configuration at 3 repeats (15 outer folds), same seed and partitions as R8a.

## Headline metrics

| metric | value |
|---|---|
| all700_macro_f1 | 0.7322 |
| all700_macro_f1_std | 0.0301 |
| all700_weighted_f1 | 0.7332 |
| all700_weighted_f1_std | 0.0295 |
| all700_balanced_acc | 0.7328 |
| all700_balanced_acc_std | 0.0302 |
| all700_accuracy | 0.7343 |
| all700_accuracy_std | 0.0287 |
| all700_roc_auc | 0.8035 |
| all700_roc_auc_std | 0.0333 |
| all700_single_macro_f1 | 0.7226 |
| all700_unpruned_macro_f1 | 0.7322 |
| all700_mean_members | 45.6667 |
| all700_mean_members_unpruned | 45.6667 |
| all700_n_eval_folds | 15 |
| all700_n_recordings | 700 |

## Per-fold — all700

| fold | n_test | thr | n_members | n_members_full | macro_f1 | weighted_f1 | balanced_acc | accuracy | roc_auc | unpruned_macro_f1 | unpruned_weighted_f1 | unpruned_balanced_acc | unpruned_accuracy | unpruned_roc_auc | single_macro_f1 | single_weighted_f1 | single_balanced_acc | single_accuracy | single_roc_auc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 140 | 0.5100 | 45 | 45 | 0.7424 | 0.7429 | 0.7424 | 0.7429 | 0.8467 | 0.7424 | 0.7429 | 0.7424 | 0.7429 | 0.8467 | 0.7277 | 0.7283 | 0.7275 | 0.7286 | 0.8315 |
| 1 | 140 | 0.4700 | 51 | 51 | 0.7453 | 0.7468 | 0.7456 | 0.7500 | 0.7870 | 0.7453 | 0.7468 | 0.7456 | 0.7500 | 0.7870 | 0.6893 | 0.6907 | 0.6895 | 0.6929 | 0.7557 |
| 2 | 140 | 0.5400 | 40 | 40 | 0.7357 | 0.7356 | 0.7385 | 0.7357 | 0.8092 | 0.7357 | 0.7356 | 0.7385 | 0.7357 | 0.8092 | 0.6417 | 0.6405 | 0.6482 | 0.6429 | 0.7126 |
| 3 | 140 | 0.5100 | 49 | 49 | 0.6843 | 0.6873 | 0.6857 | 0.6929 | 0.7318 | 0.6843 | 0.6873 | 0.6857 | 0.6929 | 0.7318 | 0.6909 | 0.6939 | 0.6925 | 0.7000 | 0.7310 |
| 4 | 140 | 0.4800 | 37 | 37 | 0.7283 | 0.7288 | 0.7293 | 0.7286 | 0.8077 | 0.7283 | 0.7288 | 0.7293 | 0.7286 | 0.8077 | 0.6999 | 0.7002 | 0.7015 | 0.7000 | 0.7833 |
| 5 | 140 | 0.5200 | 32 | 32 | 0.7497 | 0.7501 | 0.7498 | 0.7500 | 0.7917 | 0.7497 | 0.7501 | 0.7498 | 0.7500 | 0.7917 | 0.7494 | 0.7499 | 0.7492 | 0.7500 | 0.7836 |
| 6 | 140 | 0.5200 | 51 | 51 | 0.7567 | 0.7571 | 0.7567 | 0.7571 | 0.8197 | 0.7567 | 0.7571 | 0.7567 | 0.7571 | 0.8197 | 0.7286 | 0.7286 | 0.7299 | 0.7286 | 0.7851 |
| 7 | 140 | 0.5100 | 42 | 42 | 0.7173 | 0.7192 | 0.7168 | 0.7214 | 0.7727 | 0.7173 | 0.7192 | 0.7168 | 0.7214 | 0.7727 | 0.7240 | 0.7260 | 0.7236 | 0.7286 | 0.7686 |
| 8 | 140 | 0.4900 | 48 | 48 | 0.7428 | 0.7426 | 0.7461 | 0.7429 | 0.8237 | 0.7428 | 0.7426 | 0.7461 | 0.7429 | 0.8237 | 0.7277 | 0.7286 | 0.7277 | 0.7286 | 0.8142 |
| 9 | 140 | 0.4700 | 51 | 51 | 0.7095 | 0.7116 | 0.7093 | 0.7143 | 0.8092 | 0.7095 | 0.7116 | 0.7093 | 0.7143 | 0.8092 | 0.7105 | 0.7124 | 0.7101 | 0.7143 | 0.7957 |
| 10 | 140 | 0.4500 | 53 | 53 | 0.7258 | 0.7270 | 0.7256 | 0.7286 | 0.8460 | 0.7258 | 0.7270 | 0.7256 | 0.7286 | 0.8460 | 0.7410 | 0.7419 | 0.7405 | 0.7429 | 0.8250 |
| 11 | 140 | 0.5100 | 52 | 52 | 0.7211 | 0.7215 | 0.7212 | 0.7214 | 0.7915 | 0.7211 | 0.7215 | 0.7212 | 0.7214 | 0.7915 | 0.7642 | 0.7644 | 0.7648 | 0.7643 | 0.7931 |
| 12 | 140 | 0.5200 | 45 | 45 | 0.7068 | 0.7074 | 0.7074 | 0.7071 | 0.7979 | 0.7068 | 0.7074 | 0.7074 | 0.7071 | 0.7979 | 0.7122 | 0.7136 | 0.7117 | 0.7143 | 0.7717 |
| 13 | 140 | 0.5400 | 37 | 37 | 0.8139 | 0.8144 | 0.8145 | 0.8143 | 0.8552 | 0.8139 | 0.8144 | 0.8145 | 0.8143 | 0.8552 | 0.7911 | 0.7922 | 0.7901 | 0.7929 | 0.8335 |
| 14 | 140 | 0.4800 | 52 | 52 | 0.7037 | 0.7056 | 0.7033 | 0.7071 | 0.7621 | 0.7037 | 0.7056 | 0.7033 | 0.7071 | 0.7621 | 0.7415 | 0.7426 | 0.7412 | 0.7429 | 0.7743 |

## Selected ensemble members — all700

| fold | thr | inner_blend_f1 | top_members | weight_recording | weight_window | weight_sequence |
|---|---|---|---|---|---|---|
| 0 | 0.5100 | 0.7419 | all+avail|raw|xgb:0.20 | audio+video|raw|rf:0.05 | video|raw|rf:0.05 | all|raw|lgbm:0.05 | all|raw|rf:0.05 | phys+video|raw|rf:0.04 | 1.0000 | 0 | 0 |
| 1 | 0.4700 | 0.7379 | video|raw|extratrees:0.08 | phys+audio|raw|rf:0.07 | phys+video|raw|rf:0.05 | phys+audio|raw|extratrees:0.05 | phys|raw|rf:0.04 | phys+audio|raw|svc_rbf:0.04 | 1.0000 | 0 | 0 |
| 2 | 0.5400 | 0.7393 | audio+video|raw|logreg_l2w:0.13 | phys+audio|raw|svc_rbf:0.10 | phys+audio|raw|mlp:0.06 | phys|raw|rf:0.05 | phys+audio|raw|extratrees:0.05 | all+avail|raw|svc_rbf:0.04 | 1.0000 | 0 | 0 |
| 3 | 0.5100 | 0.7709 | phys+video|raw|extratrees:0.09 | video|raw|extratrees:0.08 | phys|raw|rf:0.07 | phys+audio|raw|extratrees:0.05 | phys+video|raw|rf:0.05 | all|raw|xgb:0.04 | 1.0000 | 0 | 0 |
| 4 | 0.4800 | 0.7504 | all+avail|raw|rf:0.15 | phys+audio|raw|extratrees:0.14 | phys+audio|raw|rf:0.08 | all|raw|extratrees:0.07 | all+avail|raw|lgbm:0.05 | all+avail|raw|extratrees:0.05 | 1.0000 | 0 | 0 |
| 5 | 0.5200 | 0.7363 | all+avail|raw|extratrees:0.45 | all|raw|extratrees:0.08 | all|raw|rf:0.07 | all+avail|raw|rf:0.04 | audio+video|raw|rf:0.04 | video|raw|extratrees:0.04 | 1.0000 | 0 | 0 |
| 6 | 0.5200 | 0.7570 | all|raw|rf:0.14 | audio+video|raw|lgbm:0.13 | phys+video|raw|mlp:0.05 | audio+video|raw|xgb:0.04 | phys+video|raw|extratrees:0.04 | all|raw|svc_rbf:0.04 | 1.0000 | 0 | 0 |
| 7 | 0.5100 | 0.7581 | all+avail|raw|xgb:0.14 | all|raw|xgb:0.12 | all|raw|rf:0.07 | phys+video|raw|rf:0.06 | phys|raw|rf:0.05 | audio+video|raw|rf:0.05 | 1.0000 | 0 | 0 |
| 8 | 0.4900 | 0.7620 | audio+video|raw|extratrees:0.11 | audio+video|raw|xgb:0.08 | all|raw|extratrees:0.07 | audio+video|raw|rf:0.07 | video|raw|rf:0.05 | all|raw|rf:0.05 | 1.0000 | 0 | 0 |
| 9 | 0.4700 | 0.7428 | all+avail|raw|lgbm:0.18 | audio+video|raw|mlp:0.09 | all|raw|rf:0.07 | phys|raw|svc_rbf:0.07 | phys|raw|logreg_l2w:0.05 | all|raw|extratrees:0.04 | 1.0000 | 0 | 0 |
| 10 | 0.4500 | 0.7332 | all|raw|lgbm:0.12 | video|raw|rf:0.07 | all+avail|raw|xgb:0.07 | phys+video|raw|rf:0.05 | audio+video|raw|lgbm:0.05 | video|raw|mlp:0.04 | 1.0000 | 0 | 0 |
| 11 | 0.5100 | 0.7619 | phys+video|raw|extratrees:0.08 | all|raw|extratrees:0.08 | phys+audio|raw|extratrees:0.07 | all|raw|rf:0.05 | all+avail|raw|rf:0.05 | phys+video|raw|mlp:0.04 | 1.0000 | 0 | 0 |
| 12 | 0.5200 | 0.7580 | all+avail|raw|lgbm:0.17 | audio+video|raw|xgb:0.09 | all+avail|raw|rf:0.07 | phys+video|raw|rf:0.07 | all|raw|mlp:0.05 | phys+audio|raw|lgbm:0.05 | 1.0000 | 0 | 0 |
| 13 | 0.5400 | 0.7249 | all+avail|raw|extratrees:0.20 | phys+audio|raw|extratrees:0.08 | all|raw|extratrees:0.05 | video|raw|extratrees:0.05 | all|raw|rf:0.05 | all+avail|raw|rf:0.05 | 1.0000 | 0 | 0 |
| 14 | 0.4800 | 0.7567 | phys+audio|raw|extratrees:0.16 | phys|raw|mlp:0.06 | phys+video|raw|rf:0.05 | all+avail|raw|extratrees:0.04 | video|raw|rf:0.03 | all|raw|svc_rbf:0.03 | 1.0000 | 0 | 0 |

## Inner-CV candidate ranking — all700

| candidate | inner_macro_f1 |
|---|---|
| all+avail|raw|extratrees | 0.7305 |
| phys+audio|raw|extratrees | 0.7299 |
| all|raw|extratrees | 0.7293 |
| all|raw|lgbm | 0.7273 |
| all|raw|rf | 0.7258 |
| all+avail|raw|lgbm | 0.7251 |
| all+avail|raw|rf | 0.7234 |
| audio+video|raw|extratrees | 0.7216 |
| phys+audio|raw|rf | 0.7205 |
| all+avail|raw|svc_rbf | 0.7183 |
| all|raw|svc_rbf | 0.7183 |
| all|raw|xgb | 0.7178 |
| audio+video|raw|lgbm | 0.7176 |
| audio+video|raw|svc_rbf | 0.7174 |
| all+avail|raw|xgb | 0.7162 |
| audio|raw|rf | 0.7152 |
| audio+video|raw|rf | 0.7152 |
| phys+audio|raw|lgbm | 0.7148 |
| audio|raw|lgbm | 0.7127 |
| audio|raw|extratrees | 0.7126 |
| audio+video|raw|xgb | 0.7121 |
| audio|raw|xgb | 0.7107 |
| audio|raw|logreg_l2w | 0.7099 |
| phys+audio|raw|xgb | 0.7061 |
| audio|raw|svc_rbf | 0.7051 |

## Config

```json
{
  "protocol": "subject-shared RepeatedStratifiedKFold (paper-style)",
  "n_folds": 5,
  "repeats": 3,
  "seed": 101,
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
  "n_par": 1,
  "jobs": 6,
  "window_views": [],
  "torch_archs": [],
  "inner_folds": 3,
  "ensemble_cum_keep": 1.0,
  "selection": "bagged greedy w/ replacement on inner OOF; threshold tuned on inner OOF"
}
```

## Environment

- Windows-10-10.0.26200-SP0
- Python 3.11.9
