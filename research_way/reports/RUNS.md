# Run registry

Primary metric: `complete364_macro_f1` (macro F1, 364 all-modality recordings, subject GroupKFold). Majority reference 0.418.

Full narrative for every run: [RUN_LOG.md](../RUN_LOG.md)

| run | complete364_macro_f1 | vs majority | verdict | when |
|---|---|---|---|---|
| `c5_confirm_seed202` | 0.5499 | +0.1319 | NEW BEST | 2026-08-03 15:26:17 |
| `c1_nested_ensemble` | 0.5327 | +0.1147 | FIRST RUN | 2026-08-03 14:15:22 |
| `c2_subject_relative` | 0.5199 | +0.1019 | no improvement | 2026-08-03 14:20:37 |
| `c4b_video_features_fixedk` | 0.5100 | +0.0920 | no improvement | 2026-08-03 15:16:02 |
| `c5_confirm_seed303` | 0.5043 | +0.0863 | no improvement | 2026-08-03 15:31:47 |
| `c5_confirm_seed101` | 0.5035 | +0.0855 | no improvement | 2026-08-03 15:21:32 |
| `c4_video_features` | 0.4979 | +0.0799 | no improvement | 2026-08-03 15:06:40 |
| `c3_inner_selected_k` | 0.4898 | +0.0718 | no improvement | 2026-08-03 14:28:17 |
| `deep_full_temporal` | 0.4850 | +0.0670 | backfilled | 2026-07-30 22:38:00 |
| `deep_full_static` | 0.4759 | +0.0579 | backfilled | 2026-07-30 22:38:00 |
| `deep_a1a2_static` | 0.4752 | +0.0572 | backfilled | 2026-08-02 19:33:00 |
| `deep_a1a2_temporal` | 0.4439 | +0.0259 | backfilled | 2026-08-02 19:33:00 |

**Reportable result: 0.5192** — mean of the three `c5_confirm_*` runs on
subject partitions never used during the search. The search maximum (0.5327)
and the best single partition (0.5499) are optimistically biased and must not
be quoted alone. See RUN_LOG.md §3.7 and §7.4.
