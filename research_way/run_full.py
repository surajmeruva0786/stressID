"""End-to-end driver for the FULL-scale StressID pipeline.

Same stages as `run_small.py`, but over the entire corpus: all 64 subjects, all
11 tasks, 700 recordings, no modality requirement. Artefacts are namespaced
`full` (`data/manifest_full.csv`, `data/cache_full/`, `results/full/`) so the
small-subset run stays intact and comparable.

    python run_full.py                  # subset -> preprocess -> splits
                                        # -> baselines -> train -> evaluate
    python run_full.py --stage train    # run a single stage
    python run_full.py --workers 8      # preprocessing parallelism

Preprocessing the full corpus takes roughly an hour (video decode dominates);
it is cached per recording, so re-running only picks up what is missing.
"""
from __future__ import annotations

import argparse
import sys
import time

from src.config import full_config, a1a2_config
from src import subset, preprocess, splits, baselines, train, evaluate, physfeat_cache


STAGES = ("subset", "preprocess", "physfeat", "splits", "baselines", "train", "evaluate")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=STAGES + ("all",), default="all")
    ap.add_argument("--force", action="store_true", help="rebuild the preprocess cache")
    ap.add_argument("--workers", type=int, default=None,
                    help="preprocessing worker processes (default: min(6, cores))")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=None, help="number of seeds")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = full_config()
    if args.epochs:
        cfg.epochs = args.epochs
    if args.folds:
        cfg.n_folds = args.folds
    if args.seeds:
        cfg.seeds = tuple(range(args.seeds))
    if args.device:
        cfg.device = args.device

    todo = STAGES if args.stage == "all" else (args.stage,)
    t0 = time.time()
    for st in todo:
        print(f"\n{'=' * 72}\n== STAGE: {st}\n{'=' * 72}", flush=True)
        ts = time.time()
        if st == "subset":
            subset.main(cfg)
        elif st == "preprocess":
            preprocess.main(cfg, force=args.force, workers=args.workers)
        elif st == "splits":
            splits.main(cfg)
        elif st == "baselines":
            baselines.run(cfg)
        elif st == "train":
            train.run(cfg, verbose=True)
        elif st == "evaluate":
            evaluate.run(cfg)
        print(f"== stage {st} took {time.time() - ts:.0f}s", flush=True)
    print(f"\n[run_full] total {time.time() - t0:.0f}s | results -> {cfg.results_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
