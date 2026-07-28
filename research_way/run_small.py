"""End-to-end driver for the SMALL-scale StressID pipeline.

    python run_small.py                 # full run: subset -> preprocess -> splits
                                        #           -> baselines -> train -> evaluate
    python run_small.py --stage train   # run a single stage
    python run_small.py --smoke         # 1 seed / 2 folds / 6 epochs (~2 min sanity run)

Stages implemented (mapped to the planning document):
    Stage  0  data subset + leakage-free GroupKFold splits + corrected baselines (E1)
    Stage  1  temporal windowing
    Stage  2  modality-specific encoders
    Stage  3  cross-modal attention fusion + modality dropout
    Stage  4  temporal aggregation
    Stage  5  multi-task output heads
    Stage  6  evaluation suite (E2, E4, E5, E6, E8, E12, E13)

Not implemented here (needs data / code this repo does not have):
    Stage -1  multi-dataset pretraining  (needs WESAD, SWELL-KW, K-EmoCon, ...)
    Stage 5.5 competitor re-implementation (E0)
    E7 / E10 / E11 (cross-dataset transfer + data-scaling)
"""
from __future__ import annotations

import argparse
import sys
import time

from src.config import Config
from src import subset, preprocess, splits, baselines, train, evaluate


STAGES = ("subset", "preprocess", "splits", "baselines", "train", "evaluate")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=STAGES + ("all",), default="all")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run to verify the pipeline end to end")
    ap.add_argument("--force", action="store_true", help="rebuild the preprocess cache")
    ap.add_argument("--subjects", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = Config()
    # NOTE: --smoke does not change n_folds, because the splits on disk are fixed
    # and shared by every experiment. It only shortens seeds/epochs.
    if args.smoke:
        cfg.seeds, cfg.epochs, cfg.run_name = (0,), 6, "smoke"
    if args.subjects:
        cfg.n_subjects = args.subjects
    if args.epochs:
        cfg.epochs = args.epochs
    if args.device:
        cfg.device = args.device

    todo = STAGES if args.stage == "all" else (args.stage,)
    t0 = time.time()
    for st in todo:
        print(f"\n{'=' * 72}\n== STAGE: {st}\n{'=' * 72}", flush=True)
        if st == "subset":
            subset.main(cfg)
        elif st == "preprocess":
            preprocess.main(cfg, force=args.force)
        elif st == "splits":
            splits.main(cfg)
        elif st == "baselines":
            baselines.run(cfg)
        elif st == "train":
            train.run(cfg, verbose=True)
        elif st == "evaluate":
            evaluate.run(cfg)
    print(f"\n[run_small] total {time.time() - t0:.0f}s | results -> {cfg.results_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
