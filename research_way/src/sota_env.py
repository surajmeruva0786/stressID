"""Compute-budget constants, shared by the runner and the candidate classes.

Lives in its own module only to break the import cycle: `sota.py` builds
candidates from `sota_models.py`, and those candidates need to know the same
core count and GPU status the runner does.

The budget is deliberately not "everything available". This box concurrently
runs the user's other training jobs, and it has a single 4 GB Quadro P1000, so:

  JOBS      6 of 12 cores. A 12-way tree fit on a 1500-column matrix with 6 GB
            of free RAM is also the fastest route to swapping.
  HAVE_GPU  gates XGBoost onto the card (12 s -> 6 s per fit on the real design
            matrix, measured with the GPU already 95% busy) and enables the
            torch sequence candidates.

Override the core count with the SOTA_JOBS environment variable.
"""
from __future__ import annotations

import os

JOBS = int(os.environ.get("SOTA_JOBS", "6"))


def gpu_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


HAVE_GPU = gpu_available()
