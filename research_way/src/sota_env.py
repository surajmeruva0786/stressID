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

# Every worker process inherits these. Without them each of the 4 sweep workers
# would spin up its own BLAS thread pool sized to all 12 cores, and the
# oversubscription costs more than the parallelism gains.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")


def gpu_available() -> bool:
    """Detect CUDA once, in the parent, and publish the answer via the env.

    Worker processes must NOT reach the torch import here. Each `import torch`
    costs ~500 MB of resident memory, and four sweep workers doing it killed
    two R1 runs on a box with 1.7 GB free (the user's other training jobs hold
    the rest). Workers inherit SOTA_GPU from the parent and answer from the
    string, so only the parent ever loads torch -- and only when a torch
    candidate is actually used.
    """
    cached = os.environ.get("SOTA_GPU")
    if cached is not None:
        return cached == "1"
    try:
        import torch
        ok = bool(torch.cuda.is_available())
    except Exception:
        ok = False
    os.environ["SOTA_GPU"] = "1" if ok else "0"
    return ok


HAVE_GPU = gpu_available()
