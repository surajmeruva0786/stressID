"""Persist and restore trained fold models.

`train.py` keeps only the best-val-F1 state in memory and writes predictions to
CSV, which is enough for `evaluate.py` but throws the weights away. This module
writes a self-contained checkpoint per fold model: the weights plus everything
needed to rebuild the network and reproduce the input pipeline (config, task /
subject vocabularies, and the TRAIN-only audio/video normalisation stats).

    from src.checkpoint import load_model, list_checkpoints

    model, meta = load_model("results/full/checkpoints/temporal_seed0_fold0.pt")
    model.eval()

The normalisation stats in `meta["norm"]` are fold-specific (computed from that
fold's training subjects only). Applying a different fold's stats at inference
reintroduces the leakage the split was designed to avoid, so always use the ones
that ship with the checkpoint -- see `apply_norm`.
"""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .model import build_model

CKPT_VERSION = 1


def checkpoint_dir(cfg: Config) -> Path:
    return cfg.results_dir / "checkpoints"


def checkpoint_path(cfg: Config, variant: str, seed: int, fold: int) -> Path:
    return checkpoint_dir(cfg) / f"{variant}_seed{seed}_fold{fold}.pt"


def save_checkpoint(path: Path, *, state: dict, cfg: Config, variant: str,
                    seed: int, fold: int, best_epoch: int, best_val_f1: float,
                    subject_vocab: dict, norm: dict, fold_subjects: dict) -> Path:
    """Write one fold model. `state` is the best-val-F1 state_dict (on CPU)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "ckpt_version": CKPT_VERSION,
        "state_dict": state,
        "config": {f.name: getattr(cfg, f.name) for f in fields(cfg)},
        "variant": variant,
        "temporal": variant == "temporal",
        "seed": seed,
        "fold": fold,
        "best_epoch": best_epoch,
        "best_val_f1": float(best_val_f1),
        "subject_vocab": subject_vocab,
        "n_subjects": len(subject_vocab),
        "n_tasks": len(cfg.tasks),
        "tasks": list(cfg.tasks),
        "norm": norm,
        "fold_subjects": fold_subjects,
    }, path)
    return path


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    """Raw checkpoint dict, no model construction."""
    # weights_only=False: the payload carries config/vocab metadata, not just tensors.
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def load_model(path: str | Path, device: str | None = None) -> tuple:
    """Rebuild the network with its trained weights.

    Returns `(model, meta)` where `meta` is the checkpoint minus the weights.
    The model is returned in eval mode on `device` (default: the device recorded
    in the checkpoint, falling back to CPU when CUDA is unavailable).
    """
    ck = load_checkpoint(path)
    cfg = Config(**ck["config"])
    dev = device or cfg.device
    if dev == "cuda" and not torch.cuda.is_available():
        dev = "cpu"

    model = build_model(cfg, n_tasks=ck["n_tasks"], n_subjects=ck["n_subjects"],
                        temporal=ck["temporal"])
    model.load_state_dict(ck["state_dict"])
    model.to(dev).eval()

    meta = {k: v for k, v in ck.items() if k != "state_dict"}
    meta["cfg"] = cfg
    meta["device"] = dev
    return model, meta


def apply_norm(audio: np.ndarray, video: np.ndarray, norm: dict) -> tuple:
    """Standardise audio/video exactly as `StressIDWindows` does at train time."""
    if "audio_mean" in norm:
        audio = (audio - norm["audio_mean"]) / norm["audio_std"]
    if "video_mean" in norm:
        video = (video - norm["video_mean"]) / norm["video_std"]
    return audio, video


def list_checkpoints(cfg: Config) -> list[Path]:
    d = checkpoint_dir(cfg)
    return sorted(d.glob("*.pt")) if d.exists() else []
