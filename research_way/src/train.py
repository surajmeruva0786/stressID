"""Training driver: GroupKFold x seeds, for the temporal model and its static twin.

For every trained fold model the test set is scored under all 7 modality
conditions (full / 1-missing x3 / 2-missing x3). Raw per-recording predictions
are written to a CSV so `evaluate.py` can build every table and figure in the
Stage 6 suite without retraining anything.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

from .checkpoint import checkpoint_path, save_checkpoint
from .config import Config
from .dataset import StressIDWindows, compute_norm, make_loader, load_manifest
from .losses import compute_loss
from .model import build_model
from .splits import load_splits

# name -> (physio, audio, video) keep-mask
MASK_CONDITIONS = {
    "full":        (1, 1, 1),
    "no_physio":   (0, 1, 1),
    "no_audio":    (1, 0, 1),
    "no_video":    (1, 1, 0),
    "physio_only": (1, 0, 0),
    "audio_only":  (0, 1, 0),
    "video_only":  (0, 0, 1),
}


def set_seed(s: int) -> None:
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def _to_device(batch: dict, dev: str) -> dict:
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def predict(model, loader, cfg: Config, force_mask=None) -> dict:
    model.eval()
    probs, l3, yb, ya, keys = [], [], [], [], []
    for batch in loader:
        b = _to_device(batch, cfg.device)
        out = model(b, force_mask=force_mask)
        probs.append(torch.sigmoid(out["logit_binary"]).cpu().numpy())
        l3.append(out["logit_affect3"].cpu().numpy())
        yb.append(batch["y_binary"].numpy())
        ya.append(batch["y_affect3"].numpy())
        keys.extend(batch["key"])
    return {"prob": np.concatenate(probs), "logit3": np.concatenate(l3),
            "y_binary": np.concatenate(yb), "y_affect3": np.concatenate(ya),
            "key": keys}


def train_one_fold(cfg: Config, man: pd.DataFrame, fold: dict, seed: int,
                   temporal: bool, verbose: bool = False,
                   save_ckpt: bool = True) -> tuple:
    set_seed(seed)
    subject_vocab = {s: i for i, s in enumerate(sorted(man["subject"].unique()))}
    norm = compute_norm(man, cfg, fold["train"])

    mk = lambda subs: StressIDWindows(man, cfg, subs, norm, subject_vocab)
    tr_ds, va_ds, te_ds = mk(fold["train"]), mk(fold["val"]), mk(fold["test"])
    tr, va, te = (make_loader(tr_ds, cfg, True), make_loader(va_ds, cfg, False),
                  make_loader(te_ds, cfg, False))

    model = build_model(cfg, n_tasks=len(cfg.tasks), n_subjects=len(subject_vocab),
                        temporal=temporal).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.epochs * max(len(tr), 1), pct_start=0.3)

    ytr = tr_ds.man["binary"].values
    pos_w = torch.tensor([(len(ytr) - ytr.sum()) / max(ytr.sum(), 1)],
                         dtype=torch.float32, device=cfg.device)
    cnt = np.bincount(tr_ds.man["affect3"].values, minlength=3).astype(float)
    cls_w = torch.tensor(len(tr_ds.man) / (3 * np.maximum(cnt, 1)),
                         dtype=torch.float32, device=cfg.device)

    best = {"f1": -1.0, "state": None, "epoch": -1}
    from sklearn.metrics import f1_score
    for ep in range(cfg.epochs):
        model.train()
        agg = {}
        for batch in tr:
            b = _to_device(batch, cfg.device)
            out = model(b)
            loss, parts = compute_loss(out, b, cfg, pos_w, cls_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v / max(len(tr), 1)

        p = predict(model, va, cfg)
        f1 = f1_score(p["y_binary"], (p["prob"] >= 0.5).astype(int),
                      average="macro", zero_division=0)
        if f1 > best["f1"]:
            best = {"f1": f1, "epoch": ep,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        if verbose and (ep % 10 == 0 or ep == cfg.epochs - 1):
            print(f"      ep{ep:3d} loss={agg.get('loss', 0):.3f} "
                  f"bce={agg.get('bce', 0):.3f} inv={agg.get('subj_inv', 0):.3f} "
                  f"valF1={f1:.3f}", flush=True)

    model.load_state_dict(best["state"])

    if save_ckpt:
        variant = "temporal" if temporal else "static"
        best["ckpt"] = save_checkpoint(
            checkpoint_path(cfg, variant, seed, fold["fold"]),
            state=best["state"], cfg=cfg, variant=variant, seed=seed,
            fold=fold["fold"], best_epoch=best["epoch"], best_val_f1=best["f1"],
            subject_vocab=subject_vocab, norm=norm,
            fold_subjects={k: list(fold[k]) for k in ("train", "val", "test")},
        )

    rows = []
    key2meta = man.set_index("key")
    for cond, msk in MASK_CONDITIONS.items():
        p = predict(model, te, cfg, force_mask=msk)
        for i, k in enumerate(p["key"]):
            meta = key2meta.loc[k]
            rows.append({
                "key": k, "subject": meta["subject"], "task": meta["task"],
                "condition": cond, "prob": float(p["prob"][i]),
                "logit3_0": float(p["logit3"][i, 0]), "logit3_1": float(p["logit3"][i, 1]),
                "logit3_2": float(p["logit3"][i, 2]),
                "y_binary": int(p["y_binary"][i]), "y_affect3": int(p["y_affect3"][i]),
                "nat_physio": int(meta["has_physio"]), "nat_audio": int(meta["has_audio"]),
                "nat_video": int(meta["has_video"]),
            })
    return pd.DataFrame(rows), best


def run(cfg: Config | None = None, variants=("temporal", "static"),
        verbose: bool = False) -> pd.DataFrame:
    cfg = cfg or Config()
    if cfg.device == "cuda" and not torch.cuda.is_available():
        cfg.device = "cpu"
        print("[train] CUDA unavailable -> CPU")

    man = load_manifest(cfg)
    folds = load_splits(cfg)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(cfg.results_dir / "config.json")

    all_rows, log = [], []
    t0 = time.time()
    for variant in variants:
        temporal = variant == "temporal"
        for seed in cfg.seeds:
            for fold in folds:
                ts = time.time()
                df, best = train_one_fold(cfg, man, fold, seed, temporal, verbose)
                df["variant"] = variant
                df["seed"] = seed
                df["fold"] = fold["fold"]
                all_rows.append(df)
                msg = (f"[train] {variant:<8} seed={seed} fold={fold['fold']} "
                       f"bestValF1={best['f1']:.3f}@ep{best['epoch']} "
                       f"({time.time() - ts:.0f}s)")
                print(msg, flush=True)
                log.append(msg)

    preds = pd.concat(all_rows, ignore_index=True)
    out = cfg.results_dir / "predictions.csv"
    preds.to_csv(out, index=False)
    (cfg.results_dir / "train_log.txt").write_text("\n".join(log), encoding="utf-8")
    print(f"[train] done in {time.time() - t0:.0f}s -> {out}  ({len(preds)} rows)")
    return preds


if __name__ == "__main__":
    run(verbose=True)
