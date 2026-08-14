"""Candidates for the nested harness: recording-level, window-level, and GPU.

A *candidate* is anything that can turn (train rows, test rows) into a
probability per test recording. Wrapping them behind one interface is what lets
a 16x-larger window-level model and a plain recording-level random forest
compete on identical inner OOF folds and end up in the same ensemble — which is
the point, because they make different mistakes.

    cand.fit_predict(tr_idx, te_idx, y) -> probs over te_idx

Three kinds:

  TabularCandidate   one row per recording; the original setting.
  WindowCandidate    fits on window rows (~11 000 instead of 700, each carrying
                     its recording's label), predicts per window, then averages
                     back to a recording decision.
  TorchWindowCandidate
                     a masked sequence encoder on the GPU. Sees the window
                     sequence as a sequence rather than a bag, so it can use
                     the shape of the response over the task, not just its
                     average.
"""
from __future__ import annotations

import numpy as np

from .sota_env import HAVE_GPU, JOBS  # noqa: F401  (JOBS used by callers)


class TabularCandidate:
    """One feature vector per recording.

    The design matrix is referenced by *path*, not by value, and memory-mapped
    on first use. Holding it inline cost the campaign its first real run: with
    64 candidates each carrying a 700x1505 float64 array, a 4-process inner
    sweep pickled ~540 MB into workers and the OS killed them (6 GB free on a
    box already running two other training jobs). Memory-mapping makes the
    pickled candidate a few hundred bytes and lets every worker share one page
    cache for the same matrix.
    """

    def __init__(self, name: str, path, factory):
        self.name = name
        self.path = str(path)
        self.factory = factory
        self._X = None

    @property
    def X(self) -> np.ndarray:
        if self._X is None:
            self._X = np.load(self.path, mmap_mode="r")
        return self._X

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_X"] = None          # never pickle the matrix itself
        return d

    def fit_predict(self, tr, te, y) -> np.ndarray:
        X = self.X
        c = self.factory()
        c.fit(np.ascontiguousarray(X[tr]), y[tr])
        Xte = np.ascontiguousarray(X[te])
        if hasattr(c, "predict_proba"):
            return c.predict_proba(Xte)[:, 1]
        return 1.0 / (1.0 + np.exp(-c.decision_function(Xte)))


class WindowCandidate:
    """Fit on windows, decide on recordings.

    Each of a recording's valid windows becomes a training row carrying that
    recording's label. The label is noisier per window — not every second of a
    Stroop task is stressful — but there are 16x as many rows, and on 560
    training recordings that trade is what the tree learners are short of.

    Aggregation back to the recording is the *mean* of window probabilities
    rather than a vote: a task where two windows are confidently stressed and
    fourteen are ambiguous should not score the same as one where all sixteen
    are mildly stressed, and averaging probabilities keeps that distinction.
    The trimmed variant additionally drops the extreme quartiles, which helps
    when a single corrupted window would otherwise swing the recording.
    """

    def __init__(self, name: str, W: np.ndarray, V: np.ndarray, factory,
                 agg: str = "mean"):
        self.name = name
        self.W = W
        self.V = V
        self.factory = factory
        self.agg = agg

    def _expand(self, idx):
        rows, owner = [], []
        for j, i in enumerate(idx):
            v = np.where(self.V[i])[0]
            if len(v) == 0:
                v = np.array([0])
            rows.append(self.W[i][v])
            owner.append(np.full(len(v), j))
        return np.concatenate(rows), np.concatenate(owner)

    def fit_predict(self, tr, te, y) -> np.ndarray:
        Xtr, otr = self._expand(tr)
        ytr = y[tr][otr]
        c = self.factory()
        c.fit(Xtr, ytr)
        Xte, ote = self._expand(te)
        p = (c.predict_proba(Xte)[:, 1] if hasattr(c, "predict_proba")
             else 1.0 / (1.0 + np.exp(-c.decision_function(Xte))))
        out = np.zeros(len(te))
        for j in range(len(te)):
            q = p[ote == j]
            if len(q) == 0:
                out[j] = 0.5
            elif self.agg == "trimmed" and len(q) >= 4:
                lo, hi = np.percentile(q, [25, 75])
                out[j] = q[(q >= lo) & (q <= hi)].mean()
            else:
                out[j] = q.mean()
        return out


# --------------------------------------------------------------- torch model

class _SeqEncoder:
    """Lazy import wrapper so the module imports without torch installed."""


def _build_torch(d_in: int, d_model: int, dropout: float, arch: str):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Linear(d_in, d_model), nn.LayerNorm(d_model), nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.GELU(),
            )
            self.arch = arch
            if arch == "gru":
                self.rnn = nn.GRU(d_model, d_model // 2, batch_first=True,
                                  bidirectional=True)
            elif arch == "attn":
                self.att = nn.Linear(d_model, 1)
            self.head = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(d_model * 2, d_model), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(d_model, 1))

        def forward(self, x, m):
            # x [B, W, D], m [B, W] float mask
            h = self.enc(x) * m[..., None]
            if self.arch == "gru":
                h, _ = self.rnn(h)
                h = h * m[..., None]
            n = m.sum(1, keepdim=True).clamp(min=1)
            avg = h.sum(1) / n
            if self.arch == "attn":
                a = self.att(h).squeeze(-1).masked_fill(m < 0.5, -1e9)
                a = a.softmax(-1)
                mx = (h * a[..., None]).sum(1)
            else:
                mx = h.masked_fill(m[..., None] < 0.5, -1e9).max(1).values
                mx = torch.nan_to_num(mx, neginf=0.0)
            return self.head(torch.cat([avg, mx], -1)).squeeze(-1)

    return Net()


class TorchWindowCandidate:
    """Masked sequence encoder trained on the GPU.

    Deliberately small (a two-layer window encoder, ~130 k parameters at
    d_model=96). The lesson already recorded in this repo is that a 1.2 M
    -parameter transformer memorises 448 recordings rather than generalising;
    the fix is not a bigger model, it is a model sized to the data. Capacity
    goes into the temporal aggregation, which is the part the tabular
    candidates genuinely cannot express.

    A held-out slice of the training rows drives early stopping, so no test row
    influences when training ends.
    """

    def __init__(self, name: str, W: np.ndarray, V: np.ndarray, arch: str = "gru",
                 d_model: int = 96, dropout: float = 0.3, epochs: int = 120,
                 lr: float = 2e-3, batch: int = 64, seed: int = 0,
                 patience: int = 18):
        self.name = name
        self.W, self.V = W, V
        self.arch, self.d_model, self.dropout = arch, d_model, dropout
        self.epochs, self.lr, self.batch, self.seed = epochs, lr, batch, seed
        self.patience = patience

    def fit_predict(self, tr, te, y) -> np.ndarray:
        import torch
        import torch.nn as nn

        dev = torch.device("cuda" if HAVE_GPU else "cpu")
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        # standardise on TRAIN windows only
        flat = self.W[tr][self.V[tr]]
        mu, sd = flat.mean(0), flat.std(0) + 1e-6

        def prep(idx):
            x = (self.W[idx] - mu) / sd
            m = self.V[idx].astype(np.float32)
            return (torch.tensor(x * m[..., None], dtype=torch.float32),
                    torch.tensor(m))

        perm = rng.permutation(len(tr))
        n_val = max(8, int(0.15 * len(tr)))
        vi, ti = perm[:n_val], perm[n_val:]
        Xt, Mt = prep(tr[ti]); yt = torch.tensor(y[tr][ti], dtype=torch.float32)
        Xv, Mv = prep(tr[vi]); yv = torch.tensor(y[tr][vi], dtype=torch.float32)
        Xe, Me = prep(te)

        net = _build_torch(self.W.shape[-1], self.d_model, self.dropout,
                           self.arch).to(dev)
        pos = float(yt.mean().clamp(0.05, 0.95))
        lossf = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor((1 - pos) / pos, device=dev))
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

        Xt, Mt, yt = Xt.to(dev), Mt.to(dev), yt.to(dev)
        Xv, Mv, yv = Xv.to(dev), Mv.to(dev), yv.to(dev)
        Xe, Me = Xe.to(dev), Me.to(dev)

        best, best_state, bad = 1e9, None, 0
        for ep in range(self.epochs):
            net.train()
            order = torch.randperm(len(Xt), device=dev)
            for b in range(0, len(order), self.batch):
                s = order[b:b + self.batch]
                opt.zero_grad()
                loss = lossf(net(Xt[s], Mt[s]), yt[s])
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            sched.step()
            net.eval()
            with torch.no_grad():
                vl = float(lossf(net(Xv, Mv), yv))
            if vl < best - 1e-4:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            p = torch.sigmoid(net(Xe, Me)).cpu().numpy()
        del Xt, Mt, Xv, Mv, Xe, Me, net
        if HAVE_GPU:
            import torch as _t
            _t.cuda.empty_cache()
        return np.nan_to_num(p, nan=0.5)
