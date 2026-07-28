"""Stages 2-5 of the pipeline.

  Stage 2  modality-specific encoders  (per window -> a few tokens)
  Stage 3  cross-modal attention fusion + modality dropout  (KEY CONTRIBUTION)
  Stage 4  temporal aggregation over the window sequence + task-type embedding
  Stage 5  output heads: binary stress / 3-class affect / stress score

`temporal=False` gives the STATIC counterpart used for the headline E12
ablation: identical encoders and fusion, but the window sequence is mean-pooled
instead of processed by a Transformer over time.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config

NEG = torch.finfo(torch.float32).min


# ============================================================ Stage 2 encoders

class PhysioEncoder(nn.Module):
    """ECG/EDA/RR -> 1D-CNN -> BiLSTM -> `n_tokens` tokens."""

    def __init__(self, cfg: Config, in_ch: int = 3):
        super().__init__()
        d = cfg.d_model
        self.cnn = nn.Sequential(
            nn.Conv1d(in_ch, 32, 15, stride=2, padding=7), nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, 11, stride=2, padding=5), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 96, 7, stride=2, padding=3), nn.BatchNorm1d(96), nn.GELU(),
            nn.Conv1d(96, d, 5, stride=2, padding=2), nn.BatchNorm1d(d), nn.GELU(),
        )
        self.rnn = nn.LSTM(d, d // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.pool = nn.AdaptiveAvgPool1d(cfg.tokens_per_modality)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # [N, T, C] -> [N, k, d]
        h = self.cnn(x.transpose(1, 2))                    # [N, d, T']
        h, _ = self.rnn(h.transpose(1, 2))                 # [N, T', d]
        h = self.pool(h.transpose(1, 2)).transpose(1, 2)
        return self.norm(h)


class AudioEncoder(nn.Module):
    """log-mel -> 1D-CNN over time -> `n_tokens` tokens.

    Drop-in slot for a fine-tuned wav2vec2 encoder (E9): swap this module for
    one that consumes the raw waveform and emits [N, k, d]."""

    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.d_model
        self.cnn = nn.Sequential(
            nn.Conv1d(cfg.n_mels, 64, 7, stride=2, padding=3), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 96, 5, stride=2, padding=2), nn.BatchNorm1d(96), nn.GELU(),
            nn.Conv1d(96, d, 3, stride=2, padding=1), nn.BatchNorm1d(d), nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(cfg.tokens_per_modality)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # [N, F, n_mels]
        h = self.cnn(x.transpose(1, 2))
        h = self.pool(h).transpose(1, 2)
        return self.norm(h)


class VideoEncoder(nn.Module):
    """Face-crop frames -> per-frame CNN -> Transformer over frames -> tokens.

    Drop-in slot for per-frame OpenFace AU vectors: replace `frame_cnn` with a
    linear projection of the AU vector and keep the frame Transformer."""

    def __init__(self, cfg: Config):
        super().__init__()
        d, s = cfg.d_model, cfg.video_face_size
        self.frame_cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, d),
        )
        self.pos = nn.Parameter(torch.randn(1, cfg.video_frames, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, cfg.n_heads, d * 2, cfg.dropout,
                                           batch_first=True, norm_first=True,
                                           activation="gelu")
        self.tr = nn.TransformerEncoder(layer, 1)
        self.pool = nn.AdaptiveAvgPool1d(cfg.tokens_per_modality)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # [N, Fv, S, S]
        n, f, s1, s2 = x.shape
        h = self.frame_cnn(x.reshape(n * f, 1, s1, s2)).reshape(n, f, -1)
        h = self.tr(h + self.pos[:, :f])
        h = self.pool(h.transpose(1, 2)).transpose(1, 2)
        return self.norm(h)


# ==================================================== Stage 3 cross-modal fusion

class CrossModalBlock(nn.Module):
    """Every modality queries the union of the OTHER modalities' tokens.

    A learned always-valid `null` key/value pair guarantees the attention
    softmax is never fully masked, which is what makes 1- and 2-modality
    inference numerically safe rather than a NaN.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.d_model
        self.attn = nn.ModuleDict({
            m: nn.MultiheadAttention(d, cfg.n_heads, cfg.dropout, batch_first=True)
            for m in ("physio", "audio", "video")})
        self.ln_q = nn.ModuleDict({m: nn.LayerNorm(d) for m in self.attn})
        self.ln_kv = nn.ModuleDict({m: nn.LayerNorm(d) for m in self.attn})
        self.ff = nn.ModuleDict({m: nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d * 2), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(d * 2, d)) for m in self.attn})
        self.null_kv = nn.Parameter(torch.randn(1, 1, d) * 0.02)

    def forward(self, tok: dict, key_pad: dict) -> dict:
        out = {}
        for m in tok:
            others = [o for o in tok if o != m]
            kv = torch.cat([tok[o] for o in others] + [
                self.null_kv.expand(tok[m].size(0), -1, -1)], dim=1)
            pad = torch.cat([key_pad[o] for o in others] + [
                torch.zeros(tok[m].size(0), 1, dtype=torch.bool, device=tok[m].device)],
                dim=1)
            a, _ = self.attn[m](self.ln_q[m](tok[m]), self.ln_kv[m](kv),
                                self.ln_kv[m](kv), key_padding_mask=pad,
                                need_weights=False)
            h = tok[m] + a
            out[m] = h + self.ff[m](h)
        return out


class FusionHead(nn.Module):
    """CLS token attends over all available tokens -> one vector per window."""

    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.d_model
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.mod_emb = nn.Parameter(torch.randn(3, 1, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, cfg.n_heads, cfg.dropout, batch_first=True)
        self.ln = nn.LayerNorm(d)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU())

    def forward(self, tok: dict, key_pad: dict) -> torch.Tensor:
        order = ("physio", "audio", "video")
        kv = torch.cat([tok[m] + self.mod_emb[i] for i, m in enumerate(order)], dim=1)
        pad = torch.cat([key_pad[m] for m in order], dim=1)
        # keep the CLS row valid even when every modality is dropped
        all_masked = pad.all(dim=1, keepdim=True)
        pad = pad & ~all_masked
        q = self.cls.expand(kv.size(0), -1, -1)
        h, _ = self.attn(q, self.ln(kv), self.ln(kv), key_padding_mask=pad,
                         need_weights=False)
        return self.out(h.squeeze(1))


# ======================================================== Stage 4 + 5 full model

class TemporalCrossModalNet(nn.Module):
    def __init__(self, cfg: Config, n_tasks: int, n_subjects: int, temporal: bool = True):
        super().__init__()
        self.cfg, self.temporal = cfg, temporal
        d = cfg.d_model

        self.enc = nn.ModuleDict({
            "physio": PhysioEncoder(cfg, len(cfg.physio_channels)),
            "audio": AudioEncoder(cfg),
            "video": VideoEncoder(cfg),
        })
        self.fusion = nn.ModuleList([CrossModalBlock(cfg) for _ in range(cfg.fusion_layers)])
        self.fusion_head = FusionHead(cfg)

        self.task_emb = nn.Embedding(n_tasks, d)
        if temporal:
            self.time_pos = nn.Parameter(torch.randn(1, cfg.max_windows + 1, d) * 0.02)
            self.time_cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
            layer = nn.TransformerEncoderLayer(d, cfg.n_heads, d * 2, cfg.dropout,
                                               batch_first=True, norm_first=True,
                                               activation="gelu")
            self.time_tr = nn.TransformerEncoder(layer, cfg.temporal_layers)
        self.final_norm = nn.LayerNorm(d)

        self.head_binary = nn.Linear(d, 1)
        self.head_affect3 = nn.Linear(d, 3)
        self.head_score = nn.Linear(d, 1)
        self.head_subject = nn.Linear(d, n_subjects)   # diagnostic probe only

    # ------------------------------------------------------------------ helper
    def _modality_dropout(self, mod_mask: torch.Tensor) -> torch.Tensor:
        """Stage 3 masked-token training: drop each modality for a whole
        recording with prob p, but never drop everything."""
        if not self.training or self.cfg.modality_dropout_p <= 0:
            return mod_mask
        b = mod_mask.size(0)
        keep = (torch.rand(b, 1, 3, device=mod_mask.device)
                >= self.cfg.modality_dropout_p).float()
        m = mod_mask * keep
        dead = m.sum(dim=(1, 2)) == 0                       # would lose everything
        if dead.any():
            m[dead] = mod_mask[dead]
        return m

    # ----------------------------------------------------------------- forward
    def forward(self, batch: dict, force_mask: tuple | None = None) -> dict:
        cfg = self.cfg
        b, w = batch["win_mask"].shape
        mod_mask = batch["mod_mask"]                        # [B, W, 3]
        if force_mask is not None:                          # inference-time ablation
            keep = torch.tensor(force_mask, dtype=torch.float32,
                                device=mod_mask.device).view(1, 1, 3)
            mod_mask = mod_mask * keep
        mod_mask = self._modality_dropout(mod_mask)

        flat = {
            "physio": batch["physio"].reshape(b * w, *batch["physio"].shape[2:]),
            "audio": batch["audio"].reshape(b * w, *batch["audio"].shape[2:]),
            "video": batch["video"].reshape(b * w, *batch["video"].shape[2:]),
        }
        tok, key_pad = {}, {}
        k = cfg.tokens_per_modality
        for i, m in enumerate(("physio", "audio", "video")):
            t = self.enc[m](flat[m])                        # [B*W, k, d]
            avail = mod_mask[:, :, i].reshape(b * w, 1, 1)
            tok[m] = t * avail
            key_pad[m] = (avail.reshape(b * w, 1) == 0).expand(-1, k)

        for blk in self.fusion:
            tok = blk(tok, key_pad)
        win_rep = self.fusion_head(tok, key_pad).reshape(b, w, -1)   # [B, W, d]

        win_valid = batch["win_mask"]                                # [B, W]
        win_rep = win_rep * win_valid.unsqueeze(-1)

        # ---- Stage 4
        tsk = self.task_emb(batch["task_id"]).unsqueeze(1)           # [B, 1, d]
        if self.temporal:
            seq = torch.cat([self.time_cls.expand(b, -1, -1), win_rep + tsk], dim=1)
            seq = seq + self.time_pos[:, : seq.size(1)]
            pad = torch.cat([torch.zeros(b, 1, device=seq.device), 1 - win_valid],
                            dim=1).bool()
            h = self.time_tr(seq, src_key_padding_mask=pad)[:, 0]
        else:
            denom = win_valid.sum(dim=1, keepdim=True).clamp(min=1.0)
            h = (win_rep.sum(dim=1) / denom) + tsk.squeeze(1)
        z = self.final_norm(h)

        return {
            "logit_binary": self.head_binary(z).squeeze(-1),
            "logit_affect3": self.head_affect3(z),
            "score": self.head_score(z).squeeze(-1),
            "logit_subject": self.head_subject(z),
            "embedding": z,
            "used_mask": mod_mask,
        }


def build_model(cfg: Config, n_tasks: int, n_subjects: int, temporal: bool = True):
    return TemporalCrossModalNet(cfg, n_tasks, n_subjects, temporal)
