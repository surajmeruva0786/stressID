"""Stage 5 loss:  BCE_binary + CE_3class + lambda*MSE + contrastive subject-invariance."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import Config


def subject_invariance_loss(z: torch.Tensor, y: torch.Tensor, subj: torch.Tensor,
                            temperature: float = 0.1) -> torch.Tensor:
    """Supervised contrastive loss that is explicitly subject-disentangling.

    positives : same stress label, DIFFERENT subject  (pull together)
    negatives : same subject,      DIFFERENT label    (push apart)

    Pulling same-label/different-subject pairs together while pushing
    same-subject/different-label pairs apart removes subject identity from the
    directions that encode stress, rather than removing it globally.
    """
    if z.size(0) < 3:
        return z.new_zeros(())
    zn = F.normalize(z, dim=-1)
    sim = zn @ zn.t() / temperature
    eye = torch.eye(z.size(0), dtype=torch.bool, device=z.device)

    same_y = y.view(-1, 1) == y.view(1, -1)
    same_s = subj.view(-1, 1) == subj.view(1, -1)

    pos = same_y & ~same_s & ~eye
    neg = (same_s & ~same_y) | (~same_y & ~same_s)
    valid = pos.any(dim=1) & neg.any(dim=1)
    if not valid.any():
        return z.new_zeros(())

    sim = sim.masked_fill(eye, torch.finfo(sim.dtype).min)
    exp = torch.exp(sim)
    pos_sum = (exp * pos).sum(dim=1)
    den = pos_sum + (exp * neg).sum(dim=1)
    loss = -torch.log((pos_sum + 1e-8) / (den + 1e-8))
    return loss[valid].mean()


def compute_loss(out: dict, batch: dict, cfg: Config,
                 pos_weight: torch.Tensor | None = None,
                 class_weight: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    l_bin = F.binary_cross_entropy_with_logits(
        out["logit_binary"], batch["y_binary"], pos_weight=pos_weight)
    l_aff = F.cross_entropy(out["logit_affect3"], batch["y_affect3"],
                            weight=class_weight)
    l_reg = F.mse_loss(torch.sigmoid(out["score"]), batch["y_score"])
    l_inv = subject_invariance_loss(out["embedding"], batch["y_binary"],
                                    batch["subject_id"], cfg.subj_inv_temperature)

    total = (cfg.w_binary * l_bin + cfg.w_affect3 * l_aff
             + cfg.w_regression * l_reg + cfg.w_subject_inv * l_inv)
    return total, {"loss": float(total), "bce": float(l_bin), "ce3": float(l_aff),
                   "mse": float(l_reg), "subj_inv": float(l_inv)}
