"""The batched pointer head used by the training pipeline."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedPointerHead(nn.Module):
    """Scores each option's readout vector against the decision vector, for a padded batch of questions."""

    def __init__(self, hidden_size, d_head=256, normalize=True):
        super().__init__()
        self.opt_proj = nn.Linear(hidden_size, d_head, bias=False)
        self.dec_proj = nn.Linear(hidden_size, d_head, bias=False)
        self.scale = d_head ** -0.5
        self.normalize = normalize

    def forward(self, opt, dec, mask=None):
        """opt [B, N, D], dec [B, D], mask [B, N] (True = real option)  ->  scores [B, N]; padded slots are -inf."""
        if self.normalize:
            opt = F.rms_norm(opt, opt.shape[-1:])
            dec = F.rms_norm(dec, dec.shape[-1:])
        scores = torch.bmm(self.opt_proj(opt), self.dec_proj(dec).unsqueeze(-1)).squeeze(-1) * self.scale
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        return scores
