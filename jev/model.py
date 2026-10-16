"""The single-question pointer head and wrapper from the step-by-step tutorials. The pipeline uses head.py instead."""
import math

import torch
import torch.nn as nn


class PointerHead(nn.Module):
    def __init__(self, hidden_size, d_head=256):
        super().__init__()
        self.opt_proj = nn.Linear(hidden_size, d_head, bias=False)
        self.dec_proj = nn.Linear(hidden_size, d_head, bias=False)
        self.d_head = d_head

    def forward(self, opt_vecs, dec_vec):
        """opt_vecs: [num_options, hidden_size], dec_vec: [hidden_size]  ->  scores: [num_options]"""
        opt_projs = self.opt_proj(opt_vecs)
        dec_proj = self.dec_proj(dec_vec)
        scores = torch.einsum("nd,d->n", opt_projs, dec_proj)
        return scores / math.sqrt(self.d_head)


class JevModel(nn.Module):
    def __init__(self, core, head):
        super().__init__()
        self.core = core
        self.head = head

    def forward(self, input_ids, opt_positions, decide_position):
        h = self.core(input_ids=input_ids).last_hidden_state[0].float()
        return self.head(h[opt_positions], h[decide_position])
