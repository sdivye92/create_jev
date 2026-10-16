"""Run the backbone over rendered questions and cache the readout vectors (encode, extract, FeatureSet)."""
import random

import torch
from torch.nn.utils.rnn import pad_sequence

from .data import random_order, reorder_question


def build_requests(records, n_orders, seed, keep_original_first=True):
    """One request per (question, option order). With keep_original_first, order 0 is the original."""
    rng = random.Random(seed)
    requests = []
    for r_idx, rec in enumerate(records):
        for q in rec["questions"]:
            for k in range(n_orders):
                q_k = q if (k == 0 and keep_original_first) else reorder_question(q, random_order(len(q["options"]), rng))
                requests.append({"state": rec["state"], "question": q_k, "type": q["type"], "qid": q["id"], "rec": r_idx})
    return requests


class FeatureSet:
    """Readout vectors for many questions, padded into single tensors that live on the GPU in bf16."""

    def __init__(self, opt, dec, mask, labels, meta):
        self.opt, self.dec, self.mask, self.labels, self.meta = opt, dec, mask, labels, meta

    def __len__(self):
        return len(self.labels)

    def take(self, idx):
        idx_list = idx.tolist()
        meta = {k: [v[i] for i in idx_list] for k, v in self.meta.items()}
        return FeatureSet(self.opt[idx], self.dec[idx], self.mask[idx], self.labels[idx], meta)

    def where_rec_below(self, k):
        keep = torch.tensor([r < k for r in self.meta["rec"]], device=self.labels.device)
        return self.take(torch.nonzero(keep).squeeze(-1))

    def batch(self, idx):
        return self.opt[idx].float(), self.dec[idx].float(), self.mask[idx], self.labels[idx]

    def epoch(self, batch_size, generator):
        perm = torch.randperm(len(self), generator=generator, device=self.labels.device)
        for i in range(0, len(self), batch_size):
            yield self.batch(perm[i:i + batch_size])


def render_requests(renderer, requests):
    return [renderer.render_question({"state": r["state"], "questions": [r["question"]]}, 0) for r in requests]


def encode(core, rendered, idx, pad_id, n_max):
    """Run the backbone on rendered[i] for i in idx and gather the readout vectors. Differentiable.

    rendered[i] is (ids, option_end_positions, decide_position) and, for questions that contain images, a fourth
    element {"types", "pixels", "grids"} from MMRenderer. Returns opt [B, n_max, D], dec [B, D], mask [B, n_max].
    Right padding is exact here: the backbone is causal, so real tokens never see the padding after them.
    """
    device = next(core.parameters()).device
    seqs = [torch.tensor(rendered[i][0]) for i in idx]
    ids = pad_sequence(seqs, batch_first=True, padding_value=pad_id)
    lengths = torch.tensor([len(s) for s in seqs])
    attn = (torch.arange(ids.shape[1])[None, :] < lengths[:, None]).long()

    extras = {}
    with_images = [(row, rendered[i][3]) for row, i in enumerate(idx) if len(rendered[i]) > 3 and rendered[i][3]["pixels"]]
    if with_images:
        types = torch.zeros_like(ids)
        pixels, grids = [], []
        for row, e in with_images:
            types[row, :len(e["types"])] = torch.tensor(e["types"])
            pixels += e["pixels"]
            grids += e["grids"]
        extras = {"mm_token_type_ids": types.to(device), "pixel_values": torch.cat(pixels).to(device), "image_grid_thw": torch.cat(grids).to(device)}

    h = core(input_ids=ids.to(device), attention_mask=attn.to(device), use_cache=False, **extras).last_hidden_state

    ends = torch.zeros(len(idx), n_max, dtype=torch.long)
    ends_ok = torch.zeros(len(idx), n_max, dtype=torch.bool)
    for row, i in enumerate(idx):
        e = rendered[i][1]
        ends[row, :len(e)] = torch.tensor(e)
        ends_ok[row, :len(e)] = True
    dpos = torch.tensor([rendered[i][2] for i in idx])

    rows = torch.arange(len(idx), device=device)
    ends, ends_ok, dpos = ends.to(device), ends_ok.to(device), dpos.to(device)
    opt = h[rows[:, None], ends] * ends_ok[..., None]
    dec = h[rows, dpos]
    return opt, dec, ends_ok


@torch.no_grad()
def extract(core, renderer, requests, pad_id, hidden_size, batch_size=64):
    """Run the frozen backbone over all requests in length-sorted padded batches and cache the readout vectors."""
    device = next(core.parameters()).device
    rendered = render_requests(renderer, requests)
    n_max = max(len(r[1]) for r in rendered)
    total = len(requests)
    opt = torch.zeros(total, n_max, hidden_size, dtype=torch.bfloat16, device=device)
    dec = torch.zeros(total, hidden_size, dtype=torch.bfloat16, device=device)
    mask = torch.zeros(total, n_max, dtype=torch.bool, device=device)

    order = sorted(range(total), key=lambda i: len(rendered[i][0]))
    for start in range(0, total, batch_size):
        idx = order[start:start + batch_size]
        o, d, m = encode(core, rendered, idx, pad_id, n_max)
        idx_t = torch.tensor(idx, device=device)
        opt[idx_t], dec[idx_t], mask[idx_t] = o, d, m

    labels = torch.tensor([r["question"]["label"] for r in requests], device=device)
    meta = {k: [r[k] if k != "options" else r["question"]["options"] for r in requests] for k in ("type", "qid", "rec", "options")}
    return FeatureSet(opt, dec, mask, labels, meta)
