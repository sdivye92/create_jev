"""Losses, head training, evaluation, and the LoRA helpers (peft) with end-to-end adapter training."""
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .calibration import brier_score, probabilities
from .features import build_requests, encode, render_requests
from .head import MaskedPointerHead

LORA_TARGET_REGEX = r".*language_model\.layers\.\d+\..*\.(in_proj_qkv|in_proj_z|out_proj|q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def masked_cross_entropy(scores, mask, labels):
    """Cross-entropy over the real options only: padded slots are set to -inf so they get zero probability."""
    return F.cross_entropy(scores.masked_fill(~mask, float("-inf")), labels)


def combined_loss(scores, mask, labels, brier_weight=0.0):
    loss = masked_cross_entropy(scores, mask, labels)
    if brier_weight:
        loss = loss + brier_weight * brier_score(probabilities(scores, mask), labels, mask)
    return loss


def chance_accuracy(fs):
    return sum(1 / len(o) for o in fs.meta["options"]) / len(fs)


@torch.no_grad()
def predict_scores(head, fs, batch_size=512):
    """Raw scores [E, N] for a cached FeatureSet, on the CPU in fp32."""
    head.eval()
    out = []
    for i in range(0, len(fs), batch_size):
        o, d, m, _ = fs.batch(torch.arange(i, min(i + batch_size, len(fs)), device=fs.labels.device))
        out.append(head(o, d, m).float().cpu())
    return torch.cat(out)


def evaluate(head, fs):
    scores = predict_scores(head, fs)
    mask, labels = fs.mask.cpu(), fs.labels.cpu()
    probs = probabilities(scores, mask)
    preds = probs.argmax(-1)
    correct = (preds == labels).tolist()
    loss = -torch.log(probs.gather(1, labels[:, None]).squeeze(1).clamp_min(1e-12)).mean().item()

    by_q, picked = defaultdict(list), defaultdict(set)
    for ok, p, qid, rec, typ, opts in zip(correct, preds.tolist(), fs.meta["qid"], fs.meta["rec"], fs.meta["type"], fs.meta["options"]):
        by_q[qid].append(ok)
        if typ != "score":
            picked[(rec, qid)].add(opts[p])
    return {
        "acc": sum(correct) / len(correct),
        "loss": loss,
        "by_question": {k: sum(v) / len(v) for k, v in by_q.items()},
        "stability": sum(len(s) == 1 for s in picked.values()) / max(len(picked), 1),
    }


def train_head(fs, val_fs, hidden, epochs=30, lr=1e-3, batch_size=64, weight_decay=0.01, seed=0,
               normalize=True, schedule=True, brier_weight=0.0, log_every=0):
    """Train a pointer head on cached backbone features (the backbone is frozen, so its output never changes)."""
    device = fs.labels.device
    torch.manual_seed(seed)
    head = MaskedPointerHead(hidden, normalize=normalize).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    steps = epochs * math.ceil(len(fs) / batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=steps, pct_start=0.1, anneal_strategy="cos") if schedule else None
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    history = []
    for epoch in range(epochs):
        head.train()
        total = 0.0
        for o, d, m, l in fs.epoch(batch_size, generator):
            optimizer.zero_grad()
            loss = combined_loss(head(o, d), m, l, brier_weight)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            total += loss.item() * len(l)
        v = evaluate(head, val_fs) if val_fs is not None else {"loss": float("nan"), "acc": float("nan")}
        history.append({"train_loss": total / len(fs), "val_loss": v["loss"], "val_acc": v["acc"]})
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"epoch {epoch:2d}  train loss {history[-1]['train_loss']:.4f} | val loss {v['loss']:.4f} | val acc {v['acc']:.3f}")
    return head, history


def add_lora(core, rank=8, alpha=16, dropout=0.0, target_regex=LORA_TARGET_REGEX):
    """Inject LoRA adapters into the language model's linear layers (in place) using peft. Returns the PeftModel."""
    from peft import LoraConfig, get_peft_model
    config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=target_regex, bias="none")
    return get_peft_model(core, config)


def enable_checkpointing(core):
    """Recompute each decoder layer's activations in the backward pass instead of storing them."""
    for layer in core.language_model.layers:
        if not hasattr(layer, "_plain_forward"):
            layer._plain_forward = layer.forward
            layer.forward = lambda *a, _l=layer, **k: checkpoint(_l._plain_forward, *a, use_reentrant=False, **k)


def disable_checkpointing(core):
    for layer in core.language_model.layers:
        if hasattr(layer, "_plain_forward"):
            layer.forward = layer._plain_forward
            del layer._plain_forward


def train_lora(core, head, renderer, records, pad_id, epochs=2, batch_size=8, lora_lr=2e-4, head_lr=5e-4, brier_weight=0.0,
               weight_decay=0.01, seed=0, checkpointing=True, max_steps=None, log_every=25, on_epoch_end=None):
    """Train the LoRA adapters and the head together, end to end through the backbone.

    Each epoch shows every question once, in a fresh random option order (the label moves with its option),
    so there is no feature cache: the backbone's output changes at every step.
    """
    device = next(core.parameters()).device
    lora_params = [p for p in core.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": lora_params, "lr": lora_lr}, {"params": head.parameters(), "lr": head_lr}], weight_decay=weight_decay)
    n_questions = sum(len(r["questions"]) for r in records)
    steps_per_epoch = math.ceil(n_questions / batch_size)
    total_steps = epochs * steps_per_epoch if max_steps is None else min(max_steps, epochs * steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[lora_lr, head_lr], total_steps=total_steps, pct_start=0.1, anneal_strategy="cos")
    if checkpointing:
        enable_checkpointing(core)

    history, step, t0 = [], 0, time.time()
    for epoch in range(epochs):
        requests = build_requests(records, n_orders=1, seed=seed * 1000 + epoch, keep_original_first=False)
        rendered = render_requests(renderer, requests)
        labels = torch.tensor([r["question"]["label"] for r in requests], device=device)
        order = torch.randperm(len(requests), generator=torch.Generator().manual_seed(seed * 1000 + epoch)).tolist()
        head.train()
        running = []
        for i in range(0, len(order), batch_size):
            if step >= total_steps:
                break
            idx = order[i:i + batch_size]
            n_max = max(len(rendered[j][1]) for j in idx)
            opt, dec, mask = encode(core, rendered, idx, pad_id, n_max)
            loss = combined_loss(head(opt.float(), dec.float(), mask), mask, labels[idx], brier_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            running.append(loss.item())
            if log_every and step % log_every == 0:
                print(f"epoch {epoch} step {step:4d}/{total_steps}  loss {sum(running[-log_every:]) / len(running[-log_every:]):.4f}  ({time.time() - t0:.0f}s elapsed, peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB)")
        history.append({"epoch": epoch, "train_loss": sum(running) / max(len(running), 1), "steps": step})
        if on_epoch_end:
            history[-1].update(on_epoch_end(epoch))
        if step >= total_steps:
            break
    if checkpointing:
        disable_checkpointing(core)
    return history


def save_run(path, peft_model, head):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(path)
    torch.save(head.state_dict(), path / "head.pt")
