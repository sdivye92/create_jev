
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import sys
sys.path.insert(0, "..")

import json, math, time, statistics
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForMultimodalLM

from jev.render import Renderer
from jev.features import FeatureSet
from jev.head import MaskedPointerHead
from jev.data import load_jsonl
from jev.features import build_requests, extract

MODEL_ID = "Qwen/Qwen3.5-0.8B"
processor = AutoProcessor.from_pretrained(MODEL_ID)
tokenizer = processor.tokenizer
def masked_cross_entropy(scores, mask, labels):
    return F.cross_entropy(scores.masked_fill(~mask, float("-inf")), labels)
@torch.no_grad()
def evaluate(head, fs):
    head.eval()
    preds, loss_sum = [], 0.0
    for i in range(0, len(fs), 512):
        o, d, m, l = fs.batch(torch.arange(i, min(i + 512, len(fs)), device=device))
        sc = head(o, d, m)
        preds.append(sc.argmax(-1))
        loss_sum += F.cross_entropy(sc, l, reduction="sum").item()
    preds = torch.cat(preds)
    correct = (preds == fs.labels).cpu().tolist()

    by_q, picked = defaultdict(list), defaultdict(set)
    for ok, p, qid, rec, typ, opts in zip(correct, preds.tolist(), fs.meta["qid"], fs.meta["rec"], fs.meta["type"], fs.meta["options"]):
        by_q[qid].append(ok)
        if typ != "score":
            picked[(rec, qid)].add(opts[p])
    return {
        "acc": sum(correct) / len(correct),
        "loss": loss_sum / len(fs),
        "by_question": {k: sum(v) / len(v) for k, v in by_q.items()},
        "stability": sum(len(s) == 1 for s in picked.values()) / len(picked),
    }


def train(fs, val_fs, epochs=30, lr=1e-3, batch_size=64, weight_decay=0.01, seed=0, normalize=True, schedule=True, log_every=0):
    torch.manual_seed(seed)
    head = MaskedPointerHead(HIDDEN, normalize=normalize).to(device)
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
            loss = masked_cross_entropy(head(o, d), m, l)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            total += loss.item() * len(l)
        v = evaluate(head, val_fs)
        history.append({"train_loss": total / len(fs), "val_loss": v["loss"], "val_acc": v["acc"]})
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"epoch {epoch:2d}  train loss {history[-1]['train_loss']:.4f} | val loss {v['loss']:.4f} | val acc {v['acc']:.3f}")
    return head, history


HIDDEN = 1024
device = torch.device("cuda")
train_fs, val_fs, test_fs = torch.load("/tmp/nb7_feats.pt", weights_only=False)
val_records = load_jsonl("../data/synthetic/val.jsonl")
head, _ = train(train_fs, val_fs, epochs=30, lr=1e-3, seed=0)
print("val acc", round(evaluate(head, val_fs)["acc"], 3))
o, d, m, l = val_fs.batch(torch.arange(len(val_fs), device=device))
with torch.no_grad(): pred = head(o, d, m).argmax(-1).tolist()
shown = 0
for i, p in enumerate(pred):
    if val_fs.meta["qid"][i] != "department" or p == val_fs.labels[i].item(): continue
    opts = val_fs.meta["options"][i]; true = opts[val_fs.labels[i].item()].split(": ")[0]; got = opts[p].split(": ")[0]
    if true in ("returns", "other") and shown < 12:
        state = val_records[val_fs.meta["rec"][i]]["state"].replace("\n", " | ")
        print(f"true={true:8s} predicted={got:10s} options={[o_.split(': ')[0] for o_ in opts]}\n    {state[:150]}")
        shown += 1
