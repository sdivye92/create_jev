
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
from jev.head import MaskedPointerHead
from jev.data import load_jsonl
from jev.features import build_requests, extract

MODEL_ID = "Qwen/Qwen3.5-0.8B"
processor = AutoProcessor.from_pretrained(MODEL_ID)
tokenizer = processor.tokenizer
model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto").eval()

core = model.model
for p in core.parameters():
    p.requires_grad_(False)

renderer = Renderer(tokenizer)
device = next(core.parameters()).device
HIDDEN = model.config.text_config.hidden_size
PAD = tokenizer.pad_token_id
print("loaded | GPU memory allocated: %.2f GB" % (torch.cuda.memory_allocated() / 1e9))

train_records = load_jsonl("../data/synthetic/train.jsonl")
val_records = load_jsonl("../data/synthetic/val.jsonl")
print(len(train_records), "train records,", len(val_records), "val records (the 300 test records stay untouched until the end)\n")

rec = train_records[3]
print("STATE:\n" + rec["state"] + "\n")
for q in rec["questions"]:
    print(f"[{q['type']}] {q['id']}: {q['prompt']}")
    for i, o in enumerate(q["options"]):
        print(f"   {'*' if i == q['label'] else ' '} {i}: {o}")

probe = build_requests(train_records[:10], n_orders=2, seed=0)[:24]
batched = extract(core, renderer, probe, PAD, HIDDEN, batch_size=8)
single = extract(core, renderer, probe, PAD, HIDDEN, batch_size=1)

cos_dec = F.cosine_similarity(batched.dec.float(), single.dec.float(), dim=-1)
cos_opt = F.cosine_similarity(batched.opt.float(), single.opt.float(), dim=-1)[batched.mask]
print("decide vectors: min cosine %.5f | option vectors: min cosine %.5f" % (cos_dec.min().item(), cos_opt.min().item()))
print("bit-identical rows: %d of %d" % ((batched.dec == single.dec).all(-1).sum().item(), len(batched)))
print("padding slots are exactly zero:", bool((batched.opt[~batched.mask] == 0).all()))

N_TRAIN, N_VAL = 800, 300

torch.cuda.synchronize(); t0 = time.time()
train_fs = extract(core, renderer, build_requests(train_records[:N_TRAIN], n_orders=6, seed=1), PAD, HIDDEN)
torch.cuda.synchronize(); t_train = time.time() - t0

t0 = time.time()
val_fs = extract(core, renderer, build_requests(val_records[:N_VAL], n_orders=4, seed=2), PAD, HIDDEN)
torch.cuda.synchronize(); t_val = time.time() - t0

print(f"train: {len(train_fs)} examples in {t_train:.0f}s ({1000 * t_train / len(train_fs):.1f} ms each)")
print(f"val:   {len(val_fs)} examples in {t_val:.0f}s")
print("max options per question:", train_fs.opt.shape[1])
print("cached features: %.0f MB" % ((train_fs.opt.numel() + train_fs.dec.numel() + val_fs.opt.numel() + val_fs.dec.numel()) * 2 / 1e6))

def masked_cross_entropy(scores, mask, labels):
    """scores [B, N] (junk in padded slots), mask [B, N] (True = real option), labels [B]  ->  scalar mean loss"""
    return F.cross_entropy(scores.masked_fill(~mask, float("-inf")), labels)

head_check = MaskedPointerHead(HIDDEN).to(device)
o, d, m, l = train_fs.batch(torch.arange(32, device=device))
scores = head_check(o, d)     # no mask passed, so padded slots hold ordinary-looking numbers

fast = masked_cross_entropy(scores, m, l)
slow = torch.stack([F.cross_entropy(scores[i, m[i]].unsqueeze(0), l[i:i + 1]) for i in range(len(l))]).mean()
forgot = F.cross_entropy(scores, l)

print("masked batch loss     : %.6f" % fast.item())
print("one example at a time : %.6f" % slow.item())
print("forgot the mask       : %.6f   <- silently wrong" % forgot.item())
assert torch.allclose(fast, slow, atol=1e-5), "masked loss does not match the one-at-a-time loss"

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

def chance_accuracy(fs):
    return sum(1 / len(o) for o in fs.meta["options"]) / len(fs)

print("chance accuracy on val: %.3f\n" % chance_accuracy(val_fs))

t0 = time.time()
head, history = train(train_fs, val_fs, epochs=30, lr=1e-3, log_every=3)
print("\ntraining took %.1fs" % (time.time() - t0))

r = evaluate(head, val_fs)
print(f"\nval: accuracy {r['acc']:.3f} | loss {r['loss']:.3f} | order stability {r['stability']:.3f}")
print("by question:", {k: round(v, 3) for k, v in r["by_question"].items()})

configs = [
    ("constant lr 1e-3, 12 epochs",           dict(epochs=12, schedule=False)),
    ("one-cycle 1e-3, 30 epochs",             dict(epochs=30, schedule=True)),
    ("one-cycle 1e-3, 30 epochs, no norm",    dict(epochs=30, schedule=True, normalize=False)),
]
for name, kw in configs:
    accs, losses = [], []
    for seed in range(5):
        h, _ = train(train_fs, val_fs, seed=seed, lr=1e-3, **kw)
        r = evaluate(h, val_fs)
        accs.append(r["acc"]); losses.append(r["loss"])
    print(f"{name:38s} val acc {statistics.mean(accs):.3f} +- {statistics.pstdev(accs):.3f} (worst {min(accs):.3f}) | val loss {statistics.mean(losses):.3f}")

print("records  examples   val acc            order stability")
for k in [25, 50, 100, 200, 400, 800]:
    sub = train_fs.where_rec_below(k)
    accs, stabs = [], []
    for seed in range(3):
        h, _ = train(sub, val_fs, epochs=15, seed=seed)
        r = evaluate(h, val_fs)
        accs.append(r["acc"]); stabs.append(r["stability"])
    print(f"{k:6d}  {len(sub):8d}   {statistics.mean(accs):.3f} +- {statistics.pstdev(accs):.3f}      {statistics.mean(stabs):.3f}")

train_fs_1order = extract(core, renderer, build_requests(train_records[:N_TRAIN], n_orders=1, seed=1), PAD, HIDDEN)

for name, fs, epochs in [("1 order (original only)", train_fs_1order, 90), ("6 orders (shuffled)", train_fs, 15)]:
    accs, stabs = [], []
    for seed in range(3):
        h, _ = train(fs, val_fs, epochs=epochs, seed=seed)
        r = evaluate(h, val_fs)
        accs.append(r["acc"]); stabs.append(r["stability"])
    print(f"{name:26s} val acc {statistics.mean(accs):.3f} | order stability {statistics.mean(stabs):.3f}")

o, d, m, l = val_fs.batch(torch.arange(len(val_fs), device=device))
with torch.no_grad():
    pred = head(o, d, m).argmax(-1).tolist()

total, wrong = defaultdict(int), defaultdict(lambda: defaultdict(int))
for i, p in enumerate(pred):
    if val_fs.meta["qid"][i] != "department":
        continue
    opts = val_fs.meta["options"][i]
    true, got = opts[val_fs.labels[i].item()].split(": ")[0], opts[p].split(": ")[0]
    total[true] += 1
    if true != got:
        wrong[true][got] += 1

print(f"{'true answer':13s} {'n':>4s}  {'error rate':>10s}   most common mistakes")
for true in sorted(total, key=lambda t: -sum(wrong[t].values()) / total[t]):
    n_wrong = sum(wrong[true].values())
    top = sorted(wrong[true].items(), key=lambda kv: -kv[1])[:3]
    print(f"{true:13s} {total[true]:4d}  {n_wrong / total[true]:10.1%}   {top}")

by_ticket = defaultdict(lambda: [0, 0])      # record index -> [wrong orders, total orders]
for i, p in enumerate(pred):
    if val_fs.meta["qid"][i] == "department":
        k = val_fs.meta["rec"][i]
        by_ticket[k][1] += 1
        by_ticket[k][0] += p != val_fs.labels[i].item()

bad = [(k, w, t) for k, (w, t) in by_ticket.items() if w > 0]
print(f"{len(bad)} of {len(by_ticket)} tickets have at least one wrong option order; {sum(w == t for _, w, t in bad)} are wrong in every order\n")
for k, w, t in sorted(bad, key=lambda x: -x[1])[:10]:
    q = next(q for q in val_records[k]["questions"] if q["id"] == "department")
    true = q["options"][q["label"]].split(": ")[0]
    options = [o.split(": ")[0] for o in q["options"]]
    print(f"wrong in {w}/{t} orders | true={true:9s} | options={options}\n    {val_records[k]['state'][:110]!r}")

test_records = load_jsonl("../data/synthetic/test.jsonl")
test_fs = extract(core, renderer, build_requests(test_records, n_orders=4, seed=3), PAD, HIDDEN)

final_head, _ = train(train_fs, val_fs, epochs=30, lr=1e-3, seed=0)
for name, fs in [("val", val_fs), ("test", test_fs)]:
    r = evaluate(final_head, fs)
    print(f"{name:5s} accuracy {r['acc']:.3f} | loss {r['loss']:.3f} | order stability {r['stability']:.3f} | chance {chance_accuracy(fs):.3f}")
    print("      by question:", {k: round(v, 3) for k, v in r["by_question"].items()})