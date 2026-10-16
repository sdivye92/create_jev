import os, sys, time, math, statistics as st
from collections import defaultdict
os.environ["HF_HUB_DISABLE_XET"] = "1"; os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
sys.path.insert(0, "/home/divye/Documents/Projects/create_jev")
os.chdir("/home/divye/Documents/Projects/create_jev/notebooks")
import torch, torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForMultimodalLM
from jev.render import Renderer
from jev.head import MaskedPointerHead
from jev.data import load_jsonl
from jev.features import build_requests, extract

processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B"); tokenizer = processor.tokenizer
model = AutoModelForMultimodalLM.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16, device_map="auto").eval()
core = model.model
for p in core.parameters(): p.requires_grad_(False)
renderer = Renderer(tokenizer); device = next(core.parameters()).device
HIDDEN = model.config.text_config.hidden_size; PAD = tokenizer.pad_token_id
train_recs = load_jsonl("../data/synthetic/train.jsonl"); val_recs = load_jsonl("../data/synthetic/val.jsonl")

def masked_ce(scores, mask, labels): return F.cross_entropy(scores.masked_fill(~mask, float("-inf")), labels)


@torch.no_grad()
def evaluate(head, fs):
    head.eval(); preds = []; loss = 0.0
    for i in range(0, len(fs), 512):
        o, d, m, l = fs.batch(torch.arange(i, min(i + 512, len(fs)), device=device))
        sc = head(o, d, m); preds.append(sc.argmax(-1)); loss += F.cross_entropy(sc, l, reduction="sum").item()
    preds = torch.cat(preds); ok = (preds == fs.labels).cpu().tolist()
    right = defaultdict(list); picked = defaultdict(set)
    for k, (q, r, opts, p, good) in enumerate(zip(fs.meta["qid"], fs.meta["rec"], fs.meta["options"], preds.tolist(), ok)):
        right[q].append(good); picked[(r, q)].add(opts[p])
    stab_keys = [k for k in picked if k[1] != "urgency"]
    return {"acc": sum(ok) / len(ok), "loss": loss / len(fs), "by_q": {k: sum(v) / len(v) for k, v in right.items()},
            "stab": sum(len(picked[k]) == 1 for k in stab_keys) / len(stab_keys)}

def train(fs, val, epochs, lr, bs=64, wd=0.01, seed=0, normalize=True, verbose=False):
    torch.manual_seed(seed); head = MaskedPointerHead(HIDDEN, normalize=normalize).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    gen = torch.Generator(device=device); gen.manual_seed(seed); hist = []
    for ep in range(epochs):
        head.train(); tot = 0.0
        for o, d, m, l in fs.epoch(bs, gen):
            opt.zero_grad(); loss = masked_ce(head(o, d), m, l); loss.backward(); opt.step(); tot += loss.item() * len(l)
        v = evaluate(head, val); hist.append((tot / len(fs), v["loss"], v["acc"]))
        if verbose: print(f"  epoch {ep:2d} train loss {tot/len(fs):.4f} | val loss {v['loss']:.4f} val acc {v['acc']:.3f}")
    return head, hist


def masked_ce(scores, mask, labels): return F.cross_entropy(scores.masked_fill(~mask, float("-inf")), labels)


@torch.no_grad()
def evaluate(head, fs):
    head.eval(); preds = []; loss = 0.0
    for i in range(0, len(fs), 512):
        o, d, m, l = fs.batch(torch.arange(i, min(i + 512, len(fs)), device=device))
        sc = head(o, d, m); preds.append(sc.argmax(-1)); loss += F.cross_entropy(sc, l, reduction="sum").item()
    preds = torch.cat(preds); ok = (preds == fs.labels).cpu().tolist()
    right = defaultdict(list); picked = defaultdict(set)
    for k, (q, r, opts, p, good) in enumerate(zip(fs.meta["qid"], fs.meta["rec"], fs.meta["options"], preds.tolist(), ok)):
        right[q].append(good); picked[(r, q)].add(opts[p])
    stab_keys = [k for k in picked if k[1] != "urgency"]
    return {"acc": sum(ok) / len(ok), "loss": loss / len(fs), "by_q": {k: sum(v) / len(v) for k, v in right.items()},
            "stab": sum(len(picked[k]) == 1 for k in stab_keys) / len(stab_keys)}

def train(fs, val, epochs, lr, bs=64, wd=0.01, seed=0, normalize=True, verbose=False, sched=None, clip=None):
    torch.manual_seed(seed); head = MaskedPointerHead(HIDDEN, normalize=normalize).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * math.ceil(len(fs) / bs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1, anneal_strategy="cos") if sched else None
    gen = torch.Generator(device=device); gen.manual_seed(seed); hist = []
    for ep in range(epochs):
        head.train(); tot = 0.0
        for o, d, m, l in fs.epoch(bs, gen):
            opt.zero_grad(); loss = masked_ce(head(o, d), m, l); loss.backward()
            if clip: torch.nn.utils.clip_grad_norm_(head.parameters(), clip)
            opt.step(); tot += loss.item() * len(l)
            if scheduler: scheduler.step()
        v = evaluate(head, val); hist.append((tot / len(fs), v["loss"], v["acc"]))
        if verbose: print(f"  epoch {ep:2d} train loss {tot/len(fs):.4f} | val loss {v['loss']:.4f} val acc {v['acc']:.3f}")
    return head, hist



cache = "/tmp/nb7_feats.pt"
if os.path.exists(cache):
    tr, va, te = torch.load(cache, weights_only=False)
else:
    test_recs = load_jsonl("../data/synthetic/test.jsonl")
    tr = extract(core, renderer, build_requests(train_recs[:800], 6, 1), PAD, HIDDEN, 64)
    va = extract(core, renderer, build_requests(val_recs[:300], 4, 2), PAD, HIDDEN, 64)
    te = extract(core, renderer, build_requests(test_recs, 4, 3), PAD, HIDDEN, 64)
    torch.save((tr, va, te), cache)
print("features ready", len(tr), len(va), len(te))

configs = [
  ("constant lr 1e-3, 12 ep",            dict(epochs=12, lr=1e-3)),
  ("cosine (one-cycle) 1e-3, 15 ep",     dict(epochs=15, lr=1e-3, sched=True)),
  ("cosine (one-cycle) 3e-3, 15 ep",     dict(epochs=15, lr=3e-3, sched=True)),
  ("cosine 1e-3 + grad clip 1.0, 15 ep", dict(epochs=15, lr=1e-3, sched=True, clip=1.0)),
  ("cosine 1e-3, 30 ep",                 dict(epochs=30, lr=1e-3, sched=True)),
]
print("final-epoch val metrics, 5 seeds")
for name, kw in configs:
    accs, losses, stabs = [], [], []
    t0 = time.time()
    for s in range(5):
        h, hist = train(tr, va, seed=s, **kw); r = evaluate(h, va)
        accs.append(r["acc"]); losses.append(r["loss"]); stabs.append(r["stab"])
    print(f"  {name:38s} acc {st.mean(accs):.3f} ± {st.pstdev(accs):.3f} (worst {min(accs):.3f}) | val loss {st.mean(losses):.3f} | stability {st.mean(stabs):.3f} | {time.time()-t0:.0f}s")
