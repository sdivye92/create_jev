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


test_recs = load_jsonl("../data/synthetic/test.jsonl")
N_TRAIN, N_VAL = 800, 300
t0 = time.time()
tr = extract(core, renderer, build_requests(train_recs[:N_TRAIN], 6, 1), PAD, HIDDEN, 64)
va = extract(core, renderer, build_requests(val_recs[:N_VAL], 4, 2), PAD, HIDDEN, 64)
te = extract(core, renderer, build_requests(test_recs, 4, 3), PAD, HIDDEN, 64)
print("extracted in %.0fs" % (time.time() - t0))

chance = lambda fs: sum(1 / len(o) for o in fs.meta["options"]) / len(fs)
print("chance: val %.3f test %.3f" % (chance(va), chance(te)))

print("\nA) epoch curve (normalize=True, lr=1e-3, 3 seeds averaged for val acc; seed 0 shown for loss)")
runs = [train(tr, va, 30, 1e-3, seed=s)[1] for s in range(3)]
for ep in [0, 1, 2, 3, 5, 8, 12, 16, 20, 25, 29]:
    print(f"  epoch {ep:2d}: train loss {runs[0][ep][0]:.4f} | val loss {runs[0][ep][1]:.4f} | val acc {st.mean(r[ep][2] for r in runs):.3f}")
best_ep = max(range(30), key=lambda e: st.mean(r[e][2] for r in runs)); print("  best epoch by mean val acc:", best_ep, "->", round(st.mean(r[best_ep][2] for r in runs), 3))
best_loss_ep = min(range(30), key=lambda e: st.mean(r[e][1] for r in runs)); print("  best epoch by mean val loss:", best_loss_ep, "->", round(st.mean(r[best_loss_ep][1] for r in runs), 4))

print("\nB) learning curve (normalize=True, lr=1e-3, 12 epochs, 3 seeds)")
for k in [25, 50, 100, 200, 400, 800]:
    sub = tr.where_rec_below(k); accs = []; stabs = []
    for s in range(3):
        h, _ = train(sub, va, 12, 1e-3, seed=s); r = evaluate(h, va); accs.append(r["acc"]); stabs.append(r["stab"])
    print(f"  {k:4d} records ({len(sub):5d} examples): val acc {st.mean(accs):.3f} ± {st.pstdev(accs):.3f} | order stability {st.mean(stabs):.3f}")

print("\nC) final model (12 epochs, seed 0): val and per-question; test once")
head, _ = train(tr, va, 12, 1e-3, seed=0)
for name, fs in [("val", va), ("test", te)]:
    r = evaluate(head, fs)
    print(f"  {name}: acc {r['acc']:.3f} | loss {r['loss']:.3f} | stability {r['stab']:.3f} | by question:", {k: round(v, 3) for k, v in r["by_q"].items()})

print("\nD) shuffling ablation: train on original order only (n_orders=1) vs 6 orders; same number of steps")
tr1 = extract(core, renderer, build_requests(train_recs[:N_TRAIN], 1, 1), PAD, HIDDEN, 64)
for name, fs, ep in [("1 order ", tr1, 72), ("6 orders", tr, 12)]:
    accs, stabs = [], []
    for s in range(3):
        h, _ = train(fs, va, ep, 1e-3, seed=s); r = evaluate(h, va); accs.append(r["acc"]); stabs.append(r["stab"])
    print(f"  {name}: val acc {st.mean(accs):.3f} | order stability {st.mean(stabs):.3f}")

print("\nE) errors on val (final model): department errors by true option text and what was predicted")
head.eval(); errs = defaultdict(int); tot = defaultdict(int)
o, d, m, l = va.batch(torch.arange(len(va), device=device))
with torch.no_grad(): pr = head(o, d, m).argmax(-1).tolist()
for i, p in enumerate(pr):
    if va.meta["qid"][i] != "department": continue
    true = va.meta["options"][i][va.labels[i].item()].split(": ")[0]; pred = va.meta["options"][i][p].split(": ")[0]
    tot[true] += 1; errs[(true, pred)] += (p != va.labels[i].item())
for t in sorted(tot): print(f"  true={t:11s} n={tot[t]:4d} errors:", {k[1]: v for k, v in errs.items() if k[0] == t and v})
