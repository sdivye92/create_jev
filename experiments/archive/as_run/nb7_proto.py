import os, sys, time, math
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

reqs = build_requests(train_recs[:10], 2, 0)[:24]
b = extract(core, renderer, reqs, PAD, HIDDEN, batch_size=8)
s = extract(core, renderer, reqs, PAD, HIDDEN, batch_size=1)
cos = F.cosine_similarity(b.dec.float(), s.dec.float(), dim=-1)
cos_o = F.cosine_similarity(b.opt.float(), s.opt.float(), dim=-1)[b.mask]
print("batched vs single | decide cosine min %.5f | option cosine min %.5f | exact-equal rows %d/%d | masks equal %s" % (
    cos.min().item(), cos_o.min().item(), (b.dec == s.dec).all(-1).sum().item(), len(b), torch.equal(b.mask, s.mask)))
print("padding slots are zero:", bool((b.opt[~b.mask] == 0).all()))

N_TRAIN, N_VAL = 800, 300
torch.cuda.synchronize(); t0 = time.time()
tr = extract(core, renderer, build_requests(train_recs[:N_TRAIN], 6, 1), PAD, HIDDEN, 64); torch.cuda.synchronize(); t_tr = time.time() - t0
t0 = time.time(); va = extract(core, renderer, build_requests(val_recs[:N_VAL], 4, 2), PAD, HIDDEN, 64); torch.cuda.synchronize(); t_va = time.time() - t0
print(f"train: {len(tr)} examples in {t_tr:.1f}s ({1000*t_tr/len(tr):.1f} ms each) | val: {len(va)} in {t_va:.1f}s | n_max options {tr.opt.shape[1]}")
print("feature memory: %.0f MB" % ((tr.opt.numel() + tr.dec.numel() + va.opt.numel() + va.dec.numel()) * 2 / 1e6))

def masked_ce(scores, mask, labels): return F.cross_entropy(scores.masked_fill(~mask, float("-inf")), labels)

h = MaskedPointerHead(HIDDEN).to(device)
o, d, m, l = tr.batch(torch.arange(20, device=device))
single = torch.stack([F.cross_entropy(h(o[i:i+1][:, m[i]], d[i:i+1]), l[i:i+1]) for i in range(20)]).mean()
print("masked batch loss %.6f vs mean of single losses %.6f" % (masked_ce(h(o, d), m, l).item(), single.item()))

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

chance = sum(1 / len(o) for o in va.meta["options"]) / len(va)
print("\nval chance accuracy %.3f\n" % chance)
for norm in [True, False]:
    for lr in [1e-4, 3e-4, 1e-3]:
        t0 = time.time(); head, hist = train(tr, va, 8, lr, normalize=norm)
        print(f"normalize={norm!s:5} lr={lr:<6g} epoch0 train loss {hist[0][0]:7.3f} | final train loss {hist[-1][0]:.4f} val loss {hist[-1][1]:.4f} val acc {hist[-1][2]:.3f} | best val acc {max(h[2] for h in hist):.3f} | {time.time()-t0:.1f}s")
