
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import sys
sys.path.insert(0, "..")

import json, math, random
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForMultimodalLM

from jev.render import Renderer
from jev.model import PointerHead

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
print("loaded | GPU memory allocated: %.2f GB" % (torch.cuda.memory_allocated() / 1e9))

smoke = [json.loads(line) for line in open("../data/smoke_12.jsonl")]
train_records, heldout_records = smoke[:8], smoke[8:]
print(len(train_records), "train records,", len(heldout_records), "held-out records")

for name, recs in [("train", train_records), ("held-out", heldout_records)]:
    print(f"\n{name}: label counts per question")
    for qi, qid in enumerate(["department", "escalate", "frustration"]):
        opts = recs[0]["questions"][qi]["options"]
        counts = [sum(r["questions"][qi]["label"] == k for r in recs) for k in range(len(opts))]
        print(f"  {qid:12s}", dict(zip(opts, counts)))

def shuffle_question(q, rng):
    """Return a copy of q with its options shuffled and the label moved to follow its option."""
    if q["type"] == "score":
        return dict(q)
    perm = list(range(len(q["options"])))
    rng.shuffle(perm)
    return {**q, "options": [q["options"][i] for i in perm], "label": perm.index(q["label"])}

rng = random.Random(0)
moved = 0
for rec in smoke:
    for q in rec["questions"]:
        before = dict(q)
        for _ in range(50):
            s = shuffle_question(q, rng)
            assert sorted(s["options"]) == sorted(q["options"]), "options were lost or changed"
            assert s["options"][s["label"]] == q["options"][q["label"]], "label no longer points at the right option"
            if q["type"] == "score":
                assert s["options"] == q["options"] and s["label"] == q["label"], "score questions must stay in order"
            moved += s["options"] != q["options"]
        assert q == before, "the input dict was modified"
print("all checks passed;", moved, "of", 12 * 3 * 50, "shuffled copies had a different order from the original")

def build_examples(records, n_shuffles, seed):
    rng = random.Random(seed)
    examples = []
    for r_idx, rec in enumerate(records):
        for q in rec["questions"]:
            for s in range(n_shuffles):
                q_s = q if s == 0 else shuffle_question(q, rng)
                ids, ends, dpos = renderer.render_question({"state": rec["state"], "questions": [q_s]}, 0)
                with torch.no_grad():
                    h = core(input_ids=torch.tensor([ids], device=device)).last_hidden_state[0].float()
                examples.append({
                    "opt": h[ends], "dec": h[dpos], "label": q_s["label"],
                    "type": q["type"], "qid": q["id"], "rec": r_idx, "options": q_s["options"],
                })
    return examples

import time
t0 = time.time()
heldout_examples = build_examples(heldout_records, n_shuffles=6, seed=123)
print(len(heldout_examples), "held-out examples cached in %.1fs" % (time.time() - t0))

def train_head(examples, epochs=40, lr=1e-4, seed=0):
    torch.manual_seed(seed)
    head = PointerHead(HIDDEN).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    rng = random.Random(seed)
    for epoch in range(epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        total = 0.0
        for i in order:
            ex = examples[i]
            optimizer.zero_grad()
            scores = head(ex["opt"], ex["dec"])
            loss = F.cross_entropy(scores.unsqueeze(0), torch.tensor([ex["label"]], device=device))
            loss.backward()
            optimizer.step()
            total += loss.item()
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:3d}  mean loss {total / len(examples):.4f}")
    return head

def evaluate(head, examples):
    head.eval()
    right = defaultdict(list)
    picked = defaultdict(set)
    with torch.no_grad():
        for ex in examples:
            pred = head(ex["opt"], ex["dec"]).argmax().item()
            right[ex["type"]].append(pred == ex["label"])
            picked[(ex["rec"], ex["qid"])].add(ex["options"][pred])
    allr = [v for vs in right.values() for v in vs]
    return {
        "acc": sum(allr) / len(allr),
        "acc_by_type": {t: sum(v) / len(v) for t, v in right.items()},
        "stability": sum(len(s) == 1 for s in picked.values()) / len(picked),
    }

def majority_baseline(train_recs, test_recs):
    hits, total = 0, 0
    for qi in range(len(train_recs[0]["questions"])):
        votes = defaultdict(int)
        for r in train_recs:
            q = r["questions"][qi]
            votes[q["options"][q["label"]]] += 1
        guess = max(votes, key=votes.get)
        for r in test_recs:
            q = r["questions"][qi]
            hits += q["options"][q["label"]] == guess
            total += 1
    return hits / total


import statistics as st, io, contextlib

def chance_accuracy(examples):
    return sum(1 / len(e["options"]) for e in examples) / len(examples)

def train_head2(examples, epochs, lr, seed, d_head=256, wd=0.01):
    torch.manual_seed(seed)
    head = PointerHead(HIDDEN, d_head=d_head).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    rng = random.Random(seed)
    for epoch in range(epochs):
        order = list(range(len(examples))); rng.shuffle(order)
        for i in order:
            ex = examples[i]
            optimizer.zero_grad()
            loss = F.cross_entropy(head(ex["opt"], ex["dec"]).unsqueeze(0), torch.tensor([ex["label"]], device=device))
            loss.backward(); optimizer.step()
    return head

all_train = build_examples(train_records, n_shuffles=6, seed=1)
SEEDS = range(5)
print("random-guess accuracy on held-out: %.3f\n" % chance_accuracy(heldout_examples))

print("A) LEARNING CURVE: held-out accuracy vs number of training records (lr 1e-4, 40 epochs, 5 seeds)")
for k in [1, 2, 4, 6, 8]:
    ex = [e for e in all_train if e["rec"] < k]
    accs, trs = [], []
    for seed in SEEDS:
        h = train_head2(ex, epochs=40, lr=1e-4, seed=seed)
        accs.append(evaluate(h, heldout_examples)["acc"]); trs.append(evaluate(h, ex)["acc"])
    print(f"  {k} records ({len(ex):3d} examples): train acc {st.mean(trs):.3f} | held-out acc {st.mean(accs):.3f} ± {st.pstdev(accs):.3f}")

print("\nB) REGULARISATION at 8 records (5 seeds)")
variants = [
    ("baseline (d_head=256, wd=0.01, 40 ep)", dict(epochs=40, lr=1e-4, d_head=256, wd=0.01)),
    ("stronger weight decay (wd=1.0)",        dict(epochs=40, lr=1e-4, d_head=256, wd=1.0)),
    ("smaller head (d_head=8)",               dict(epochs=40, lr=1e-4, d_head=8, wd=0.01)),
    ("early stop (3 epochs)",                 dict(epochs=3,  lr=1e-4, d_head=256, wd=0.01)),
    ("early stop (1 epoch)",                  dict(epochs=1,  lr=1e-4, d_head=256, wd=0.01)),
]
for name, kw in variants:
    accs, trs = [], []
    for seed in SEEDS:
        h = train_head2(all_train, seed=seed, **kw)
        accs.append(evaluate(h, heldout_examples)["acc"]); trs.append(evaluate(h, all_train)["acc"])
    print(f"  {name:42s} train {st.mean(trs):.3f} | held-out {st.mean(accs):.3f} ± {st.pstdev(accs):.3f}")

print("\nC) what do the cached features look like? (why memorising is easy)")
X = torch.stack([e["dec"] for e in all_train if e["options"] == None or True])
print("  decide vectors:", tuple(X.shape), "| distinct (text, question) pairs in train:", 8 * 3, "| head parameters: 524,288")
