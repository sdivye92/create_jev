
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

def chance_accuracy(examples):
    return sum(1 / len(e["options"]) for e in examples) / len(examples)

print("random-guess accuracy on held-out:   %.3f" % chance_accuracy(heldout_examples))
print("majority-label baseline on held-out: %.3f" % majority_baseline(train_records, heldout_records))

t0 = time.time()
train_examples = build_examples(train_records, n_shuffles=6, seed=1)
print(len(train_examples), "train examples cached in %.1fs\n" % (time.time() - t0))

t0 = time.time()
head = train_head(train_examples, epochs=40, lr=1e-4, seed=0)
print("\ntraining took %.1fs" % (time.time() - t0))

for name, ex in [("train", train_examples), ("held-out", heldout_examples)]:
    r = evaluate(head, ex)
    print(f"\n{name}: accuracy {r['acc']:.3f}  | order stability {r['stability']:.3f}")
    print("  by type:", {t: round(a, 3) for t, a in r["acc_by_type"].items()})

import io, contextlib, statistics

for n in [1, 6]:
    ex = build_examples(train_records, n_shuffles=n, seed=1)
    accs, stabs = [], []
    for seed in range(3):
        with contextlib.redirect_stdout(io.StringIO()):
            h = train_head(ex, epochs=40 if n == 6 else 240, lr=1e-4, seed=seed)
        r = evaluate(h, heldout_examples)
        accs.append(r["acc"])
        stabs.append(r["stability"])
    print(f"n_shuffles={n}: held-out accuracy {statistics.mean(accs):.3f} (sd {statistics.pstdev(accs):.3f}) | order stability {statistics.mean(stabs):.3f} (sd {statistics.pstdev(stabs):.3f})")