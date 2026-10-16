import json

cells = []
def md(s):   cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))

md(r'''
# 06 — Training the head on the smoke set

**Where we are.** Notebook 5 gave us a `JevModel`: a frozen 853M backbone plus a 524k-parameter pointer head, with a verified gradient path. Nothing has been trained yet. This notebook trains the head and measures what it learned.

**What's new here, compared to training the CNNs you know:**
1. **Variable-size outputs.** Each question has its own number of options, so there is no fixed-size output layer and no fixed-size batch tensor. The loss is a cross-entropy over however many options that question has.
2. **Label bookkeeping under shuffling.** To stop the model learning positional shortcuts (notebook 5), we show each question in shuffled option orders, and the label has to follow its option. This is where silent bugs live, so you'll write it and test it.
3. **Feature caching.** The backbone is frozen, so its output for a given input never changes. We run it once per input and train the head on the saved vectors, which turns each epoch into about a tenth of a second.
4. **Evaluating honestly.** We hold out whole records, compare against a trivial baseline, and measure the order-stability property from notebook 5.

**The plan:** (1) load a 12-record dataset and split it, (2) write `shuffle_question`, (3) cache the backbone features, (4) write the training step, (5) evaluate, (6) run an experiment on whether the shuffling helps.

> **GPU memory:** shut down the kernel from notebook 5 before running this one (restart it or close its tab), since each one holds a copy of the model.
''')

code(r'''
%load_ext autoreload
%autoreload 2

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
''')

md(r'''
### The code now lives in a module

The delimiters, sanitizer and `render_question` from notebook 4 are now `jev/render.py` (a `Renderer` class), and your `PointerHead` and `JevModel` from notebook 5 are in `jev/model.py`. I checked that the module produces exactly the same token ids as the notebook version on all 36 questions in the dataset. `%autoreload 2` means that if you edit those files, the kernel picks up the change without a restart.
''')

md(r'''
## Step 1: the data and the split

`data/smoke_12.jsonl` has 12 customer-support records, each with the same three questions (department, escalate, frustration). The first 8 are for training and the last 4 are held out.

**Why split by record and not by question?** All three questions of a record share the same `state` text. If question 1 of a record were in training and question 2 in the held-out set, the model would have already seen that exact state, and the held-out score would look better than it deserves. Splitting by record keeps unseen texts unseen.
''')

code(r'''
with open("../data/smoke_12.jsonl") as f:
    smoke = [json.loads(line) for line in f]
train_records, heldout_records = smoke[:8], smoke[8:]
print(len(train_records), "train records,", len(heldout_records), "held-out records")

for name, recs in [("train", train_records), ("held-out", heldout_records)]:
    print(f"\n{name}: label counts per question")
    for qi, qid in enumerate(["department", "escalate", "frustration"]):
        opts = recs[0]["questions"][qi]["options"]
        counts = [sum(r["questions"][qi]["label"] == k for r in recs) for k in range(len(opts))]
        print(f"  {qid:12s}", dict(zip(opts, counts)))
''')

md(r'''
## Step 2: shuffling the options, and moving the label with them

Write `shuffle_question(q, rng)`. It takes one question dict (with `options` and `label`) and returns a **new** dict with the options in a random order and the `label` updated to point at the same option text as before. Two rules:
- Do not modify the input dict (we reuse the original question many times).
- Questions of type `"score"` are returned unchanged. Their options are ordered (calm < frustrated < furious) and we agreed in notebook 4 to keep that convention fixed, so shuffling them would destroy it.

Hints: `rng.shuffle(perm)` shuffles a list of indices in place. If `perm[new_position] = old_position`, then the new label is the `new_position` where `old_position == q["label"]`, which is `perm.index(q["label"])`.
''')

code(r'''
def shuffle_question(q, rng):
    """Return a copy of q with its options shuffled and the label moved to follow its option."""
    # TODO
    raise NotImplementedError("fill in shuffle_question")
''')

md(r'''
<details><summary>Solution (open after you've tried)</summary>

```python
if q["type"] == "score":
    return dict(q)
perm = list(range(len(q["options"])))
rng.shuffle(perm)
return {**q, "options": [q["options"][i] for i in perm], "label": perm.index(q["label"])}
```

`perm[new_position]` is the old position of the option now sitting at `new_position`, so `perm.index(old_label)` finds where the correct option landed.

</details>
''')

md(r'''
A function like this can be wrong and still run, so test the one property that matters: **the option at the new label must be the same text as the option at the old label.** Run it many times over every question in the dataset.
''')

code(r'''
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
''')

md(r'''
## Step 3: cache the backbone features

The backbone is frozen and deterministic, so for a given rendered question its output at the readout positions is always the same. Running the 0.8B model on every training step would waste almost all of the time. Instead, run it once per (question, option order), keep the readout vectors, and train the head on those.

`build_examples` does this. For each question it makes `n_shuffles` versions: the original order first, then shuffled ones. The result is a list of small dicts holding the option vectors, the decide vector, the label, and some bookkeeping used by the evaluation.

**One thing to notice for later:** this trick only works while the backbone is frozen. When we add LoRA, the backbone's output changes every step and the cache becomes invalid.
''')

code(r'''
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
''')

md(r'''
## Step 4: the training step

You know this loop from your CNN days; the only new part is that there is no batch tensor, so each example is scored on its own (a small loop, which is fine at this size; padding and masking would be the way to batch it later).

Fill in the body of the inner loop: zero the gradients, score the example with the head, take the cross-entropy against `ex["label"]`, backpropagate, and step the optimizer. `F.cross_entropy` expects logits of shape `[batch, classes]` and a target of shape `[batch]`, so the scores need a batch dimension of 1. Leave the result in a variable called `loss`.
''')

code(r'''
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
            # TODO: zero grads, score ex["opt"] and ex["dec"] with the head, cross-entropy against ex["label"], backward, optimizer step. Keep the result in `loss`.
            raise NotImplementedError("fill in the training step")
            total += loss.item()
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:3d}  mean loss {total / len(examples):.4f}")
    return head
''')

md(r'''
<details><summary>Solution (open after you've tried)</summary>

```python
optimizer.zero_grad()
scores = head(ex["opt"], ex["dec"])
loss = F.cross_entropy(scores.unsqueeze(0), torch.tensor([ex["label"]], device=device))
loss.backward()
optimizer.step()
```

</details>
''')

md(r'''
## Step 5: evaluation

Accuracy alone can fool you, so we compute three things:
1. **Accuracy**, overall and per question type.
2. **Two baselines.** *Random guessing* is what you score knowing nothing (1/3 on a three-option question, 1/2 on yes/no). *The majority label* is what you score by always predicting the option text that was most common in the training records. If the model can't beat both, it has learned nothing about the text. The two can differ, because the held-out label mix isn't the training label mix.
3. **Order stability**, the measurement we promised in notebook 5. For each held-out question we have 6 option orders. Did the model pick the same option text under all of them? Note that `score` questions are never shuffled, so they are trivially stable and set a floor of one third.
''')

code(r'''
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
''')

md(r'''
### Train, then look at the numbers

Train on the 8 training records with 6 option orders each, then evaluate on both sets.
''')

code(r'''
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
''')

md(r'''
**Read the output before moving on, and answer in a comment:**
1. Look at the `epoch 0` line. It is the mean loss over the first epoch, and a model that knows nothing scores about `ln(3) = 1.1`. How does yours compare? The vectors have norms around 110 (notebook 5) and AdamW moves every weight by roughly `lr` per step, so think about what that does to the scores early on. Then rerun with `lr=1e-3` and `lr=3e-5` and compare the `epoch 0` lines.
2. Training accuracy reaches 1.0 and the loss goes to about 0. What does that tell you about the plumbing, and what does it *not* tell you?
3. Compare held-out accuracy with *both* baselines. With only 12 held-out questions each one is worth about 8 points, so is the gap big enough to mean anything?
4. Which question type is weakest on held-out data, and is that worth reading anything into?

<details><summary>My answers (from my runs; yours may differ in the last digit)</summary>

1. Averaged over 3 seeds, the epoch-0 mean loss was about 21 to 50 at `lr=1e-3` (the number depends on how many orders are used), about 2.5 to 2.8 at `1e-4`, and about 1.3 to 1.7 at `3e-5` (close to `ln 3`). A single run varies: seed 0 at `1e-4` prints 1.76. A step of `lr` per weight moves each projected value by roughly `lr` times the sum of the input's absolute values, and for norm-110 inputs that sum is in the thousands, so the scores swing by whole units per step and the softmax is confidently wrong early on. Lowering the learning rate fixes it; normalising the vectors before the head would also work.
2. It tells you the loop works end to end: gradients reach the head, the loss falls, and the head can fit 144 examples. It does not tell you the model generalises, because 8 distinct texts are easy to memorise with half a million parameters.
3. Across learning rates and seeds my held-out accuracy ranged from 0.44 to 0.58 against 0.39 for random guessing, with up to 0.1 spread between seeds. That is perhaps a little above chance, but it rests on 12 independent questions, so I would not claim the head generalises. The majority baseline (0.33) is *below* chance here only because the held-out label mix differs from the training mix, which is why random guessing is the more honest reference.
4. With this few questions, a per-type difference is noise. Don't read anything into it.

</details>
''')

md(r'''
## Step 6: does the shuffling actually help?

We added shuffling because of an argument (notebook 5), not a measurement. Here is the measurement: train with no shuffling (`n_shuffles=1`, original order only) and with 6 orders, then compare accuracy and order stability on the held-out set. Held-out always has 6 orders, so the stability numbers are comparable.

Each setting is trained with 3 seeds and averaged, because one seed can mislead badly at this data size. Predict before you run: which of the two should be more order-stable, and by how much do you expect accuracy to differ?
''')

code(r'''
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
''')

md(r'''
The number of epochs differs (240 vs 40) on purpose: with 1 order there are 6 times fewer examples per epoch, so this gives both runs about the same number of gradient steps. Otherwise the comparison would be about training length and not about shuffling.

**Discussion:**
- What did shuffling change, and was it what you predicted?
- With only 4 held-out records, how much should you trust the difference? What would you do to get a more trustworthy answer?

<details><summary>What I saw</summary>

With 3 seeds at `lr=1e-4`, held-out accuracy was about 0.50 without shuffling and 0.51 with it (no real difference), while order stability rose from about 0.53 to about 0.81. The same pattern held at `lr=3e-4` (0.53 to 0.86) and `3e-5` (0.53 to 0.78). So shuffling bought order-robustness at no cost in accuracy, which is what the notebook 5 argument predicted.

Two cautions. First, my very first run used `lr=1e-3` with one seed and showed the opposite (stability 0.67 without shuffling, 0.33 with), purely because training at that learning rate was unstable. A single seed at a bad setting can reverse a real effect, which is why this cell averages seeds. Second, stability here is measured on 12 held-out questions, 4 of which (the `score` ones) are never shuffled, so treat the size of the gap as indicative and the direction as the finding. A trustworthy answer needs many more held-out records.

</details>
''')

md(r'''
## Wrap-up

You now have a complete training pipeline: render, cache backbone features, train a pointer head with shuffled orders and correctly moved labels, and evaluate against a baseline with an order-stability check. Everything up to here runs in seconds because the backbone is frozen.

What we learned, in order of how much to trust it: (1) the learning rate matters a lot because the backbone vectors are large, and `1e-4` is sane where `1e-3` is not; (2) shuffling options with correctly moved labels makes the answer much less sensitive to option order at no cost in accuracy; (3) held-out accuracy is only modestly above chance, but 8 training records and 12 held-out questions cannot support a claim either way.

The structural limit is that a frozen backbone gives the head fixed vectors, and the head can only re-weight what is already in them. Whether that is enough depends on the task and on having far more data than we have. That is exactly the question the next notebook takes up.

**Next, notebook 7:** LoRA. Instead of freezing the whole backbone, we add small trainable low-rank adapters inside it, so the model can change *what it puts into* the readout vectors. The price is that the cache trick stops working (the backbone output changes every step), so training gets much slower, and we'll need a larger dataset to say anything real.
''')

nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "create_jev (Qwen3.5)", "language": "python", "name": "create_jev"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for kind, src in cells:
    cell = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    nb["cells"].append(cell)

out = "/home/divye/Documents/Projects/create_jev/notebooks/06_training_the_head.ipynb"
with open(out, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", out, "with", len(cells), "cells")
