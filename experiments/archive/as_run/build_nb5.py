import json

cells = []
def md(s):   cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))

md(r'''
# 05 — Model surgery: from "text generator" to "decision reader"

**Where we are.** Notebook 4 gave us `render_question`, which turns one (state, question) pair into token ids plus two kinds of positions: `opt_end_positions` (where each option's `<|box_end|>` sits) and `decide_position` (where the final `<|fim_suffix|>` sits). Nothing reads those positions yet. This notebook builds the part of the model that does.

**Three pieces of surgery:**
1. **Throw away what we don't use**: the `lm_head`, which turns a hidden vector into a score for every vocabulary token. We never generate text.
2. **Choose the readout**: which hidden state do we read, and what is it allowed to have seen?
3. **Add a new head** (a "pointer head") that scores each option against the decision vector.

**What you'll have at the end:** a `JevModel` that takes a rendered question and returns one score per option, plus proof that gradients reach the head and not the frozen backbone.

> **GPU memory:** the kernels from notebooks 02-04 each hold a copy of the model. On an 8 GB card, shut those kernels down before loading another one here.
''')

code(r'''
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import json, math, re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL_ID = "Qwen/Qwen3.5-0.8B"
processor = AutoProcessor.from_pretrained(MODEL_ID)
tokenizer = processor.tokenizer
model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto").eval()
print(type(model).__name__, "| GPU memory allocated: %.2f GB" % (torch.cuda.memory_allocated() / 1e9))
''')

md(r'''
### Carried over from notebook 4 (unchanged)

The delimiters, the sanitizer, and `render_question`, copied so this notebook runs on its own. When we write the training notebook we'll move this into a module so there is a single copy.
''')

code(r'''
DELIMITERS = {
    "state":   "<|fim_prefix|>",
    "q":       "<|fim_middle|>",
    "opt":     "<|box_start|>",
    "opt_end": "<|box_end|>",
    "decide":  "<|fim_suffix|>",
}
DELIM_IDS = {tokenizer(s, add_special_tokens=False).input_ids[0] for s in DELIMITERS.values()}

_SPECIAL_RE = re.compile(r"<\|([^|<>]{1,64})\|>")

def sanitize_user_text(text):
    return _SPECIAL_RE.sub(lambda m: f"<¦{m.group(1)}¦>", text)

def tok_user(s):
    ids = tokenizer(sanitize_user_text(s), add_special_tokens=False).input_ids
    return [i for i in ids if i not in DELIM_IDS]

def render_question(record, qi, tokenizer=tokenizer, delimiters=DELIMITERS):
    def delim(name):
        return tokenizer(delimiters[name], add_special_tokens=False).input_ids
    q = record["questions"][qi]
    ids, opt_end_positions = [], []
    ids += delim("state");  ids += tok_user(record["state"])
    ids += delim("q");      ids += tok_user(q.get("prompt", q["id"]))
    for opt in q["options"]:
        ids += delim("opt"); ids += tok_user(opt); ids += delim("opt_end")
        opt_end_positions.append(len(ids) - 1)
    ids += delim("decide")
    return ids, opt_end_positions, len(ids) - 1

smoke = [json.loads(line) for line in open("../data/smoke.jsonl")]
print(len(smoke), "smoke record(s) loaded")
''')

md(r'''
## Surgery 1: what do we throw away?

`Qwen3_5ForConditionalGeneration` is two things stacked:
- `.model` (`Qwen3_5Model`): the vision tower plus the 24-layer language stack. Produces one hidden vector per token.
- `.lm_head`: a `Linear(1024 -> 248,320)` that turns each hidden vector into a score for every token in the vocabulary. This is what you need to *generate* text.

We don't generate, so we never call `lm_head`. Answer these two questions in a comment before you run the next cell:
1. In notebook 1 you found `lm_head` is tied to the input embedding. If we stop using `lm_head`, do we save any *parameters*? Could we also delete the embedding matrix?
2. If not parameters, what *do* we save?
''')

code(r'''
print("lm_head tied to input embedding?", model.lm_head.weight is model.get_input_embeddings().weight)

# Surgery 1: from now on we only ever call the inner model.
core = model.model
print("core:", type(core).__name__)
print("core params: {:,}".format(sum(p.numel() for p in core.parameters())))

# What would lm_head's output cost during training? Change B and T to your own guess.
B, T, V = 8, 400, model.config.text_config.vocab_size
print(f"logits for B={B}, T={T}: {B*T*V*2/1e9:.2f} GB in bf16, {B*T*V*4/1e9:.2f} GB in fp32")
''')

md(r'''
<details><summary>My answers (open after you've tried)</summary>

1. **No parameters saved.** The head's weight *is* the embedding matrix (same tensor), and the embedding is still needed on the way *in* to turn token ids into vectors. So `core` has all 852,985,920 parameters either way. Skipping `lm_head` just means we never use that tensor a second time.
2. **We save compute and activation memory.** Running `lm_head` over every position makes a `[batch, tokens, 248,320]` logits tensor, which is gigabytes at training sizes and is then upcast to fp32 for the loss. We only need the hidden vectors at a handful of positions.

</details>
''')

md(r'''
## Surgery 2: which hidden state do we read?

`core(input_ids=...)` returns `last_hidden_state`, shape `[batch, tokens, 1024]`: the output of layer 24 after the final norm. Two things to check and think about:

**(a) Is the last entry of `hidden_states` the same tensor as `last_hidden_state`?** In notebook 3 we compared `hidden_states[-1]` as if it were the final output. Verify that was legitimate.

**(b) What has each readout position seen?** The backbone is *causal*. The `<|box_end|>` after option 0 has seen the state, the question and option 0, but not options 1 and 2. The `<|fim_suffix|>` decide token comes last, so it has seen everything. Keep that asymmetry in mind; we come back to it.
''')

code(r'''
device = next(core.parameters()).device
ids, opt_end_positions, decide_position = render_question(smoke[0], 0)
x = torch.tensor([ids], device=device)

with torch.no_grad():
    out = core(input_ids=x, output_hidden_states=True)

print("last_hidden_state shape:", tuple(out.last_hidden_state.shape))
print("hidden_states[-1] is last_hidden_state?", torch.equal(out.hidden_states[-1], out.last_hidden_state))

h = out.last_hidden_state[0].float()     # [T, D]
opt_vecs = h[opt_end_positions]          # [num_options, D]  one vector per </opt>
dec_vec = h[decide_position]             # [D]               the <decide> vector
print("opt_vecs:", tuple(opt_vecs.shape), "| dec_vec:", tuple(dec_vec.shape))
print("norms: options", [round(n, 1) for n in opt_vecs.norm(dim=-1).tolist()], "| decide", round(dec_vec.norm().item(), 1))
''')

md(r'''
Now the experiment that decides how the head must be built. The simplest possible "head" would just compare the raw vectors: `score_k = opt_vec_k . dec_vec`. Look at what that gives before we commit to anything.
''')

code(r'''
unit = F.normalize(opt_vecs, dim=-1)
print("cosine between the </opt> vectors:\n", (unit @ unit.T).cpu().numpy().round(3))
print("cosine(option_k, decide):", [round(v, 3) for v in (unit @ F.normalize(dec_vec, dim=-1)).tolist()])

raw = opt_vecs @ dec_vec
scaled = raw / math.sqrt(opt_vecs.shape[-1])
print("raw dot products      :", [round(v) for v in raw.tolist()])
print("divided by sqrt(1024) :", [round(v, 1) for v in scaled.tolist()])
print("softmax of that       :", [round(v, 4) for v in scaled.softmax(-1).tolist()])
''')

md(r'''
**Questions (answer in your own words before opening mine):**
1. The three `</opt>` vectors are very similar to each other. All three are the *same token* sitting in almost the same context. What does that say about where the information that tells the options apart lives in these 1024 numbers?
2. The logits are in the hundreds. What does a softmax do with values that far apart, and why is that a problem both for learning and for calibrated probabilities later?
3. So what should the head do to the vectors *before* the dot product, and what should it do *after*?

<details><summary>My answers</summary>

1. Most of each vector is shared: "I am an option-end delimiter in this document". What distinguishes the options is a small residual on top of that. A raw dot product is dominated by the shared part, so it can't pick out the difference.
2. With gaps of tens of units, softmax puts essentially all probability on one option (on my run: `[1.0, 0.0, 0.0]`), and the gradient to the others is about zero. Training stalls and probabilities are all-or-nothing, which is the opposite of calibrated.
3. **Before:** learned linear projections (one for options, one for the decision vector) so training can pick out the directions that matter and drop the shared part. **After:** divide by `sqrt(d_head)`. A dot product of two `d`-dimensional vectors has a spread that grows like `sqrt(d)`; dividing keeps the logits at a sane scale at initialization.

</details>
''')

md(r'''
## Surgery 3: the pointer head

**Why a pointer head instead of an ordinary classifier?** A classifier has a fixed output layer, e.g. `Linear(1024, 3)`, so output 0 always means the same thing. But our options are different every question: 2 options or 7, "billing" or "calm". The head can't have one output per option *identity*. Instead it scores each option by looking at **that option's own vector**, so it works for any number of options in any order. It "points" at one of the options in the input.

**The design**, from the questions above:
```
score_k = ( W_opt . opt_vec_k ) . ( W_dec . dec_vec ) / sqrt(d_head)
```

Your task: fill in `forward`. It's three lines. Shapes are in the docstring.
''')

code(r'''
class PointerHead(nn.Module):
    def __init__(self, hidden_size, d_head=256):
        super().__init__()
        self.opt_proj = nn.Linear(hidden_size, d_head, bias=False)
        self.dec_proj = nn.Linear(hidden_size, d_head, bias=False)
        self.d_head = d_head

    def forward(self, opt_vecs, dec_vec):
        """opt_vecs: [num_options, hidden_size], dec_vec: [hidden_size]  ->  scores: [num_options]"""
        # TODO: project both, take each option's dot product with the decision vector, scale.
        raise NotImplementedError("fill in the scoring function")
''')

md(r'''
<details><summary>Solution (open after you've tried)</summary>

```python
o = self.opt_proj(opt_vecs)        # [num_options, d_head]
d = self.dec_proj(dec_vec)         # [d_head]
return (o @ d) / math.sqrt(self.d_head)
```

`o @ d` is a matrix times a vector: one dot product per option, so the result is `[num_options]`.

</details>
''')

md(r'''
### Putting the pieces together

`JevModel` is deliberately thin: run the backbone, pick the vectors at the readout positions, hand them to the head. The `.float()` matters: the backbone runs in bf16, but the head is tiny and trains better in fp32.
''')

code(r'''
class JevModel(nn.Module):
    def __init__(self, core, head):
        super().__init__()
        self.core = core
        self.head = head

    def forward(self, input_ids, opt_positions, decide_position):
        h = self.core(input_ids=input_ids).last_hidden_state[0].float()   # [T, D]; batch of 1 for now
        return self.head(h[opt_positions], h[decide_position])            # [num_options]

torch.manual_seed(0)
head = PointerHead(model.config.text_config.hidden_size).to(device)     # fp32
jev = JevModel(core, head)

with torch.no_grad():
    scores = jev(x, opt_end_positions, decide_position)
print("options:", smoke[0]["questions"][0]["options"])
print("scores :", [round(v, 3) for v in scores.tolist()])
print("probs  :", [round(v, 3) for v in scores.softmax(-1).tolist()], "(untrained head, so arbitrary)")
''')

md(r'''
## An experiment on the causal asymmetry

The head scores every option with the same function, so it treats the options symmetrically. But the *inputs* to the head come from a causal backbone: option 2's vector has seen options 0 and 1, option 0's has seen neither. So reordering the options should change the vectors.

Run it: reverse the option order, map the scores back to the original order, and compare. Predict first: will the scores match?
''')

code(r'''
def option_scores(record, qi):
    ids, ends, dpos = render_question(record, qi)
    xx = torch.tensor([ids], device=device)
    with torch.no_grad():
        hh = core(input_ids=xx).last_hidden_state[0].float()
        return head(hh[ends], hh[dpos]), hh[ends]

rec_rev = json.loads(json.dumps(smoke[0]))
rec_rev["questions"][0]["options"] = rec_rev["questions"][0]["options"][::-1]

s_orig, v_orig = option_scores(smoke[0], 0)
s_rev, v_rev = option_scores(rec_rev, 0)
s_back, v_back = s_rev.flip(0), v_rev.flip(0)     # put back into the original option order

print("scores, original order          :", [round(v, 3) for v in s_orig.tolist()])
print("scores, reversed (mapped back)  :", [round(v, 3) for v in s_back.tolist()])
print("same option's </opt> vector, cosine across the two orders:",
      [round(F.cosine_similarity(a, b, dim=0).item(), 3) for a, b in zip(v_orig, v_back)])
''')

md(r'''
**Discussion:**
- Why do the scores change if the head is symmetric? (Look at the cosine line.)
- Is this a bug to fix, or a property to manage? You can't remove it without giving up the causal backbone, so what could you do during *training* to stop the model from learning "the first option is usually right" or from being fragile to ordering? (Hint: any2jev's README calls this out. The labels have to move with the options.)
- This is also something to *measure* after training: how often does the argmax change when the same options are shuffled?
''')

md(r'''
## Freeze the backbone and check the gradient path

The cheapest first training run keeps the 853M backbone frozen and trains only the head. Verify the wiring before spending any time on training: after one forward and backward pass, the head should have gradients and the backbone should have none.
''')

code(r'''
for p in core.parameters():
    p.requires_grad_(False)

logits = jev(x, opt_end_positions, decide_position)
label = torch.tensor([smoke[0]["questions"][0]["label"]], device=device)
loss = F.cross_entropy(logits.unsqueeze(0), label)
loss.backward()

print("loss: %.3f   (ln 3 = %.3f is what a uniform guess scores; an untrained head can land on either side by luck)" % (loss.item(), math.log(3)))
print("head got gradients?    ", all(p.grad is not None for p in head.parameters()))
print("any backbone gradient? ", any(p.grad is not None for p in core.parameters()))
trainable = sum(p.numel() for p in jev.parameters() if p.requires_grad)
total = sum(p.numel() for p in jev.parameters())
print(f"trainable: {trainable:,} of {total:,} ({100 * trainable / total:.3f}%)")
''')

md(r'''
## Wrap-up

You now have:
- The `lm_head` out of the picture (and a clear reason why that saves memory but not parameters).
- A readout: `last_hidden_state` at the `<|box_end|>` positions and the `<|fim_suffix|>` position.
- A `PointerHead` you wrote, designed from evidence (vector scale and similarity) rather than copied.
- A `JevModel` whose gradient path is verified: head trains, backbone frozen.
- A known property to manage: option order changes the option vectors because the backbone is causal.

We kept the vision tower inside `core` on purpose. It's unused for text states, but it's what lets a state contain an image later.

**Next, notebook 6:** train this head on the smoke set. The test is overfitting: with only a handful of records, accuracy should reach 1.0, which proves the whole pipeline (render, backbone, head, loss, optimizer) works. Then we ask whether a frozen backbone is enough on real data, which leads into LoRA.
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

out = "/home/divye/Documents/Projects/create_jev/notebooks/05_model_surgery.ipynb"
with open(out, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", out, "with", len(cells), "cells")
