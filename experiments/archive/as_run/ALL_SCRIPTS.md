# Experiment scripts, as they were run

These are the scripts used during the exploration, copied verbatim from the scratch folder and are listed in the order they were written. Nothing was cleaned up, so they contain hard-coded paths (the scratch folder, `/tmp/nb7_feats.pt`) and some depend on module APIs that have since changed. Each entry says which results it produced (the section numbers refer to `results/RESULTS.md`) and whether it still runs.

The supported way to run training and evaluation today is the command line in `jev/README.md`. The scripts below are the record of how the numbers in `results/RESULTS.md` were obtained.

Where results came from, in short: small-data experiments from `nb6_*`; scaling experiments from `nb7_*` (the `_proto` ones are the constant-learning-rate prototypes, `nb7_run.py` is the final version); the bf16 versus fp32 finding from `fp32test.py` and the files in `experiments/`; LoRA memory facts from `lora_*`; the vision interface from `vis_probe.py`.

| Script | Group | Status | Results section |
|---|---|---|---|
| `instr.py` | Cache-reuse investigation | runs | Section 1.1 |
| `fp32test.py` | Cache-reuse investigation | runs | Section 1.1 |
| `norm_test.py` | Cache-reuse investigation | runs | Section 1.1 |
| `nb5_probe.py` | Readout vectors before training | runs | Section 1.2 |
| `nb5_run.py` | Notebook check | runs | Section 1.2 |
| `build_nb5.py` | Notebook generator | runs | - |
| `nb6_sweep.py` | Small-data experiments | runs | Section 2.3 |
| `nb6_run.py` | Small-data experiments | runs | Sections 2.2 and 2.4 |
| `build_nb6.py` | Notebook generator | runs | - |
| `nb6_curve.py` | Small-data experiments | runs | Sections 2.5 and 2.6 |
| `gen_raw.py` | Data | runs | Section 3.1 |
| `nb7_proto.py` | Scaling experiments (prototype) | runs | Sections 3.2 and 3.4 (single-seed table) |
| `nb7_proto2.py` | Scaling experiments (prototype) | runs | Sections 3.4 to 3.8 (constant-rate versions) |
| `nb7_proto3.py` | Scaling experiments (prototype) | runs | Section 3.4 (schedule table) |
| `nb7_errs.py` | Scaling experiments (prototype) | runs | Section 3.7 |
| `build_nb7.py` | Notebook generator | runs | - |
| `nb7_run.py` | Scaling experiments (final) | runs | Sections 3.3 to 3.8 |
| `lora_probe.py` | LoRA feasibility | stale | Section 8 (memory facts) |
| `lora_mem.py` | LoRA feasibility | stale | Section 8 |
| `vis_probe.py` | Vision interface | runs | Section 9.1 |
| `smoke_modules.py` | Verification | runs | Sections 7 to 9 (verification) |
| `run_nb.py` | Verification | runs | - |
| `build_nb8.py` | Notebook generator | runs | - |
| `build_nb9.py` | Notebook generator | runs | - |

## `instr.py`

Instruments the model's internals to see what positions and cache lengths the continuation actually uses. Written at 14:25. Group: Cache-reuse investigation. Results: Section 1.1.

````python
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import copy, torch
from transformers import AutoProcessor, AutoModelForMultimodalLM
import transformers.models.qwen3_5.modeling_qwen3_5 as M

processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B")
model = AutoModelForMultimodalLM.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16, device_map="auto").eval()
tok = processor.tokenizer
DEV = next(model.parameters()).device

state_text = "Customer ticket: payouts have failed three days in a row, and the bank says nothing is wrong on their end."
question_a = " Question: which team should handle this — billing, technical, or sales?"

state_ids = torch.tensor([tok(state_text, add_special_tokens=False)["input_ids"]], device=DEV)
L = state_ids.shape[1]
full = torch.tensor([tok(state_text+question_a, add_special_tokens=False)["input_ids"]], device=DEV)
q_only = full[:, L:]
qn = q_only.shape[1]

# monkeypatch TextModel.forward to log position_ids fed
orig = M.Qwen3_5TextModel.forward
logs = {}
def patched(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, use_cache=None, **kw):
    logs.setdefault('pos', []).append(None if position_ids is None else (tuple(position_ids.shape), position_ids.reshape(position_ids.shape[0], -1)[:, :3].tolist(), position_ids.reshape(position_ids.shape[0], -1)[:, -3:].tolist()))
    return orig(self, input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, **kw)
M.Qwen3_5TextModel.forward = patched

with torch.no_grad():
    ref = model(input_ids=full, output_hidden_states=True, use_cache=False)
    print("BASELINE pos fed:", logs['pos'][-1])
    so = model(input_ids=state_ids, use_cache=True)
    cache = so.past_key_values
    cl = copy.deepcopy(cache)
    print("clone get_seq_length:", cl.get_seq_length())
    logs['pos'].clear()
    cont = model(input_ids=q_only, past_key_values=cl, use_cache=True, output_hidden_states=True)
    print("CONT pos fed:", logs['pos'][-1])

rl = ref.hidden_states[-1][:, -qn:]
clast = cont.hidden_states[-1]
print("last-layer per-position max-abs-diff:")
for j in range(qn):
    d = (rl[0,j]-clast[0,j]).abs().max().item()
    n = rl[0,j].abs().max().item()
    print(f"  qpos {j}: diff={d:.4f} ref_absmax={n:.3f}")
print("last-token rel (mean):", ((rl[0,-1]-clast[0,-1]).abs()/ (rl[0,-1].abs()+1e-6)).mean().item())

# which dims blow up at last token
dd = (rl[0,-1]-clast[0,-1]).abs()
top = torch.topk(dd, 5)
print("top-5 diff dims at last token:", top.values.tolist(), top.indices.tolist())
print("  ref at those dims:", rl[0,-1][top.indices].tolist())
print("  cont at those dims:", clast[0,-1][top.indices].tolist())
````

## `fp32test.py`

Runs the cache-reuse comparison in fp32 and bf16. Source of the finding that the 0.375 gap was bf16 rounding (fp32 gap about 2e-5). Written at 14:25. Group: Cache-reuse investigation. Results: Section 1.1.

````python
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import copy, torch
from transformers import AutoProcessor, AutoModelForMultimodalLM

processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B")

def runtest(dtype, label):
    model = AutoModelForMultimodalLM.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=dtype, device_map="auto").eval()
    tok = processor.tokenizer
    DEV = next(model.parameters()).device
    state_text = "Customer ticket: payouts have failed three days in a row, and the bank says nothing is wrong on their end."
    qa = " Question: which team should handle this — billing, technical, or sales?"
    qb = " Question: is this urgent and does it need escalation?"
    state_ids = torch.tensor([tok(state_text, add_special_tokens=False)["input_ids"]], device=DEV)
    L = state_ids.shape[1]
    with torch.no_grad():
        so = model(input_ids=state_ids, use_cache=True)
        cache = so.past_key_values
        for name, q in [("A", qa), ("B", qb)]:
            full = torch.tensor([tok(state_text+q, add_special_tokens=False)["input_ids"]], device=DEV)
            q_only = full[:, L:]
            ref = model(input_ids=full, output_hidden_states=True, use_cache=False)
            cl = copy.deepcopy(cache)
            cont = model(input_ids=q_only, past_key_values=cl, use_cache=True, output_hidden_states=True)
            d = (ref.hidden_states[-1][:,-1] - cont.hidden_states[-1][:,-1]).abs().max().item()
            print(f"[{label}] Q{name}: last-token hs[-1] max-abs-diff={d:.6f} refmax={ref.hidden_states[-1][:,-1].abs().max().item():.3f}")
    del model
    torch.cuda.empty_cache()

runtest(torch.float32, "fp32")
````

## `norm_test.py`

Checks the effect of the final normalisation on the compared vectors. Written at 14:27. Group: Cache-reuse investigation. Results: Section 1.1.

````python
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import copy, torch
from transformers import AutoProcessor, AutoModelForMultimodalLM

processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B")
model = AutoModelForMultimodalLM.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16, device_map="auto").eval()
tok = processor.tokenizer
DEV = next(model.parameters()).device
lm = model.model.language_model

state_text = "Customer ticket: payouts have failed three days in a row, and the bank says nothing is wrong on their end."
qa = " Question: which team should handle this — billing, technical, or sales?"
qb = " Question: is this urgent and does it need escalation?"
state_ids = torch.tensor([tok(state_text, add_special_tokens=False)["input_ids"]], device=DEV)
L = state_ids.shape[1]

with torch.no_grad():
    so = model(input_ids=state_ids, use_cache=True)
    cache = so.past_key_values
    for name, q in [("A", qa), ("B", qb)]:
        full = torch.tensor([tok(state_text+q, add_special_tokens=False)["input_ids"]], device=DEV)
        q_only = full[:, L:]
        ref = model(input_ids=full, output_hidden_states=True, use_cache=False)
        cl = copy.deepcopy(cache)
        cont = model(input_ids=q_only, past_key_values=cl, use_cache=True, output_hidden_states=True)
        # pre-norm
        rp = ref.hidden_states[-1][:,-1]; cp = cont.hidden_states[-1][:,-1]
        # post-norm
        rn = lm.norm(ref.hidden_states[-1])[:,-1]; cnn = lm.norm(cont.hidden_states[-1])[:,-1]
        print(f"Q{name}: prenorm diff={ (rp-cp).abs().max().item():.5f} (mag {rp.abs().max().item():.1f}) | postnorm diff={(rn-cnn).abs().max().item():.5f} (mag {rn.abs().max().item():.2f})")
        # argmax token match via lm_head
        lr = model.lm_head(rn.to(model.lm_head.weight.dtype)) if hasattr(model,'lm_head') else None
````

## `nb5_probe.py`

Loads the model, renders one smoke question, and prints vector norms, cosine similarities, raw and scaled dot products, the untrained head's scores, the option-order sensitivity test, and a frozen-backbone gradient check. Written at 21:03. Group: Readout vectors before training. Results: Section 1.2.

````python
import os, re, json, math
os.environ["HF_HUB_DISABLE_XET"]="1"; os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]="1"
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForMultimodalLM

tokenizer = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B").tokenizer
model = AutoModelForMultimodalLM.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16, device_map="auto").eval()
core = model.model
dev = next(core.parameters()).device

DELIMITERS = {"state":"<|fim_prefix|>","q":"<|fim_middle|>","opt":"<|box_start|>","opt_end":"<|box_end|>","decide":"<|fim_suffix|>"}
DELIM_IDS = {tokenizer(s, add_special_tokens=False).input_ids[0] for s in DELIMITERS.values()}
_RE = re.compile(r"<\|([^|<>]{1,64})\|>")
def sanitize_user_text(t): return _RE.sub(lambda m: f"<¦{m.group(1)}¦>", t)
def tok_user(s):
    ids = tokenizer(sanitize_user_text(s), add_special_tokens=False).input_ids
    return [i for i in ids if i not in DELIM_IDS]
def render_question(record, qi, tokenizer=tokenizer, delimiters=DELIMITERS):
    def delim(n): return tokenizer(delimiters[n], add_special_tokens=False).input_ids
    q = record["questions"][qi]; ids, ends = [], []
    ids += delim("state"); ids += tok_user(record["state"]); ids += delim("q"); ids += tok_user(q.get("prompt", q["id"]))
    for opt in q["options"]:
        ids += delim("opt"); ids += tok_user(opt); ids += delim("opt_end"); ends.append(len(ids)-1)
    ids += delim("decide")
    return ids, ends, len(ids)-1

rec = [json.loads(l) for l in open("data/smoke.jsonl")][0]
ids, ends, dpos = render_question(rec, 0)
x = torch.tensor([ids], device=dev)
with torch.no_grad():
    h = core(input_ids=x).last_hidden_state[0].float()
print("h", tuple(h.shape), "ends", ends, "dpos", dpos)
opt = h[ends]; dec = h[dpos]
print("norms opt:", opt.norm(dim=-1).tolist(), "dec:", dec.norm().item())
cos = F.normalize(opt, dim=-1) @ F.normalize(opt, dim=-1).T
print("pairwise cosine between </opt> vectors:\n", cos.cpu().numpy().round(3))
print("cosine(opt_k, decide):", (F.normalize(opt, dim=-1) @ F.normalize(dec, dim=-1)).tolist())
print("raw dot opt.dec:", (opt @ dec).tolist(), " /sqrt(D):", ((opt @ dec)/math.sqrt(1024)).tolist())

class Head(nn.Module):
    def __init__(self, D=1024, d=256):
        super().__init__(); self.q=nn.Linear(D,d,bias=False); self.k=nn.Linear(D,d,bias=False); self.d=d
    def forward(self, opt_vecs, dec_vec):
        return (self.k(opt_vecs) @ self.q(dec_vec)) / math.sqrt(self.d)
torch.manual_seed(0)
head = Head().to(dev)
s = head(opt, dec); print("untrained scores:", s.tolist(), "probs:", s.softmax(-1).tolist())

# order sensitivity: reverse the options (and map scores back)
rec2 = json.loads(json.dumps(rec)); q = rec2["questions"][0]; q["options"] = q["options"][::-1]
ids2, ends2, dpos2 = render_question(rec2, 0)
with torch.no_grad():
    h2 = core(input_ids=torch.tensor([ids2], device=dev)).last_hidden_state[0].float()
s2 = head(h2[ends2], h2[dpos2]).flip(0)   # flip back to original option order
print("scores orig order      :", s.tolist())
print("scores reversed->back  :", s2.tolist())
print("max abs diff:", (s-s2).abs().max().item())
print("</opt> vec diff for same option under reorder (cosine):",
      [F.cosine_similarity(opt[i], h2[ends2].flip(0)[i], dim=0).item() for i in range(3)])

# gradient flow with frozen backbone
for p in core.parameters(): p.requires_grad_(False)
x = torch.tensor([ids], device=dev)
hh = core(input_ids=x).last_hidden_state[0].float()
loss = F.cross_entropy(head(hh[ends], hh[dpos]).unsqueeze(0), torch.tensor([0], device=dev))
loss.backward()
print("head grad ok:", head.q.weight.grad is not None, "| core grad:", any(p.grad is not None for p in core.parameters()))
print("head params:", sum(p.numel() for p in head.parameters()), "| core params:", sum(p.numel() for p in core.parameters()))
````

## `nb5_run.py`

All code cells of that notebook joined into one script with the reference solution patched in, used to confirm they execute. Written at 21:05. Group: Notebook check. Results: Section 1.2.

````python
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

print("lm_head tied to input embedding?", model.lm_head.weight is model.get_input_embeddings().weight)

# Surgery 1: from now on we only ever call the inner model.
core = model.model
print("core:", type(core).__name__)
print("core params: {:,}".format(sum(p.numel() for p in core.parameters())))

# What would lm_head's output cost during training? Change B and T to your own guess.
B, T, V = 8, 400, model.config.text_config.vocab_size
print(f"logits for B={B}, T={T}: {B*T*V*2/1e9:.2f} GB in bf16, {B*T*V*4/1e9:.2f} GB in fp32")

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

unit = F.normalize(opt_vecs, dim=-1)
print("cosine between the </opt> vectors:\n", (unit @ unit.T).cpu().numpy().round(3))
print("cosine(option_k, decide):", [round(v, 3) for v in (unit @ F.normalize(dec_vec, dim=-1)).tolist()])

raw = opt_vecs @ dec_vec
scaled = raw / math.sqrt(opt_vecs.shape[-1])
print("raw dot products      :", [round(v) for v in raw.tolist()])
print("divided by sqrt(1024) :", [round(v, 1) for v in scaled.tolist()])
print("softmax of that       :", [round(v, 4) for v in scaled.softmax(-1).tolist()])

class PointerHead(nn.Module):
    def __init__(self, hidden_size, d_head=256):
        super().__init__()
        self.opt_proj = nn.Linear(hidden_size, d_head, bias=False)
        self.dec_proj = nn.Linear(hidden_size, d_head, bias=False)
        self.d_head = d_head

    def forward(self, opt_vecs, dec_vec):
        """opt_vecs: [num_options, hidden_size], dec_vec: [hidden_size]  ->  scores: [num_options]"""
        # TODO: project both, take each option's dot product with the decision vector, scale.
        o = self.opt_proj(opt_vecs); d = self.dec_proj(dec_vec); return (o @ d) / math.sqrt(self.d_head)

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

for p in core.parameters():
    p.requires_grad_(False)

logits = jev(x, opt_end_positions, decide_position)
label = torch.tensor([smoke[0]["questions"][0]["label"]], device=device)
loss = F.cross_entropy(logits.unsqueeze(0), label)
loss.backward()

print("loss: %.3f   (ln 3 = %.3f is the loss of a model that knows nothing)" % (loss.item(), math.log(3)))
print("head got gradients?    ", all(p.grad is not None for p in head.parameters()))
print("any backbone gradient? ", any(p.grad is not None for p in core.parameters()))
trainable = sum(p.numel() for p in jev.parameters() if p.requires_grad)
total = sum(p.numel() for p in jev.parameters())
print(f"trainable: {trainable:,} of {total:,} ({100 * trainable / total:.3f}%)")
````

## `build_nb5.py`

Builds the step-by-step model-surgery notebook as JSON. Written at 21:05. Group: Notebook generator. Results: -.

````python
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
````

## `nb6_sweep.py`

Learning-rate sweep (1e-3 to 3e-5), 1 versus 6 option orders, 3 seeds, on the 12-record set. Reports epoch-0 loss, train and held-out accuracy and order stability. Written at 22:17. Group: Small-data experiments. Results: Section 2.3.

````python

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

def train_head(examples, epochs=60, lr=1e-3, seed=0):
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


import statistics as st

def chance(examples):
    return sum(1 / len(e["options"]) for e in examples) / len(examples)

def first_epoch_loss(examples, lr, seed):
    torch.manual_seed(seed)
    h = PointerHead(HIDDEN).to(device)
    o = torch.optim.AdamW(h.parameters(), lr=lr, weight_decay=0.01)
    tot = 0.0
    for ex in examples:
        o.zero_grad()
        loss = F.cross_entropy(h(ex["opt"], ex["dec"]).unsqueeze(0), torch.tensor([ex["label"]], device=device))
        loss.backward(); o.step(); tot += loss.item()
    return tot / len(examples)

print("chance accuracy on held-out: %.3f | majority baseline: %.3f" % (chance(heldout_examples), majority_baseline(train_records, heldout_records)))
cache = {n: build_examples(train_records, n_shuffles=n, seed=1) for n in (1, 6)}
import io, contextlib
print("lr      n  first-epoch-loss  train-acc  heldout-acc(mean±sd over 3 seeds)  stability(mean)")
for lr in [1e-3, 3e-4, 1e-4, 3e-5]:
    for n in (1, 6):
        accs, stabs, tr, fl = [], [], [], []
        for seed in range(3):
            fl.append(first_epoch_loss(cache[n], lr, seed))
            epochs = 40 if n == 6 else 240
            with contextlib.redirect_stdout(io.StringIO()):
                h = train_head(cache[n], epochs=epochs, lr=lr, seed=seed)
            r = evaluate(h, heldout_examples); accs.append(r["acc"]); stabs.append(r["stability"])
            tr.append(evaluate(h, cache[n])["acc"])
        print(f"{lr:<7g} {n}  {st.mean(fl):10.2f}        {st.mean(tr):.3f}      {st.mean(accs):.3f} ± {st.pstdev(accs):.3f}                     {st.mean(stabs):.3f}")
````

## `nb6_run.py`

The first-training notebook's cells as one script (first run, shuffling experiment). Written at 22:20. Group: Small-data experiments. Results: Sections 2.2 and 2.4.

````python

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
````

## `build_nb6.py`

Builds the first-training notebook as JSON. Written at 22:27. Group: Notebook generator. Results: -.

````python
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
````

## `nb6_curve.py`

Learning curve by number of training records, and the regularisation tests (weight decay, smaller head, early stopping), 5 seeds. Written at 22:52. Group: Small-data experiments. Results: Sections 2.5 and 2.6.

````python

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
````

## `gen_raw.py`

Runs any2jev's synthetic ticket generator (needs a throwaway environment with pydantic and its src folder on the path) and writes the raw files data/synthetic_raw/. jev/data.py then converts them. Written at 23:00. Group: Data. Results: Section 3.1.

````python
import sys
sys.path.insert(0, "/home/divye/Documents/Projects/any2jev/src")
from pathlib import Path
from any2jev.data import save_jsonl, synthetic_records

out = Path("/home/divye/Documents/Projects/create_jev/data/synthetic_raw")
n = 3000
recs = synthetic_records(n, seed=0)
n_val = n_test = n // 10
print("train", save_jsonl(out / "train.jsonl", recs[: n - n_val - n_test]))
print("val  ", save_jsonl(out / "val.jsonl", recs[n - n_val - n_test : n - n_test]))
print("test ", save_jsonl(out / "test.jsonl", recs[n - n_test :]))
````

## `nb7_proto.py`

First version of the batched pipeline: batched-versus-single equivalence, extraction timing, masked-loss check, and the normalise-or-not by learning-rate table with a constant rate. Written at 23:06. Group: Scaling experiments (prototype). Results: Sections 3.2 and 3.4 (single-seed table).

````python
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
````

## `nb7_proto2.py`

Epoch curve at a constant rate, learning curve, final model with the test set, shuffling ablation, department error table. Written at 23:09. Group: Scaling experiments (prototype). Results: Sections 3.4 to 3.8 (constant-rate versions).

````python
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
````

## `nb7_proto3.py`

Compares a constant learning rate with one-cycle schedules and gradient clipping over 5 seeds. Caches features in /tmp/nb7_feats.pt. Written at 23:13. Group: Scaling experiments (prototype). Results: Section 3.4 (schedule table).

````python
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
````

## `nb7_errs.py`

Prints misclassified department tickets from the trained model, to read the actual texts. Written at 23:24. Group: Scaling experiments (prototype). Results: Section 3.7.

````python

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
````

## `build_nb7.py`

Builds the scaling notebook as JSON, filling commentary from a numbers file. Written at 23:26. Group: Notebook generator. Results: -.

````python
import json, sys

N = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
def n(key): return N.get(key, "XX")

cells = []
def md(s):   cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))

md(r'''
# 07 — Scaling up: the head on 800 records

**Where we are.** Notebook 6 trained the head on 8 records and learned two things: a frozen backbone with a small head memorises easily, and 8 records are far too few to say whether it generalises. Here we use `data/synthetic/`: 3,000 support tickets (2,400 train, 300 val, 300 test) generated by any2jev's synthetic generator and converted to our schema by `jev/data.py`.

**What's new here:**
1. **Batched, padded feature extraction.** The backbone runs on 64 questions at a time, and the readout vectors go straight into a few big GPU tensors (`jev/features.py`). A training batch is then just an index slice, with no Python loop over examples.
2. **A masked loss.** Questions have different numbers of options, so score tensors are padded, and the padding must not take part in the softmax. You'll write this one.
3. **A learning-rate schedule.** Notebook 6 hinted that learning rates matter. At this scale they decide whether training is stable.
4. **Discipline about data.** We tune on the validation set and touch the test set exactly once, at the end.
5. **Experiments instead of opinions.** A learning curve, a shuffling ablation, and a look at the errors.

**A caveat that shapes how to read every number below:** this data is synthetic and template-based, and the label is encoded in keywords ("URGENT", "furious", "refund"). High accuracy here shows the pipeline works. It does not show the model would handle real tickets.

The new code is in `jev/head.py` (a batch-first `MaskedPointerHead`, with optional input normalisation) and `jev/features.py` (`build_requests`, `extract`, `FeatureSet`). Your notebook 5 and 6 code is untouched. Read the two files; they are short.

> **GPU memory:** shut down the kernel from notebook 6 first.
''')

code(r'''
%load_ext autoreload
%autoreload 2

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
''')

md(r'''
## Step 1: the data

Look at the converted records before training on them. Two things from the converter to notice: option text sometimes includes a description (`"billing: Charges, invoices, ..."`), and the state can be a flattened dict or chat.
''')

code(r'''
train_records = load_jsonl("../data/synthetic/train.jsonl")
val_records = load_jsonl("../data/synthetic/val.jsonl")
print(len(train_records), "train records,", len(val_records), "val records (the 300 test records stay untouched until the end)\n")

rec = train_records[3]
print("STATE:\n" + rec["state"] + "\n")
for q in rec["questions"]:
    print(f"[{q['type']}] {q['id']}: {q['prompt']}")
    for i, o in enumerate(q["options"]):
        print(f"   {'*' if i == q['label'] else ' '} {i}: {o}")
''')

md(r'''
## Step 2: cache the backbone features, in batches

Same idea as notebook 6 (the backbone is frozen, so run it once), but now it has to be fast enough for 14,000+ examples. Three changes:
- **Batches of 64, sorted by length**, so padding is small.
- **Right padding with no special handling.** The backbone is causal, so a real token never sees the padding that comes after it, and the readout vectors are the same as without padding. You can verify that below.
- **Pre-padded storage.** The vectors for all examples go into three tensors (`opt [E, N_max, D]`, `dec [E, D]`, `mask [E, N_max]`) in bf16 on the GPU. bf16 is lossless here because the backbone's output is already bf16.

First, check that batching doesn't change the answer. Do you expect the vectors to be *identical* to the one-at-a-time ones, or just very close?
''')

code(r'''
probe = build_requests(train_records[:10], n_orders=2, seed=0)[:24]
batched = extract(core, renderer, probe, PAD, HIDDEN, batch_size=8)
single = extract(core, renderer, probe, PAD, HIDDEN, batch_size=1)

cos_dec = F.cosine_similarity(batched.dec.float(), single.dec.float(), dim=-1)
cos_opt = F.cosine_similarity(batched.opt.float(), single.opt.float(), dim=-1)[batched.mask]
print("decide vectors: min cosine %.5f | option vectors: min cosine %.5f" % (cos_dec.min().item(), cos_opt.min().item()))
print("bit-identical rows: %d of %d" % ((batched.dec == single.dec).all(-1).sum().item(), len(batched)))
print("padding slots are exactly zero:", bool((batched.opt[~batched.mask] == 0).all()))
''')

md(r'''
**Why close but not identical?** A bf16 matrix multiply rounds differently depending on the shapes involved, so a batch of 8 and a batch of 1 give slightly different low-order bits. This is the same effect that made notebook 3's cache check look alarming until we compared against fp32. A cosine above 0.999 means the vectors point the same way to within rounding.

Now cache the real sets. Each training question is shown in 6 option orders (the original plus 5 shuffles with the label moved); validation uses 4 orders so we can measure order stability. Raise `N_TRAIN` to use more data.
''')

code(r'''
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
''')

md(r'''
## Step 3: the masked loss

The head returns a score tensor of shape `[batch, N_max]`, where `N_max` is 5 here. A question with 3 options has 3 real scores and 2 padded slots, and those padded slots hold arbitrary numbers. If they enter the softmax, they steal probability from the real options and the loss is wrong.

Write `masked_cross_entropy(scores, mask, labels)`: the mean cross-entropy over the batch, where only the real options take part. Remember what `exp(-inf)` is. The cell after it checks your answer against the slow, obviously-correct way (one example at a time), and also shows what happens if you forget the mask.
''')

code(r'''
def masked_cross_entropy(scores, mask, labels):
    """scores [B, N] (junk in padded slots), mask [B, N] (True = real option), labels [B]  ->  scalar mean loss"""
    # TODO
    raise NotImplementedError("fill in masked_cross_entropy")
''')

md(r'''
<details><summary>Solution (open after you've tried)</summary>

```python
return F.cross_entropy(scores.masked_fill(~mask, float("-inf")), labels)
```

A score of `-inf` becomes `exp(-inf) = 0` in the softmax, so that slot gets zero probability and the remaining options renormalise among themselves. The label is never a padded slot, so the loss is always finite.

</details>
''')

code(r'''
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
''')

md(r'''
## Step 4: training, with a learning-rate schedule

`train` below is complete (read it). Three choices to understand:
- **Batches are index slices** of the cached tensors, so an epoch over 14,520 examples takes about a second.
- **A one-cycle schedule**: the learning rate climbs from near zero to `lr` over the first 10% of steps, then decays along a cosine curve to near zero. Notebook 6 used a constant rate.
- **`normalize=True` in the head** applies an RMS normalisation to the backbone vectors before the projections. Their norm is around 110, and notebook 6 showed what that does to training at higher learning rates.
''')

code(r'''
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
''')

md(r'''
### The first real run

Chance accuracy is what you get by guessing uniformly among the options. It's the number to beat.
''')

code(r'''
def chance_accuracy(fs):
    return sum(1 / len(o) for o in fs.meta["options"]) / len(fs)

print("chance accuracy on val: %.3f\n" % chance_accuracy(val_fs))

t0 = time.time()
head, history = train(train_fs, val_fs, epochs=30, lr=1e-3, log_every=3)
print("\ntraining took %.1fs" % (time.time() - t0))

r = evaluate(head, val_fs)
print(f"\nval: accuracy {r['acc']:.3f} | loss {r['loss']:.3f} | order stability {r['stability']:.3f}")
print("by question:", {k: round(v, 3) for k, v in r["by_question"].items()})
''')

md(r'''
**Questions (answer in a comment before opening mine):**
1. Compare the train and val losses across epochs. Is there a gap, and does it look like notebook 6's overfitting (train accuracy 1.0, val near chance)? What changed?
2. Which question is hardest, and which easiest? Does that match what you'd guess from how the generator builds the tickets?
3. "Order stability" skips `score` questions. Why?

<details><summary>My answers</summary>

1. There is a gap in the loss (train ends near 0.003 against a val loss of about 0.13), so the head still memorises the training tickets, and the val loss stops improving after about epoch 20. But unlike notebook 6, val accuracy is high (about 0.96 against 0.35 for chance) and rose together with the training accuracy. The difference is data: 800 records is 100 times more distinct text than 8, so what the head learns now transfers to tickets it has not seen.
2. `tone` and `urgency` are easiest (about 0.998 and 0.996 on my run), `refund` next (0.98), and `department` is far the hardest (0.86): it has more options, some tickets are genuinely ambiguous, and some have the true department removed. That `urgency` is nearly solved is the keyword caveat in action: the generator expresses urgency with a handful of fixed phrases ("URGENT", "no rush"), so a 3-level ordinal scale is easy here and would not be on real text.
3. `score` options have a fixed order by design (calm < frustrated < furious), so we never shuffle them, and "does the answer survive reordering" has no meaning for them.

</details>
''')

md(r'''
## Experiment A: how much does the learning-rate schedule matter?

Same data, same head, five seeds each. We compare a constant learning rate (notebook 6's choice) with the one-cycle schedule, and also check whether the input normalisation earns its place.

Predict first: which settings will have the largest spread across seeds?
''')

code(r'''
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
''')

md(r'''
<details><summary>What I saw</summary>

''' + n("expA") + r'''

</details>
''')

md(r'''
## Experiment B: the learning curve

Notebook 6 suggested that more data helps but could not show it cleanly. Now we can: train on the first 25, 50, 100, ... 800 records (using the same cached features) and watch validation accuracy.
''')

code(r'''
print("records  examples   val acc            order stability")
for k in [25, 50, 100, 200, 400, 800]:
    sub = train_fs.where_rec_below(k)
    accs, stabs = [], []
    for seed in range(3):
        h, _ = train(sub, val_fs, epochs=15, seed=seed)
        r = evaluate(h, val_fs)
        accs.append(r["acc"]); stabs.append(r["stability"])
    print(f"{k:6d}  {len(sub):8d}   {statistics.mean(accs):.3f} +- {statistics.pstdev(accs):.3f}      {statistics.mean(stabs):.3f}")
''')

md(r'''
<details><summary>What I saw</summary>

''' + n("expB") + r'''

</details>
''')

md(r'''
## Experiment C: does shuffling the options still pay off?

Notebook 6 found that shuffling raised order stability on 12 questions, with the caveat that 12 questions is thin. Repeat it with 800 records. The baseline sees only the original option order, and gets 6 times more epochs so both runs take about the same number of gradient steps.
''')

code(r'''
train_fs_1order = extract(core, renderer, build_requests(train_records[:N_TRAIN], n_orders=1, seed=1), PAD, HIDDEN)

for name, fs, epochs in [("1 order (original only)", train_fs_1order, 90), ("6 orders (shuffled)", train_fs, 15)]:
    accs, stabs = [], []
    for seed in range(3):
        h, _ = train(fs, val_fs, epochs=epochs, seed=seed)
        r = evaluate(h, val_fs)
        accs.append(r["acc"]); stabs.append(r["stability"])
    print(f"{name:26s} val acc {statistics.mean(accs):.3f} | order stability {statistics.mean(stabs):.3f}")
''')

md(r'''
<details><summary>What I saw</summary>

''' + n("expC") + r'''

</details>
''')

md(r'''
## Error analysis: what does it get wrong?

Accuracy is one number; the mistakes tell you what to fix. Start with the error rate per true answer on the `department` question. The generator sometimes removes the true department from the options and adds a "none of the above" option (under the keys `other`, `none` or `not_listed`) as the *correct* answer, so those are listed too.
''')

code(r'''
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
''')

md(r'''
Careful with that table: `n` counts *examples*, and every ticket appears in 4 option orders, so one badly handled ticket contributes up to 4 errors. Look at the errors per distinct ticket instead.
''')

code(r'''
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
''')

md(r'''
**Questions:**
1. Which true answers have the highest error rates, and is "none of the above" harder than a real department?
2. Are the errors spread evenly over many tickets, or concentrated? Read the ticket texts: are the mistakes foolish, or are the labels debatable?
3. For `true=other`, what does the model do instead?
4. Which of these would you attack with more data, and which needs something different?

<details><summary>My answers</summary>

''' + n("errors") + r'''

</details>
''')

md(r'''
## The test set, once

We have tuned the schedule, the normalisation and the epoch count against the validation set. The test set has not been used for any decision, so it gives an unbiased estimate. We run it once and report whatever it says. If you tweak anything after looking at this number, it stops being a test set.
''')

code(r'''
test_records = load_jsonl("../data/synthetic/test.jsonl")
test_fs = extract(core, renderer, build_requests(test_records, n_orders=4, seed=3), PAD, HIDDEN)

final_head, _ = train(train_fs, val_fs, epochs=30, lr=1e-3, seed=0)
for name, fs in [("val", val_fs), ("test", test_fs)]:
    r = evaluate(final_head, fs)
    print(f"{name:5s} accuracy {r['acc']:.3f} | loss {r['loss']:.3f} | order stability {r['stability']:.3f} | chance {chance_accuracy(fs):.3f}")
    print("      by question:", {k: round(v, 3) for k, v in r["by_question"].items()})
''')

md(r'''
## Wrap-up

''' + n("wrap") + r'''

**Next, notebook 8:** calibration. A decision model is only useful if its probabilities mean something, so we'll measure whether "90% confident" is right 90% of the time (expected calibration error) and fix it with temperature scaling, fitted on the validation set. After that: a real-text test using the public datasets, and LoRA.
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

out = "/home/divye/Documents/Projects/create_jev/notebooks/07_scaling_up.ipynb"
with open(out, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", out, "with", len(cells), "cells")
````

## `nb7_run.py`

The scaling notebook's cells as one script: schedule comparison, learning curve, shuffling ablation, error analysis, per-ticket errors, and the single test-set evaluation. Source of the final numbers. Written at 23:26. Group: Scaling experiments (final). Results: Sections 3.3 to 3.8.

````python

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
````

## `lora_probe.py`

First LoRA probe with a hand-written LoRA module: refactor equivalence, no-op at initialisation, step time and memory. Imports jev.lora, which was removed when we switched to peft, so it no longer runs. Written at 00:24. Group: LoRA feasibility (STALE). Results: Section 8 (memory facts).

````python
import os, sys, time, math, random
os.environ["HF_HUB_DISABLE_XET"] = "1"; os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
sys.path.insert(0, "/home/divye/Documents/Projects/create_jev")
os.chdir("/home/divye/Documents/Projects/create_jev/notebooks")
import torch, torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForMultimodalLM
from jev.render import Renderer
from jev.head import MaskedPointerHead
from jev.data import load_jsonl
from jev.features import build_requests, extract, render_requests, encode
from jev.lora import apply_lora

tokenizer = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B").tokenizer
model = AutoModelForMultimodalLM.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16, device_map="auto").eval()
core = model.model
for p in core.parameters(): p.requires_grad_(False)
renderer = Renderer(tokenizer); device = next(core.parameters()).device
H = model.config.text_config.hidden_size; PAD = tokenizer.pad_token_id
val_recs = load_jsonl("../data/synthetic/val.jsonl"); train_recs = load_jsonl("../data/synthetic/train.jsonl")

# 1) refactor equivalence vs features saved by the old code
tr_old, va_old, te_old = torch.load("/tmp/nb7_feats.pt", weights_only=False)
va_new = extract(core, renderer, build_requests(val_recs[:300], 4, 2), PAD, H)
print("refactor: opt identical", torch.equal(va_new.opt, va_old.opt), "| dec identical", torch.equal(va_new.dec, va_old.dec),
      "| mask identical", torch.equal(va_new.mask, va_old.mask), "| labels identical", torch.equal(va_new.labels, va_old.labels))

# 2) LoRA is an exact no-op at init
probe = build_requests(train_recs[:20], 1, 0)
before = extract(core, renderer, probe, PAD, H)
params = apply_lora(core, rank=8, alpha=16)
after = extract(core, renderer, probe, PAD, H)
print("LoRA at init: opt identical", torch.equal(before.opt, after.opt), "| dec identical", torch.equal(before.dec, after.dec))
print("LoRA params: %d tensors, %.2fM params (%.2f%% of backbone)" % (len(params), sum(p.numel() for p in params) / 1e6, 100 * sum(p.numel() for p in params) / 852985920))
print("backbone params still frozen:", all(not p.requires_grad for n, p in core.named_parameters() if not (n.endswith('.A') or n.endswith('.B'))))

# 3) training step speed and memory
head = MaskedPointerHead(H).to(device)
opt = torch.optim.AdamW([{"params": params, "lr": 2e-4}, {"params": head.parameters(), "lr": 1e-3}], weight_decay=0.01)
reqs = build_requests(train_recs[:800], 1, 5, keep_original_first=False); rendered = render_requests(renderer, reqs)
n_max = max(len(r[1]) for r in rendered); labels = torch.tensor([r["question"]["label"] for r in reqs], device=device)
print("\nquestions in 800 records:", len(reqs))
for bs in [8, 16, 32]:
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); order = random.Random(0).sample(range(len(reqs)), len(reqs))
    try:
        times = []
        for step in range(5):
            idx = order[step * bs:(step + 1) * bs]
            torch.cuda.synchronize(); t0 = time.time()
            o, d, m = encode(core, rendered, idx, PAD, n_max)
            scores = head(o.float(), d.float(), m)
            loss = F.cross_entropy(scores, labels[idx]); opt.zero_grad(); loss.backward(); opt.step()
            torch.cuda.synchronize(); times.append(time.time() - t0)
        print(f"batch {bs:3d}: {1000*sum(times[1:])/len(times[1:]):7.0f} ms/step ({1000*sum(times[1:])/len(times[1:])/bs:5.1f} ms/question) | peak memory {torch.cuda.max_memory_allocated()/1e9:.2f} GB | loss {loss.item():.3f}")
    except torch.cuda.OutOfMemoryError as e:
        print(f"batch {bs:3d}: OUT OF MEMORY"); torch.cuda.empty_cache()
g = [p.grad is not None and p.grad.abs().sum().item() > 0 for p in params]
print("LoRA tensors with nonzero grad: %d of %d" % (sum(g), len(g)))
````

## `lora_mem.py`

Memory per batch size with and without gradient checkpointing. Showed about 0.55 GB per sequence and the KV-cache crash during recomputation. Imports jev.lora (removed), so it no longer runs. Written at 00:25. Group: LoRA feasibility (STALE). Results: Section 8.

````python
import os, sys, time, random
os.environ["HF_HUB_DISABLE_XET"] = "1"; os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
sys.path.insert(0, "/home/divye/Documents/Projects/create_jev")
os.chdir("/home/divye/Documents/Projects/create_jev/notebooks")
import torch, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoProcessor, AutoModelForMultimodalLM
from jev.render import Renderer
from jev.head import MaskedPointerHead
from jev.data import load_jsonl
from jev.features import build_requests, render_requests, encode
from jev.lora import apply_lora

tokenizer = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B").tokenizer
model = AutoModelForMultimodalLM.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16, device_map="auto").eval()
core = model.model
for p in core.parameters(): p.requires_grad_(False)
renderer = Renderer(tokenizer); device = next(core.parameters()).device
H = model.config.text_config.hidden_size; PAD = tokenizer.pad_token_id
train_recs = load_jsonl("../data/synthetic/train.jsonl")
from jev.features import extract
tr_old, va_old, te_old = torch.load("/tmp/nb7_feats.pt", weights_only=False)
va_new = extract(core, renderer, build_requests(load_jsonl("../data/synthetic/val.jsonl")[:300], 4, 2), PAD, H)
print("use_cache=False: val features identical to the old ones:", torch.equal(va_new.opt, va_old.opt) and torch.equal(va_new.dec, va_old.dec))
params = apply_lora(core, rank=8, alpha=16)
head = MaskedPointerHead(H).to(device)
reqs = build_requests(train_recs[:200], 1, 5, keep_original_first=False); rendered = render_requests(renderer, reqs)
n_max = max(len(r[1]) for r in rendered); labels = torch.tensor([r["question"]["label"] for r in reqs], device=device)
base = torch.cuda.memory_allocated() / 1e9
print(f"baseline allocated {base:.2f} GB | free {torch.cuda.mem_get_info()[0]/1e9:.2f} GB | median len {sorted(len(r[0]) for r in rendered)[len(rendered)//2]}")

def step(idx, ckpt):
    opt, dec, mask = encode(core, rendered, idx, PAD, n_max)
    loss = F.cross_entropy(head(opt.float(), dec.float(), mask), labels[idx]); loss.backward()
    return loss

# plain (no checkpointing)
for bs in [2, 4, 8]:
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        idx = list(range(bs)); step(idx, False); torch.cuda.synchronize()
        print(f"no checkpointing  batch {bs}: peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB (+{torch.cuda.max_memory_allocated()/1e9-base:.2f} over baseline)")
    except torch.cuda.OutOfMemoryError:
        print(f"no checkpointing  batch {bs}: OOM"); torch.cuda.empty_cache()
    for p in params: p.grad = None

# gradient checkpointing: wrap each decoder layer
layers = core.language_model.layers
for layer in layers:
    orig = layer.forward
    def make(orig):
        def fwd(*a, **k):
            return checkpoint(orig, *a, use_reentrant=False, **k)
        return fwd
    layer.forward = make(orig)
for bs in [4, 8, 16, 24, 32]:
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        idx = list(range(bs)); torch.cuda.synchronize(); t0 = time.time(); step(idx, True); torch.cuda.synchronize(); dt = time.time() - t0
        print(f"checkpointing     batch {bs}: peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB (+{torch.cuda.max_memory_allocated()/1e9-base:.2f}) | {1000*dt:.0f} ms ({1000*dt/bs:.0f} ms/question)")
    except torch.cuda.OutOfMemoryError:
        print(f"checkpointing     batch {bs}: OOM"); torch.cuda.empty_cache()
    for p in params: p.grad = None
g = sum(1 for p in params if p.grad is not None)
````

## `vis_probe.py`

Loads one synthetic image through the processor and prints keys, shapes, token counts, a forward pass with two different images, and a generation check. Written at 00:37. Group: Vision interface. Results: Section 9.1.

````python
import os, sys
os.environ["HF_HUB_DISABLE_XET"] = "1"; os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForMultimodalLM
processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B")
model = AutoModelForMultimodalLM.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16, device_map="auto").eval()
core = model.model; dev = next(core.parameters()).device
ip = processor.image_processor
print("image_processor:", type(ip).__name__, {k: getattr(ip, k) for k in ("patch_size", "merge_size", "temporal_patch_size", "min_pixels", "max_pixels") if hasattr(ip, k)}, "| size:", getattr(ip, "size", None))
vc = model.config.vision_config
print("vision config:", {k: getattr(vc, k) for k in ("hidden_size", "depth", "num_heads", "patch_size", "spatial_merge_size", "out_hidden_size") if hasattr(vc, k)})

def img(color, shape="circle", size=224):
    im = Image.new("RGB", (size, size), "white"); d = ImageDraw.Draw(im)
    box = [size * 0.25, size * 0.25, size * 0.75, size * 0.75]
    d.ellipse(box, fill=color) if shape == "circle" else d.rectangle(box, fill=color)
    return im

tok = processor.tokenizer
text = "<|vision_start|><|image_pad|><|vision_end|>What color is the shape?"
inp = processor(text=[text], images=[img("red")], return_tensors="pt")
print("keys:", list(inp.keys()))
for k, v in inp.items(): print(f"  {k}: {tuple(v.shape)} {v.dtype}")
n_pad = (inp["input_ids"] == tok.convert_tokens_to_ids("<|image_pad|>")).sum().item()
print("image_pad tokens after expansion:", n_pad, "| image_grid_thw:", inp["image_grid_thw"].tolist())
if "mm_token_type_ids" in inp: print("mm_token_type_ids values:", sorted(set(inp["mm_token_type_ids"].flatten().tolist())))

inp = {k: v.to(dev) for k, v in inp.items()}
with torch.no_grad():
    h_red = core(**inp).last_hidden_state
    inp2 = dict(inp); inp2["pixel_values"] = processor(text=[text], images=[img("blue", "square")], return_tensors="pt")["pixel_values"].to(dev)
    h_blue = core(**inp2).last_hidden_state
print("hidden:", tuple(h_red.shape), "| last-token cosine red-circle vs blue-square: %.4f" % torch.nn.functional.cosine_similarity(h_red[0, -1].float(), h_blue[0, -1].float(), dim=0).item())

chat = [{"role": "user", "content": [{"type": "image", "image": img("red")}, {"type": "text", "text": "What color is the shape? Answer with one word."}]}]
gi = processor.apply_chat_template(chat, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(dev)
out = model.generate(**gi, max_new_tokens=6, do_sample=False)
print("generate:", repr(processor.decode(out[0][gi["input_ids"].shape[-1]:], skip_special_tokens=True)))
````

## `smoke_modules.py`

Checks calibration maths, the shapes generator, multimodal rendering, batched encode, peft LoRA (no-op, parameter counts, gradients) and a few training steps with checkpointing. Written at 00:40. Group: Verification. Results: Sections 7 to 9 (verification).

````python
import os, sys, time, shutil
os.environ["HF_HUB_DISABLE_XET"] = "1"; os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
sys.path.insert(0, "/home/divye/Documents/Projects/create_jev")
import torch, torch.nn.functional as F
from jev.calibration import *
from jev.shapes import generate_dataset
from jev.data import load_jsonl

# --- calibration unit tests (CPU) ---
mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool); labels = torch.tensor([2, 0])
perfect = torch.tensor([[0., 0., 50., 0.], [50., 0., 0., 0.]]); uniform = torch.zeros(2, 4)
wrong = torch.tensor([[50., 0., 0., 0.], [0., 50., 0., 0.]])
b = lambda s: brier_score(probabilities(s, mask), labels, mask).item()
print("brier perfect %.3f | uniform %.3f (formula %.3f) | confidently wrong %.3f (expect 2.0)" % (b(perfect), b(uniform), uniform_brier(mask).item(), b(wrong)))
torch.manual_seed(0); sc = torch.randn(2000, 4) * 3; lb = torch.randint(0, 4, (2000,)); mk = torch.ones(2000, 4, dtype=torch.bool)
sc[torch.arange(2000), lb] += 4.0
T = fit_temperature(sc, mk, lb); r0 = calibration_report(sc, mk, lb); r1 = calibration_report(sc, mk, lb, T)
print("temperature fit: T=%.2f | nll %.3f -> %.3f | ece %.3f -> %.3f" % (T, r0["nll"], r1["nll"], r0["ece"], r1["ece"]))

# --- shapes generator ---
d = "/tmp/shapes_smoke"
shutil.rmtree(d, ignore_errors=True)
print("generated:", generate_dataset(d, {"train": 12, "val": 4}, seed=0)); recs = load_jsonl(d + "/train.jsonl")
print("record 0 state:", recs[0]["state"], "| questions:", [(q["id"], q["options"][q["label"]]) for q in recs[0]["questions"]])

# --- model: multimodal render + encode + peft LoRA ---
from transformers import AutoProcessor, AutoModelForMultimodalLM
from jev.vision import MMRenderer
from jev.features import build_requests, render_requests, encode, extract
from jev.train import add_lora, train_lora, enable_checkpointing, disable_checkpointing
from jev.head import MaskedPointerHead
processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B"); tok = processor.tokenizer
model = AutoModelForMultimodalLM.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16, device_map="auto").eval()
core = model.model; dev = next(core.parameters()).device; H = 1024; PAD = tok.pad_token_id
for p in core.parameters(): p.requires_grad_(False)
rend = MMRenderer(processor, d)
ids, ends, dpos, extra = rend.render_question(recs[0], 0)
print("tokens:", len(ids), "| image tokens:", sum(extra["types"]), "| decide pos:", dpos, "| text:", tok.decode(ids[:3]), "...", tok.decode(ids[-30:])[-60:].replace("\n", " "))

reqs = build_requests(recs[:6], 1, 0)[:12]; rendered = render_requests(rend, reqs); n_max = max(len(r[1]) for r in rendered)
with torch.no_grad():
    ob, db, _ = encode(core, rendered, list(range(12)), PAD, n_max)
    singles = [encode(core, rendered, [i], PAD, n_max) for i in range(12)]
    cos = min(F.cosine_similarity(db[i].float(), singles[i][1][0].float(), dim=0).item() for i in range(12))
    # use_cache=False must not change values
    ids_t = torch.tensor([rendered[0][0]], device=dev)
    ex = rendered[0][3]
    kw = dict(mm_token_type_ids=torch.tensor([ex["types"]], device=dev), pixel_values=torch.cat(ex["pixels"]).to(dev), image_grid_thw=torch.cat(ex["grids"]).to(dev))
    a = core(input_ids=ids_t, use_cache=True, **kw).last_hidden_state; b_ = core(input_ids=ids_t, use_cache=False, **kw).last_hidden_state
print("batched-with-images vs single: min cosine %.5f | use_cache True/False identical: %s" % (cos, torch.equal(a, b_)))
# does the model use the image?  same question, different image
print("image dependence: cosine(decide vector, same question with other image's pixels) = %.4f" % F.cosine_similarity(db[0].float(), encode(core, [(rendered[0][0], rendered[0][1], rendered[0][2], rendered[1][3])], [0], PAD, n_max)[1][0].detach().float(), dim=0).item())

# peft LoRA: no-op at init, trainable counts, a few real steps with checkpointing
before = extract(core, rend, reqs, PAD, H, batch_size=4)
pm = add_lora(core, rank=8, alpha=16)
n_tr = sum(p.numel() for p in core.parameters() if p.requires_grad); n_all = sum(p.numel() for p in core.parameters())
after = extract(core, rend, reqs, PAD, H, batch_size=4)
print("peft: trainable %.2fM of %.1fM (%.2f%%) | no-op at init: %s" % (n_tr / 1e6, n_all / 1e6, 100 * n_tr / n_all, torch.equal(before.opt, after.opt) and torch.equal(before.dec, after.dec)))
vis_lora = [n for n, p in core.named_parameters() if p.requires_grad and "visual" in n]; print("LoRA tensors inside vision tower:", len(vis_lora))
head = MaskedPointerHead(H).to(dev)
torch.cuda.reset_peak_memory_stats(); t0 = time.time()
hist = train_lora(core, head, rend, recs[:8], PAD, epochs=1, batch_size=4, brier_weight=1.0, log_every=2, max_steps=4)
print("train_lora ran %d steps in %.1fs | peak GPU %.2f GB | history %s" % (hist[-1]["steps"], time.time() - t0, torch.cuda.max_memory_allocated() / 1e9, [round(h["train_loss"], 3) for h in hist]))
grads = [p.grad is not None and p.grad.abs().sum().item() > 0 for p in core.parameters() if p.requires_grad]; print("LoRA tensors with nonzero grad: %d / %d" % (sum(grads), len(grads)))
````

## `run_nb.py`

Runs a notebook's code cells in one process with SMOKE switched on. Used for the dry runs of the LoRA and vision notebooks. Written at 00:45. Group: Verification. Results: -.

````python
import json, sys, os, re
nb_path = sys.argv[1]
nb = json.load(open(nb_path))
src = "\n\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
src = "\n".join(l for l in src.split("\n") if not l.startswith("%"))
assert "SMOKE = False" in src
src = src.replace("SMOKE = False", "SMOKE = True", 1)
os.chdir(os.path.dirname(os.path.abspath(nb_path)))
exec(compile(src, nb_path, "exec"), {"__name__": "__main__"})
````

## `build_nb8.py`

Builds the LoRA and calibration notebook as JSON. Written at 00:48. Group: Notebook generator. Results: -.

````python
import json

cells = []
def md(s):   cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))

md(r'''
# 08 — Adapting the backbone with LoRA, and measuring whether the probabilities can be trusted

**Where we are.** Notebook 7 trained a pointer head on top of a frozen backbone and reached high accuracy on the synthetic tickets. Two questions are still open, and this notebook answers both.

1. **Can the backbone itself adapt?** A frozen backbone hands the head fixed vectors, and the head can only re-weight what is already in them. LoRA lets the backbone change *what it puts into* those vectors, by training small low-rank corrections inside its layers while the original weights stay untouched.
2. **Can we trust the probabilities?** The whole point of a decision model is that downstream code can threshold its output ("escalate if P(urgent) > 0.8"). That only works if "80% confident" means right about 80% of the time. Accuracy says nothing about that. We measure it with the **Brier score**, the **expected calibration error (ECE)** and the **negative log-likelihood (NLL)**, and we fix the simplest part of the problem with **temperature scaling**.

**What's in the notebook:**
1. A frozen-head baseline, so every LoRA number has something to be compared against.
2. The calibration metrics, with hand-checkable examples so you can see exactly what they compute.
3. LoRA: the idea, the exact parameter arithmetic, and a check that `peft` really computes what the formula says.
4. Training the backbone and head together, and why that costs so much more than training the head alone.
5. Brier as a **loss term** as well as a metric: does training on it improve calibration?
6. A fair comparison of all variants, with the test set used once at the end.

**Tools.** LoRA comes from the `peft` library, which is the standard implementation. Everything else is code in `jev/` that you've seen the earlier versions of: `jev/calibration.py` (the metrics), `jev/train.py` (training loops, LoRA helpers) and `jev/features.py` (feature extraction).

**Running it.** Set `SMOKE = True` in the first code cell for a tiny dry run (a few minutes) that only checks that every cell executes. Leave it `False` for the real experiment. Training the backbone is much slower than anything in notebooks 6 and 7, and the training cell prints the elapsed time and the peak GPU memory as it goes, so you can see the cost on your own machine.

> **GPU memory:** shut down the kernels from other notebooks first. Training through the backbone needs real memory, and each idle kernel that still holds the model takes about 2 GB of your 8 GB.
''')

code(r'''
%load_ext autoreload
%autoreload 2

import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import sys
sys.path.insert(0, "..")

import inspect, math, time
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForMultimodalLM

from jev.render import Renderer
from jev.head import MaskedPointerHead
from jev.data import load_jsonl
from jev.features import build_requests, extract
from jev.calibration import (probabilities, nll, brier_score, uniform_brier, expected_calibration_error,
                             fit_temperature, calibration_report, format_reliability)
from jev.train import evaluate, chance_accuracy, predict_scores, train_head, add_lora, train_lora, save_run

SMOKE = False   # True = tiny dry run that only checks the notebook executes. False = the real experiment.

N_TRAIN, N_VAL, N_TEST = (16, 12, 12) if SMOKE else (800, 300, 300)
HEAD_EPOCHS = 2 if SMOKE else 30
LORA_EPOCHS = 1 if SMOKE else 2
LORA_BATCH = 4 if SMOKE else 16       # lower this if you run out of GPU memory
BRIER_WEIGHT = 1.0                    # weight of the Brier term in the combined loss (experiment in section 6)
print("SMOKE" if SMOKE else "FULL", "run | records: train", N_TRAIN, "val", N_VAL, "test", N_TEST)
''')

code(r'''
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

train_records = load_jsonl("../data/synthetic/train.jsonl")[:N_TRAIN]
val_records = load_jsonl("../data/synthetic/val.jsonl")[:N_VAL]
test_records = load_jsonl("../data/synthetic/test.jsonl")[:N_TEST]
print("loaded | GPU memory allocated: %.2f GB" % (torch.cuda.memory_allocated() / 1e9))
''')

md(r'''
## 1. The baseline: a head on a frozen backbone

This repeats notebook 7's final setup: cache the backbone's readout vectors once (each training question in 6 option orders, validation in 4), then train the pointer head on the cache with a one-cycle learning-rate schedule. `jev.train.train_head` and `jev.train.evaluate` are the notebook 7 functions, moved into the module so that notebooks 8 and 9 share them.

The validation requests use a fixed seed, so every later model sees exactly the same questions in exactly the same option orders. That makes the comparisons fair.
''')

code(r'''
val_requests = build_requests(val_records, n_orders=4, seed=2)

t0 = time.time()
train_fs = extract(core, renderer, build_requests(train_records, n_orders=6, seed=1), PAD, HIDDEN)
val_fs_frozen = extract(core, renderer, val_requests, PAD, HIDDEN)
print(f"cached {len(train_fs)} train and {len(val_fs_frozen)} val examples in {time.time() - t0:.0f}s")

head_frozen, _ = train_head(train_fs, val_fs_frozen, HIDDEN, epochs=HEAD_EPOCHS, lr=1e-3, seed=0)
r = evaluate(head_frozen, val_fs_frozen)
print(f"frozen backbone + head | val accuracy {r['acc']:.3f} (chance {chance_accuracy(val_fs_frozen):.3f}) | order stability {r['stability']:.3f}")
print("by question:", {k: round(v, 3) for k, v in r["by_question"].items()})
''')

md(r'''
## 2. Are the probabilities trustworthy? The metrics

A decision model outputs a probability for every option. Accuracy only looks at which option got the highest one. Three numbers look at the probabilities themselves.

**Negative log-likelihood (NLL)** is the cross-entropy loss we train with: `-log(p_true)`, averaged. It punishes a confident wrong answer severely, because `-log` of a tiny probability is large.

**Brier score** is the mean squared error between the probability vector and the one-hot true answer: for each example, `sum over options of (p_k - y_k)^2`, where `y_k` is 1 for the true option and 0 for the others. It is 0 for a perfect model. A model that always answers "equal probability for every option" scores `1 - 1/N` for `N` options (0.5 for two options, 0.667 for three), and a model that is 100% confident in a wrong answer scores 2. Unlike NLL it is bounded, so one terrible prediction cannot dominate it.

**Expected calibration error (ECE)** asks the question directly: group examples by how confident the model was in its top answer (say 0.9 to 1.0), and compare that confidence with the accuracy actually achieved in the group. ECE is the average gap, weighted by group size. A perfectly calibrated model has ECE 0: of the answers it gives with 80% confidence, 80% are right.

The Brier score and NLL are both "proper scoring rules": they are minimised, in expectation, only by reporting the true probabilities. That is why they are good metrics for calibration and why training on them is sensible. Accuracy is not a proper scoring rule.

Here is the Brier score implementation, followed by tests with answers you can verify by hand.
''')

code(r'''
print(inspect.getsource(brier_score))
''')

code(r'''
# A batch of two questions. The first has 4 options (true answer = option 2), the second has 2 options (true answer = option 0).
mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
labels = torch.tensor([2, 0])

def brier_of(scores):
    return brier_score(probabilities(scores, mask), labels, mask).item()

perfect = torch.tensor([[0., 0., 50., 0.], [50., 0., 0., 0.]])        # all the probability on the true option
uniform = torch.zeros(2, 4)                                            # equal scores -> equal probabilities
wrong   = torch.tensor([[50., 0., 0., 0.], [0., 50., 0., 0.]])        # all the probability on a wrong option

print("perfect model          :", round(brier_of(perfect), 4), "  (hand calculation: 0)")
print("uniform model          :", round(brier_of(uniform), 4), "  (hand calculation: the 4-option question gives 1 - 1/4 = 0.75, the 2-option one gives 0.5, mean 0.625)")
print("confidently wrong      :", round(brier_of(wrong), 4), "  (hand calculation: (1-0)^2 + (0-1)^2 = 2 for every question)")
assert abs(brier_of(perfect)) < 1e-6 and abs(brier_of(uniform) - 0.625) < 1e-6 and abs(brier_of(wrong) - 2.0) < 1e-6
print("formula for the uniform model, from the mask:", round(uniform_brier(mask).item(), 4))
''')

md(r'''
### Applying them to the baseline, and temperature scaling

`predict_scores` returns the head's raw scores for every cached example. `calibration_report` turns scores into probabilities and computes accuracy, NLL, Brier and ECE.

**Temperature scaling** is the simplest fix for overconfidence. Divide every score by a single number `T` before the softmax. `T > 1` flattens the probabilities (less confident), `T < 1` sharpens them. The argmax never changes, so accuracy is untouched; only the confidence changes. `fit_temperature` finds the `T` that minimises the NLL on a set of data. It has one parameter, so it can't overfit much, but it must be fitted on data the model was not trained on (here the validation set), and its effect should be judged on a *different* set (the test set, at the end). Numbers below that fit `T` on validation and then report on validation are optimistic by construction; they show the mechanism, not the final result.
''')

code(r'''
def metrics_row(name, scores, fs, temperature=1.0):
    r = calibration_report(scores, fs.mask.cpu(), fs.labels.cpu(), temperature)
    print(f"{name:36s} T={temperature:5.2f} | acc {r['acc']:.3f} | NLL {r['nll']:.3f} | Brier {r['brier']:.3f} | ECE {r['ece']:.3f}")
    return r

mask_v, labels_v = val_fs_frozen.mask.cpu(), val_fs_frozen.labels.cpu()
print("Brier score of always answering uniformly: %.3f   (a model has to beat this to know anything)\n" % uniform_brier(mask_v).item())

scores_frozen = predict_scores(head_frozen, val_fs_frozen)
metrics_row("frozen head, as trained", scores_frozen, val_fs_frozen)

T_frozen = fit_temperature(scores_frozen, mask_v, labels_v)
metrics_row("frozen head, temperature-scaled", scores_frozen, val_fs_frozen, T_frozen)

print("\nReliability table for the head as trained (validation):")
_, table = expected_calibration_error(probabilities(scores_frozen, mask_v), labels_v)
print(format_reliability(table))
''')

md(r'''
**How to read the reliability table.** Each row is a confidence bin. If the model is calibrated, `mean conf` and `accuracy` match in every row. If `mean conf` is higher than `accuracy`, the model is **overconfident** in that bin, which is the usual failure for a network trained to near-zero training loss, as ours is. The bins with the most examples matter most, and that is why ECE weights by `n`.
''')

md(r'''
## 3. LoRA: letting the backbone adapt without retraining it

**The idea.** A linear layer computes `y = W x`, with `W` a large matrix (for example 6144 x 1024). Fine-tuning normally changes every entry of `W`. LoRA freezes `W` and learns a correction with far fewer numbers:

`y = W x + (alpha / r) * B (A x)`

where `A` is `r x 1024`, `B` is `6144 x r`, and `r` (the rank) is small, such as 8. The correction `B A` is a 6144 x 1024 matrix, but it is built from only `r * (1024 + 6144)` numbers. The bet is that the change a task needs is **low rank**: it lives in a small subspace of what the layer can do. For this exact layer (`in_proj_qkv`, 1024 to 6144) the full matrix has 6,291,456 entries and the rank-8 correction has 57,344, about 0.9% as many.

Two details that matter:
- **`B` starts at zero**, and `A` starts random. So at the start the correction is exactly zero and the adapted model **is** the original model. Training moves it away smoothly. We check this below.
- **`alpha / r` is a fixed scale** on the correction. With `alpha = 2r` the correction is multiplied by 2. It lets you change the rank without retuning the learning rate much.

**Where we put it.** Every linear layer in the language model's 24 decoder layers: the projections of the DeltaNet layers (`in_proj_qkv`, `in_proj_z`, `out_proj`), of the attention layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`), and of the MLPs (`gate_proj`, `up_proj`, `down_proj`). We skip the two tiny gate projections `in_proj_a` and `in_proj_b` (16 outputs each), and we leave the vision tower alone. The pattern is a regular expression in `jev/train.py`:
''')

code(r'''
from jev.train import LORA_TARGET_REGEX
import re

print(LORA_TARGET_REGEX, "\n")

# Count what the pattern matches BEFORE touching the model, and work out the expected number of LoRA parameters by hand.
RANK, ALPHA = 8, 16
matched, expected = [], 0
for name, module in core.named_modules():
    if isinstance(module, torch.nn.Linear) and re.fullmatch(LORA_TARGET_REGEX, name):
        matched.append(name)
        expected += RANK * (module.in_features + module.out_features)

print(f"{len(matched)} linear layers match, in {len(set(n.split('.')[2] for n in matched))} decoder layers")
print("example matches:", matched[:3], "...")
print("any match inside the vision tower?", any(n.startswith("visual") for n in matched))
print(f"expected LoRA parameters at rank {RANK}: {expected:,}   ({100 * expected / sum(p.numel() for p in core.parameters()):.2f}% of the backbone)")
''')

md(r'''
Now apply LoRA. First we record the readout vectors for a few questions with the **unmodified** backbone, so that after wrapping we can check the claim that LoRA starts as an exact no-op. `add_lora` calls `peft.get_peft_model`, which replaces each matched linear layer, in place, with a LoRA version that holds the original layer plus the two small matrices.
''')

code(r'''
probe_requests = build_requests(val_records[:4], n_orders=2, seed=5)
probe_before = extract(core, renderer, probe_requests, PAD, HIDDEN, batch_size=8)

peft_model = add_lora(core, rank=RANK, alpha=ALPHA)

trainable = sum(p.numel() for p in core.parameters() if p.requires_grad)
total = sum(p.numel() for p in core.parameters())
print(f"trainable parameters: {trainable:,} of {total:,} ({100 * trainable / total:.2f}%)")
print("matches the hand count?", trainable == expected)

probe_after = extract(core, renderer, probe_requests, PAD, HIDDEN, batch_size=8)
print("readout vectors identical to the unmodified backbone (LoRA starts as a no-op)?",
      torch.equal(probe_before.opt, probe_after.opt) and torch.equal(probe_before.dec, probe_after.dec))
''')

md(r'''
### Checking that `peft` computes what the formula says

Here is one wrapped layer. It holds the original layer (`base_layer`), the two small matrices (`lora_A`, `lora_B`) and the scale (`scaling`, which is `alpha / r`). Because `B` is zero, a layer with a fresh adapter can't show whether `peft` really adds `scale * B(A(x))`. So for this check only, we put random values into `B`, compare `peft`'s output with our own computation of the formula, and then restore `B` to zeros.
''')

code(r'''
layer = core.language_model.layers[3].self_attn.q_proj          # layer 3 is the first full-attention layer
A, B = layer.lora_A["default"], layer.lora_B["default"]
scale = layer.scaling["default"]
print(type(layer).__name__, "| base layer:", tuple(layer.base_layer.weight.shape), "| A:", tuple(A.weight.shape), "| B:", tuple(B.weight.shape),
      "| scale = alpha/r =", scale, "| adapter dtype:", A.weight.dtype, "| base dtype:", layer.base_layer.weight.dtype)

x = torch.randn(2, 5, layer.base_layer.in_features, dtype=torch.bfloat16, device=device)
saved_B = B.weight.detach().clone()
with torch.no_grad():
    B.weight.copy_(torch.randn_like(B.weight) * 0.05)                     # make the correction non-zero, for this check only
    by_peft = layer(x)
    correction = B(A(x.to(A.weight.dtype))) * scale                       # the formula: (alpha / r) * B(A(x)), computed in the adapter's dtype
    by_hand = layer.base_layer(x) + correction.to(x.dtype)
    base_only = layer.base_layer(x)
    B.weight.copy_(saved_B)                                               # restore the zeros

diff = (by_peft.float() - by_hand.float()).abs().max().item()
size = (by_peft.float() - base_only.float()).abs().mean().item()
print(f"max difference between peft and the formula: {diff:.5f}  (bf16 rounding is about {by_peft.float().abs().max().item() * 2**-8:.5f} at this scale)")
print(f"mean size of the correction: {size:.4f}, so the check really does exercise the LoRA term")
assert diff <= by_peft.float().abs().max().item() * 2**-6 and size > 0
print("B restored to zeros:", bool((B.weight == 0).all()))
''')

md(r'''
### Why training through the backbone is expensive, and what we do about it

In notebooks 6 and 7 the backbone was frozen, so we ran it once and kept the vectors. Now the backbone's output changes every step, so there is no cache: **every training step runs the full 24-layer model forward and backward.** Two things follow.

**Time.** A forward and backward pass costs roughly three times a forward pass, and we need it for every question in every epoch, instead of once. This is why the training cell below is much slower than anything earlier.

**Memory.** To compute gradients, backpropagation needs the activations from the forward pass, and a 24-layer model stores a lot of them. I measured about 0.55 GB per sequence on this machine without any tricks, and this model's DeltaNet layers run in plain PyTorch (no fused kernels), which creates large intermediate tensors. **Gradient checkpointing** trades compute for memory: it stores only the input of each decoder layer and recomputes that layer's internals during the backward pass. With it, a batch of 16 should fit; if you run out of GPU memory, lower `LORA_BATCH` in the first code cell. We also pass `use_cache=False` so the model doesn't build a key/value cache that training never uses (without that flag, recomputation would append to the cache twice and crash).

To keep each epoch's cost down, an epoch shows every question **once**, in a fresh random option order each epoch (the label moves with its option), instead of the 6 fixed orders per question that the cached head used. Over several epochs this gives the same kind of order augmentation.
''')

md(r'''
### Training the backbone's adapters together with the head

Two choices to understand:
- **Warm start.** The new head starts from the trained frozen-backbone head. Because LoRA starts as a no-op, the combined model begins as exactly the model we already have, and the LoRA parameters get meaningful gradients from the first step. A randomly initialised head would first have to relearn what the frozen head already knows.
- **Two learning rates.** The LoRA matrices and the head are different kinds of parameters. The adapters use `2e-4` (a typical LoRA value) and the warm-started head uses `5e-4`. Both follow a one-cycle schedule.

`on_epoch_end` runs a quick validation after every epoch on 60 validation records, using the *current* adapters, so you can watch it learn without paying for a full evaluation.
''')

code(r'''
def snapshot_trainable():
    return {n: p.detach().clone() for n, p in core.named_parameters() if p.requires_grad}

def restore_trainable(snapshot):
    with torch.no_grad():
        for n, p in core.named_parameters():
            if n in snapshot:
                p.copy_(snapshot[n])

lora_init = snapshot_trainable()          # the starting point (all zeros in B), so we can retrain from scratch for the Brier experiment
variants = {}                             # name -> {"lora": snapshot, "head": state_dict}

quick_requests = build_requests(val_records[:min(60, N_VAL)], n_orders=2, seed=7)

def make_monitor(head):
    def monitor(epoch):
        fs = extract(core, renderer, quick_requests, PAD, HIDDEN, batch_size=32)
        r = evaluate(head, fs)
        print(f"  end of epoch {epoch}: quick validation accuracy {r['acc']:.3f}, loss {r['loss']:.3f}")
        return {"quick_val_acc": r["acc"], "quick_val_loss": r["loss"]}
    return monitor

def run_lora(name, brier_weight):
    restore_trainable(lora_init)
    head = MaskedPointerHead(HIDDEN).to(device)
    head.load_state_dict(head_frozen.state_dict())                        # warm start
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    history = train_lora(core, head, renderer, train_records, PAD, epochs=LORA_EPOCHS, batch_size=LORA_BATCH, lora_lr=2e-4, head_lr=5e-4,
                         brier_weight=brier_weight, seed=0, log_every=25, on_epoch_end=make_monitor(head))
    print(f"{name}: training took {time.time() - t0:.0f}s, peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    variants[name] = {"lora": snapshot_trainable(), "head": {k: v.detach().clone() for k, v in head.state_dict().items()}}
    return head, history

head_lora_ce, hist_ce = run_lora("lora_ce", brier_weight=0.0)
save_run("../runs/lora_ce", peft_model, head_lora_ce)
print("saved the adapter and head to ../runs/lora_ce")
''')

md(r'''
### Evaluating the adapted model

The validation features have to be extracted again, because the backbone is different now. Same questions, same option orders, so the comparison with the frozen baseline is like for like.
''')

code(r'''
def val_features(name):
    restore_trainable(variants[name]["lora"])
    return extract(core, renderer, val_requests, PAD, HIDDEN)

def head_of(name):
    head = MaskedPointerHead(HIDDEN).to(device)
    head.load_state_dict(variants[name]["head"])
    return head

val_fs_lora = val_features("lora_ce")
r_f, r_l = evaluate(head_frozen, val_fs_frozen), evaluate(head_of("lora_ce"), val_fs_lora)

print(f"{'':28s} {'accuracy':>9s} {'stability':>10s}   by question")
for name, r in [("frozen backbone + head", r_f), ("LoRA backbone + head", r_l)]:
    print(f"{name:28s} {r['acc']:9.3f} {r['stability']:10.3f}   {({k: round(v, 3) for k, v in r['by_question'].items()})}")
''')

md(r'''
**What to look at.** Compare the per-question numbers, not only the overall one. The error analysis in notebook 7 predicted that LoRA's benefit, if any, would show up on the questions where the frozen vectors lacked the right signal (the `department` question, especially the cases where the true department is missing from the options). Also compare `stability`: if the adapters learned to make each option's vector depend less on the order of the other options, it should rise.
''')

md(r'''
## 4. Brier as a training signal, not only a metric

So far we trained on cross-entropy (the NLL). The two losses differ in how they treat mistakes. Cross-entropy punishes a confident wrong answer without limit (`-log p` goes to infinity as `p` goes to 0), which pushes hard to fix the worst examples but also encourages extreme confidence everywhere else. The Brier score is bounded by 2 per example, so it is gentler on outliers and its gradient is smaller when the model is already confident.

We can add it as a second term: `loss = cross_entropy + BRIER_WEIGHT * brier`. This is the same Brier function as the metric; it is just differentiable, so it works as a loss. The experiment: train another LoRA model from the same starting point with `BRIER_WEIGHT = 1.0`, and also a frozen-backbone head with the same loss, then compare calibration. Because the only difference between the runs is the loss, any difference in Brier, ECE or NLL can be attributed to it, but with one seed each, treat small differences as noise.
''')

code(r'''
head_lora_brier, hist_brier = run_lora("lora_ce_brier", brier_weight=BRIER_WEIGHT)

head_frozen_brier, _ = train_head(train_fs, val_fs_frozen, HIDDEN, epochs=HEAD_EPOCHS, lr=1e-3, seed=0, brier_weight=BRIER_WEIGHT)
variants["frozen_ce"] = {"lora": lora_init, "head": {k: v.detach().clone() for k, v in head_frozen.state_dict().items()}}
variants["frozen_ce_brier"] = {"lora": lora_init, "head": {k: v.detach().clone() for k, v in head_frozen_brier.state_dict().items()}}
print("trained all four variants:", list(variants))
''')

md(r'''
## 5. Comparing all four variants on validation

Each row is one trained model. For every model we show the metrics as trained (`T = 1`), and again after temperature scaling with `T` fitted on this same validation set. The second block is optimistic for the reason given earlier, but it is the right way to compare how much a model *can* be calibrated by a single number. The final judgment is on the test set below.

Lower is better for NLL, Brier and ECE. Remember the reference: the uniform-guess Brier score printed above.
''')

code(r'''
feats = {"frozen_ce": val_fs_frozen, "frozen_ce_brier": val_fs_frozen}
for name in ["lora_ce", "lora_ce_brier"]:
    feats[name] = val_features(name)

val_scores = {name: predict_scores(head_of(name), feats[name]) for name in variants}
temps = {}

print("As trained (T = 1):")
for name in variants:
    metrics_row(name, val_scores[name], feats[name])

print("\nAfter temperature scaling (T fitted on this validation set):")
for name in variants:
    temps[name] = fit_temperature(val_scores[name], feats[name].mask.cpu(), feats[name].labels.cpu())
    metrics_row(name, val_scores[name], feats[name], temps[name])

print("\nBrier score of a uniform guesser, for reference: %.3f" % uniform_brier(mask_v).item())
''')

md(r'''
**Questions to ask of this table:**
1. **Accuracy vs calibration.** Does the model with the best accuracy also have the best Brier score and ECE? They need not agree.
2. **What does the Brier term do?** Compare each `*_brier` row with its cross-entropy-only twin: does it change accuracy, NLL, Brier and ECE, and in which direction?
3. **What does temperature scaling recover?** If a model's ECE drops a lot after one number is fitted, its ranking of options was fine and only its confidence was off. If it stays high, the problem is not just a scale.
4. **`T` is a diagnostic in itself.** `T > 1` means the model was overconfident and `T < 1` underconfident. `fit_temperature` limits `T` to between 0.05 and 20; if you see it at a limit, the scores carry almost no usable signal (a model at chance wants `T` as large as possible).
''')

md(r'''
## 6. The test set, once

We now choose one variant using **validation only**, then measure it on the test set a single time. The rule: lowest Brier score on validation after temperature scaling. The temperature used on the test set is the one fitted on validation; we do not refit it on test data.
''')

code(r'''
def val_brier_after_t(name):
    return calibration_report(val_scores[name], feats[name].mask.cpu(), feats[name].labels.cpu(), temps[name])["brier"]

chosen = min(variants, key=val_brier_after_t)
print("chosen on validation:", chosen, "| validation Brier after temperature scaling: %.4f" % val_brier_after_t(chosen))

restore_trainable(variants[chosen]["lora"])
test_fs = extract(core, renderer, build_requests(test_records, n_orders=4, seed=3), PAD, HIDDEN)
head_chosen = head_of(chosen)
scores_test = predict_scores(head_chosen, test_fs)
mask_t, labels_t = test_fs.mask.cpu(), test_fs.labels.cpu()

r = evaluate(head_chosen, test_fs)
print(f"\ntest accuracy {r['acc']:.3f} (chance {chance_accuracy(test_fs):.3f}) | order stability {r['stability']:.3f}")
print("by question:", {k: round(v, 3) for k, v in r["by_question"].items()})
print("Brier score of a uniform guesser on test: %.3f\n" % uniform_brier(mask_t).item())
metrics_row("test, as trained", scores_test, test_fs)
metrics_row("test, temperature from validation", scores_test, test_fs, temps[chosen])

print("\nReliability table on the test set, after temperature scaling:")
_, table = expected_calibration_error(probabilities(scores_test / temps[chosen], mask_t), labels_t)
print(format_reliability(table))
''')

md(r'''
## Wrap-up

What this notebook gave you:
- **LoRA, understood and verified.** The formula, the parameter count derived by hand and matched by `peft`, the proof that it starts as a no-op, and a manual reproduction of one layer's output.
- **The real cost of adapting the backbone.** No feature cache, a full forward and backward per question, and the memory consequences, handled with gradient checkpointing.
- **A calibration toolkit.** NLL, Brier, ECE and reliability tables, with hand-checkable tests, plus temperature scaling fitted on validation and judged on test.
- **A controlled comparison** of cross-entropy against cross-entropy plus Brier, for both the frozen and the adapted backbone.

How to interpret what you see: the data here is synthetic and keyword-driven, so accuracy is expected to be high for all variants and the interesting differences are in the per-question breakdown and in calibration. Treat any single-seed difference of a point or so as noise; the comparison becomes meaningful when it is repeated across seeds, which is a good next experiment.

**Next, notebook 9 (vision):** the original goal. Everything so far had text-only states. Next the state contains an **image**, the backbone's vision tower turns it into tokens, and the same pointer head, LoRA and calibration machinery decides questions about what is in the picture.
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
out = "/home/divye/Documents/Projects/create_jev/notebooks/08_lora_and_calibration.ipynb"
with open(out, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", out, "with", len(cells), "cells")
````

## `build_nb9.py`

Builds the vision notebook as JSON. Written at 00:48. Group: Notebook generator. Results: -.

````python
import json

cells = []
def md(s):   cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))

md(r'''
# 09 — Vision: a decision model whose state contains an image

**Where we are.** Qwen3.5-0.8B is a vision-language model, and until now we used only its text half. Everything so far, the delimiters, the pointer head, LoRA and the calibration tools, works on token sequences. An image can become tokens too, which means the same machinery can answer typed questions about a picture. This notebook builds that, and it is the point of the whole project.

**The plan:**
1. **How the model sees an image.** We open the vision side: what the processor produces, how many tokens an image costs, and proof that the model really responds to pixels.
2. **A dataset with exact labels.** A generator draws simple scenes (coloured circles, squares and triangles) and writes the questions with ground truth known by construction.
3. **Extending the pipeline.** The state becomes a list of parts (images and text). A new renderer turns an image part into the model's own image tokens and carries the pixels along.
4. **Training:** a pointer head on the frozen backbone first, then LoRA.
5. **Controls that check we are not fooling ourselves:** does the model use the image, or is it guessing from the question alone?
6. **Calibration and the test set**, with the tools from notebook 8.

**Why synthetic shapes?** Real photos would be more impressive and much harder to label. With generated scenes every label is correct by construction, we control the difficulty, and nothing needs downloading. The pipeline doesn't care where the images come from: the dataset records just contain a file path, so you can later point them at your own photos with your own questions and nothing in the code changes.

**Running it.** Set `SMOKE = True` in the first code cell for a tiny dry run (a few minutes) that only checks every cell executes. Leave it `False` for the real experiment. As in notebook 8, shut down other notebook kernels first: images make every sequence longer, and training through the backbone needs memory.
''')

code(r'''
%load_ext autoreload
%autoreload 2

import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import sys
sys.path.insert(0, "..")

import inspect, math, time
from collections import defaultdict

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForMultimodalLM

from jev.vision import MMRenderer, IMAGE_PLACEHOLDER
from jev.shapes import generate_dataset
from jev.head import MaskedPointerHead
from jev.data import load_jsonl
from jev.features import build_requests, render_requests, encode, extract
from jev.calibration import probabilities, uniform_brier, expected_calibration_error, fit_temperature, calibration_report, format_reliability
from jev.train import evaluate, chance_accuracy, predict_scores, train_head, add_lora, train_lora, save_run

SMOKE = False   # True = tiny dry run that only checks the notebook executes. False = the real experiment.

SIZES = {"train": 16, "val": 8, "test": 8} if SMOKE else {"train": 800, "val": 200, "test": 200}
DATA_DIR = "../data/shapes_smoke" if SMOKE else "../data/shapes"
N_ORDERS_TRAIN = 2 if SMOKE else 4
HEAD_EPOCHS = 2 if SMOKE else 30
LORA_EPOCHS = 1 if SMOKE else 2
LORA_BATCH = 4 if SMOKE else 16       # lower this if you run out of GPU memory
print("SMOKE" if SMOKE else "FULL", "run | images:", SIZES)
''')

code(r'''
MODEL_ID = "Qwen/Qwen3.5-0.8B"
processor = AutoProcessor.from_pretrained(MODEL_ID)
tokenizer = processor.tokenizer
model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto").eval()

core = model.model
for p in core.parameters():
    p.requires_grad_(False)

device = next(core.parameters()).device
HIDDEN = model.config.text_config.hidden_size
PAD = tokenizer.pad_token_id
print("loaded | GPU memory allocated: %.2f GB" % (torch.cuda.memory_allocated() / 1e9))
''')

md(r'''
## 1. How the model sees an image

An image goes through four steps before the language model reads it.

1. **Patches.** The image processor cuts the picture into square patches of `patch_size` pixels (16 here).
2. **Vision tower.** A transformer (a ViT) turns every patch into a feature vector.
3. **Merge.** Each `merge_size x merge_size` block of neighbouring patches (2 x 2 here) is merged into one vector and projected to the language model's width (1024). Four patches become one token, which keeps sequences short.
4. **Insertion.** Those vectors replace special placeholder tokens (`<|image_pad|>`) in the text sequence. From there on, the language model treats them like any other token, and its causal attention lets every later token, including our `<|fim_suffix|>` decide token, read them.

Let's look at the real numbers from this model.
''')

code(r'''
vc = model.config.vision_config
ip = processor.image_processor
print("image processor :", {k: getattr(ip, k) for k in ("patch_size", "merge_size", "temporal_patch_size")})
print("vision tower    :", {k: getattr(vc, k) for k in ("depth", "hidden_size", "num_heads", "out_hidden_size")})
print(f"parameters      : vision tower {sum(p.numel() for p in core.visual.parameters()) / 1e6:.0f}M, language model {sum(p.numel() for p in core.language_model.parameters()) / 1e6:.0f}M")
print("minimum image area the processor will use (pixels):", ip.size["shortest_edge"], "=", int(ip.size["shortest_edge"] ** 0.5), "x", int(ip.size["shortest_edge"] ** 0.5))
''')

md(r'''
The vision tower is a 12-layer transformer with 768-dimensional features, about 100M parameters, and its output width (`out_hidden_size`) is 1024, the language model's hidden size. That match is what lets image tokens and text tokens share a sequence.

Now feed the processor a text containing the image placeholder, plus one image. The processor **expands** the single `<|image_pad|>` into as many copies as the image needs.
''')

code(r'''
def demo_image(color, shape="circle", size=224):
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    box = [size * 0.25, size * 0.25, size * 0.75, size * 0.75]
    draw.ellipse(box, fill=color) if shape == "circle" else draw.rectangle(box, fill=color)
    return image

text = IMAGE_PLACEHOLDER + "What color is the shape?"
inputs = processor(text=[text], images=[demo_image("red")], return_tensors="pt")

print("what the processor returns:")
for k, v in inputs.items():
    print(f"  {k:18s} shape {tuple(v.shape)}  dtype {v.dtype}")

grid_t, grid_h, grid_w = inputs["image_grid_thw"][0].tolist()
n_pad = int((inputs["input_ids"] == tokenizer.convert_tokens_to_ids("<|image_pad|>")).sum())
merge = ip.merge_size
print(f"\nimage_grid_thw = [time {grid_t}, height {grid_h}, width {grid_w}] patches -> {grid_t * grid_h * grid_w} patches in total")
print(f"each patch row in pixel_values has {inputs['pixel_values'].shape[1]} numbers = 3 colours x {ip.temporal_patch_size} frames x {ip.patch_size} x {ip.patch_size} pixels")
print(f"<|image_pad|> tokens after expansion: {n_pad} = {grid_t * grid_h * grid_w} patches / {merge * merge} (a {merge}x{merge} block merges into one token)")
assert n_pad == grid_t * grid_h * grid_w // (merge * merge)
print(f"sequence length: {inputs['input_ids'].shape[1]} = 1 vision_start + {n_pad} image tokens + 1 vision_end + {inputs['input_ids'].shape[1] - n_pad - 2} text tokens")
print("mm_token_type_ids values:", sorted(set(inputs["mm_token_type_ids"].flatten().tolist())), "(1 marks the image-token positions)")
''')

md(r'''
**What each piece is for:**
- `pixel_values` is the pixel data, one row per patch. The processor duplicates a still image into `temporal_patch_size = 2` frames because the model was built to handle video too; that's why each row has 3 x 2 x 16 x 16 = 1536 numbers.
- `image_grid_thw` records the layout (time, height, width, counted in patches) so the model knows how to arrange the patches and how many tokens to expect.
- `mm_token_type_ids` marks which positions of the sequence are image tokens (1) and which are text (0). The model needs it because positions are **multi-axis** here: an image token has a height and a width as well as an index in the sequence (this is the "mRoPE" in the config), and the model builds them from this mask.

**Cost.** An image costs tokens in proportion to its area. The processor also **enforces a minimum area** of 65,536 pixels (256 x 256), so smaller images are enlarged first. Our 224 x 224 demo image became 256 x 256 (a 16 x 16 patch grid, 64 tokens). Check how the cost scales:
''')

code(r'''
for side in (224, 256, 384, 512):
    out = processor(text=[IMAGE_PLACEHOLDER], images=[demo_image("red", size=side)], return_tensors="pt")
    g = out["image_grid_thw"][0].tolist()
    print(f"{side} x {side} image -> grid {g[1]} x {g[2]} patches -> {out['input_ids'].shape[1] - 2} image tokens")
''')

md(r'''
Our dataset uses 256 x 256 images, the cheapest size the processor allows: 64 image tokens each. A 512 x 512 photo would cost 256 tokens, four times as much, and training cost grows with sequence length.

### Does the model actually respond to the pixels?

Two checks. First, run the backbone on the same text with two different images (a red circle and a blue square) and compare the final token's hidden state. If the image did not matter, the cosine would be exactly 1. Second, ask the model, in plain generation mode, what colour the shape is. That uses the model's language head, which our jev model discards, but it is the simplest proof that the whole vision-language stack is wired correctly in this environment.
''')

code(r'''
def backbone_last_state(image):
    batch = processor(text=[text], images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        return core(**batch).last_hidden_state[0, -1].float()

h_red, h_blue = backbone_last_state(demo_image("red")), backbone_last_state(demo_image("blue", "square"))
print("cosine of the last token's hidden state, red circle vs blue square: %.4f   (1.0 would mean the image is ignored)" % F.cosine_similarity(h_red, h_blue, dim=0).item())

chat = [{"role": "user", "content": [{"type": "image", "image": demo_image("red")}, {"type": "text", "text": "What color is the shape? Answer with one word."}]}]
batch = processor.apply_chat_template(chat, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(device)
generated = model.generate(**batch, max_new_tokens=6, do_sample=False)
print("model.generate says:", repr(processor.decode(generated[0][batch["input_ids"].shape[-1]:], skip_special_tokens=True)))
''')

md(r'''
## 2. A dataset with exact labels

`jev/shapes.py` draws scenes: 1 to 4 non-overlapping shapes (circle, square or triangle), each in one of six colours and with a distinct size, on a plain grey background. Because the generator places every shape, the answers are known exactly. Each image gets four questions:

| question | type | options | how the label is known |
|---|---|---|---|
| `largest_color` | choice | 3 to 5 colours, one of which is right | the colour of the shape with the largest side |
| `has_triangle` | noul (yes/no) | no, yes | whether any placed shape is a triangle |
| `count` | score (ordered) | one, two, three, four shapes | the number of shapes placed |
| `leftmost` | choice | circle, square, triangle (shuffled) | the shape type with the smallest x-position |

Half of the records also carry a text caption ("Photo uploaded by a user.") that says nothing about the answer. It's there so the state mixes two kinds of parts, an image and some text, as real tickets would. A record's `state` is a **list of parts**, and the image part is a file path:

`{"state": [{"type": "image", "path": "images/train_00000.png"}, {"type": "text", "text": "..."}], "questions": [...]}`
''')

code(r'''
if not os.path.exists(f"{DATA_DIR}/train.jsonl"):
    print("generated:", generate_dataset(DATA_DIR, SIZES, seed=0))
train_records, val_records, test_records = (load_jsonl(f"{DATA_DIR}/{s}.jsonl") for s in ("train", "val", "test"))
print({"train": len(train_records), "val": len(val_records), "test": len(test_records)})

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for ax, rec in zip(axes, train_records[:4]):
    ax.imshow(Image.open(f"{DATA_DIR}/{rec['state'][0]['path']}"))
    ax.axis("off")
    ax.set_title("\n".join(f"{q['id']}: {q['options'][q['label']]}" for q in rec["questions"]), fontsize=8, loc="left")
plt.tight_layout()
plt.show()
''')

code(r'''
print("label counts per question (training set):")
for qid in ("largest_color", "has_triangle", "count", "leftmost"):
    counts = defaultdict(int)
    for rec in train_records:
        q = next(q for q in rec["questions"] if q["id"] == qid)
        counts[q["options"][q["label"]]] += 1
    print(f"  {qid:14s}", dict(sorted(counts.items(), key=lambda kv: -kv[1])))
''')

md(r'''
## 3. Extending the pipeline to images

Three changes make the earlier machinery handle images, all in the code you already know:

**The renderer.** `MMRenderer` (in `jev/vision.py`) extends `Renderer`. For each state part it adds token ids: a text part goes through the usual sanitised tokenizer, and an image part becomes the model's own image block (`<|vision_start|>`, 64 copies of `<|image_pad|>`, `<|vision_end|>`), obtained by running the image through the processor. It also returns a fourth item with the pixel data, the grid and the token-type mask. The delimiters, option markers and decide token are unchanged. The image comes before the question, so the question, every option and the decide token can all attend to it.

**The encoder.** `jev.features.encode` now checks whether any rendered question has images. If so, it concatenates the batch's pixel data, grids and token types and passes them to the backbone along with the ids. Questions without images work exactly as before.

**Positions.** The readout positions (each option's `<|box_end|>` and the decide token) are found exactly as before; the image only changes the numbers in front of them.

Render one question to see it. The 64 identical image tokens are collapsed for display.
''')

code(r'''
renderer = MMRenderer(processor, DATA_DIR)

def show(ids):
    out, i = [], 0
    pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    while i < len(ids):
        if ids[i] == pad_id:
            j = i
            while j < len(ids) and ids[j] == pad_id:
                j += 1
            out.append(f"<|image_pad|>x{j - i}")
            i = j
        else:
            out.append(tokenizer.decode([ids[i]]))
            i += 1
    return "".join(out)

ids, ends, decide_pos, extra = renderer.render_question(train_records[0], 0)
print(show(ids))
print(f"\n{len(ids)} tokens in total, {sum(extra['types'])} of them image tokens | option-end positions {ends} | decide position {decide_pos}")
print("pixel data carried alongside:", [tuple(p.shape) for p in extra["pixels"]], "| grids:", [g.tolist() for g in extra["grids"]])
print("token at the decide position:", tokenizer.decode([ids[decide_pos]]))
''')

md(r'''
### Two checks before training anything

**Batching is safe.** As in notebook 7, a padded batch must give the same vectors as one-at-a-time processing, up to bf16 rounding. With images there is more room for mismatch (the pixel data of the whole batch is concatenated and the model must match each image to the right row), so this is worth checking directly.

**The readout depends on the image.** Take one rendered question and replace its pixels with another record's image, leaving the tokens identical. If the decide vector doesn't move, the image isn't reaching the readout.
''')

code(r'''
probe = build_requests(train_records[:6], n_orders=1, seed=0)[:12]
rendered = render_requests(renderer, probe)
n_max = max(len(r[1]) for r in rendered)

with torch.no_grad():
    opt_b, dec_b, _ = encode(core, rendered, list(range(len(rendered))), PAD, n_max)
    singles = [encode(core, rendered, [i], PAD, n_max) for i in range(len(rendered))]
cos = min(F.cosine_similarity(dec_b[i].float(), singles[i][1][0].float(), dim=0).item() for i in range(len(rendered)))
print("batch of %d vs one at a time: minimum cosine of the decide vectors %.5f" % (len(rendered), cos))

other = rendered[1][3]                                    # the pixels of a different record
swapped = [(rendered[0][0], rendered[0][1], rendered[0][2], other)]
with torch.no_grad():
    _, dec_swapped, _ = encode(core, swapped, [0], PAD, n_max)
print("decide vector with its own image vs another record's image: cosine %.4f   (1.0 would mean the readout ignores the image)" % F.cosine_similarity(dec_b[0].float(), dec_swapped[0].float(), dim=0).item())
''')

md(r'''
The cosine will be close to 1 even when the image matters, most likely because the decide token is a rarely used special token whose vector is dominated by its own identity (we saw in notebook 5 that all the readout vectors are very similar to each other). What counts is that it is *not exactly* 1 and that a trained head can find and use the difference. The controls in section 5 test that properly.
''')

md(r'''
## 4. A pointer head on the frozen backbone

This is notebook 7's recipe, now with images. The backbone, including the vision tower, is frozen. We run every training question through it once (each in `N_ORDERS_TRAIN` option orders), keep the readout vectors, and train the head on those.

Images make each sequence about 100 tokens instead of 50, so extraction is slower than in notebook 7. We use a smaller extraction batch (32) because the vision tower adds memory.
''')

code(r'''
val_requests = build_requests(val_records, n_orders=4, seed=2)

t0 = time.time()
train_fs = extract(core, renderer, build_requests(train_records, n_orders=N_ORDERS_TRAIN, seed=1), PAD, HIDDEN, batch_size=32)
val_fs_frozen = extract(core, renderer, val_requests, PAD, HIDDEN, batch_size=32)
print(f"cached {len(train_fs)} train and {len(val_fs_frozen)} val examples in {time.time() - t0:.0f}s")

head_frozen, _ = train_head(train_fs, val_fs_frozen, HIDDEN, epochs=HEAD_EPOCHS, lr=1e-3, seed=0)

def chance_by_question(fs):
    acc = defaultdict(list)
    for qid, opts in zip(fs.meta["qid"], fs.meta["options"]):
        acc[qid].append(1 / len(opts))
    return {k: sum(v) / len(v) for k, v in acc.items()}

def report_by_question(title, head, fs):
    r = evaluate(head, fs)
    chance = chance_by_question(fs)
    print(f"{title}: accuracy {r['acc']:.3f} (chance {chance_accuracy(fs):.3f}) | order stability {r['stability']:.3f}")
    for qid, acc in r["by_question"].items():
        print(f"   {qid:14s} accuracy {acc:.3f}   chance {chance[qid]:.3f}")
    return r

r_frozen = report_by_question("frozen backbone + head, validation", head_frozen, val_fs_frozen)
''')

md(r'''
**How to read it.** Compare each question with its own chance level, which differs (a yes/no question has chance 0.5, a 4-way count has 0.25, a colour question with 4 options has about 0.25). A frozen backbone was never asked to represent "how many shapes" or "what colour is the biggest one" in the vector at the decide token, so this experiment measures how much of that information survives at the readout position. Notice which questions come out well above chance and which do not; that tells you what the frozen representation already carries.
''')

md(r'''
## 5. Controls: is the model using the image?

A high number on a vision task proves little until you rule out the alternative: the model might be answering from the question text and the label frequencies alone. Two controls remove the image's information while keeping everything else.

**Swapped images.** Give each validation record the state (image and caption) of the *next* record, keeping its own questions and labels. The text is unchanged, the labels are unchanged, but the image no longer matches. If the model uses the image, accuracy should fall to roughly chance on the image questions. If accuracy stays high, the model is not looking at the picture.
''')

code(r'''
def with_next_state(records):
    states = [r["state"] for r in records]
    return [{**r, "state": states[(i + 1) % len(records)]} for i, r in enumerate(records)]

val_fs_swapped = extract(core, renderer, build_requests(with_next_state(val_records), n_orders=4, seed=2), PAD, HIDDEN, batch_size=32)
r_swap = report_by_question("frozen backbone + head, validation with mismatched images", head_frozen, val_fs_swapped)
print("\nlargest drop in accuracy when the image is taken away:", {k: round(r_frozen["by_question"][k] - r_swap["by_question"][k], 3) for k in r_swap["by_question"]})
''')

md(r'''
The drop in accuracy (real images minus mismatched images), question by question, is the amount of each answer that comes from the picture. A question whose accuracy does not drop was being answered without the image. Keep this control for every later model. It is the cheapest and strongest evidence that the vision path is doing the work.
''')

md(r'''
## 6. LoRA on the language layers

A frozen backbone can only expose what is already easy to read at the decide token. LoRA lets the language layers learn to route the relevant image information to the readout positions. Notebook 8 covers the mechanics (what LoRA is, why it starts as a no-op, why training needs gradient checkpointing and costs far more than the cached head); this notebook just applies them.

**What gets adapted:** the 24 language decoder layers' linear projections, about 5M parameters. The **vision tower stays frozen**: the pattern in `jev/train.py` only matches modules under `language_model`. That is a design choice, not a limit; adapting the vision tower as well is possible, and it is the obvious next experiment if the colour or counting questions stay weak.

The head is warm-started from the frozen-backbone head, as in notebook 8. Each epoch shows every question once, in a fresh random option order.
''')

code(r'''
variants = {"frozen": {"head": {k: v.detach().clone() for k, v in head_frozen.state_dict().items()}}}

peft_model = add_lora(core, rank=8, alpha=16)
print("trainable parameters: %.2fM | inside the vision tower: %d" % (sum(p.numel() for p in core.parameters() if p.requires_grad) / 1e6,
                                                                    sum(1 for n, p in core.named_parameters() if p.requires_grad and "visual" in n)))

quick_requests = build_requests(val_records[:min(60, len(val_records))], n_orders=2, seed=7)
head_lora = MaskedPointerHead(HIDDEN).to(device)
head_lora.load_state_dict(head_frozen.state_dict())

def monitor(epoch):
    fs = extract(core, renderer, quick_requests, PAD, HIDDEN, batch_size=32)
    r = evaluate(head_lora, fs)
    print(f"  end of epoch {epoch}: quick validation accuracy {r['acc']:.3f}, loss {r['loss']:.3f}")
    return {"quick_val_acc": r["acc"]}

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
history = train_lora(core, head_lora, renderer, train_records, PAD, epochs=LORA_EPOCHS, batch_size=LORA_BATCH, lora_lr=2e-4, head_lr=5e-4,
                     seed=0, log_every=25, on_epoch_end=monitor)
print(f"\ntraining took {time.time() - t0:.0f}s, peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
variants["lora"] = {"head": {k: v.detach().clone() for k, v in head_lora.state_dict().items()}}
save_run("../runs/shapes_lora", peft_model, head_lora)
''')

code(r'''
val_fs_lora = extract(core, renderer, val_requests, PAD, HIDDEN, batch_size=32)
r_lora = report_by_question("LoRA backbone + head, validation", head_lora, val_fs_lora)

print("\nthe same controls for the adapted model:")
val_fs_lora_swapped = extract(core, renderer, build_requests(with_next_state(val_records), n_orders=4, seed=2), PAD, HIDDEN, batch_size=32)
r_lora_swap = report_by_question("LoRA, validation with mismatched images", head_lora, val_fs_lora_swapped)
''')

md(r'''
**What to compare.**
- **Frozen vs LoRA, per question.** Which questions improved, and which stayed at chance? Counting and spatial questions ("furthest left") are harder for a small model than reading a colour.
- **The mismatched-image control, for the adapted model.** The adapted model should lose *more* accuracy when the image is swapped than the frozen one if it has learned to use the picture more.
- **Order stability** for the choice and yes/no questions.
''')

md(r'''
## 7. Calibration, and the test set once

The calibration tools from notebook 8 apply unchanged, since they only see scores. For each variant we report accuracy, NLL, Brier score and ECE; then fit a temperature on validation. We pick the variant with the lower validation Brier score after temperature scaling and evaluate **only that one** on the test set, once.
''')

code(r'''
def metrics_row(name, scores, fs, temperature=1.0):
    r = calibration_report(scores, fs.mask.cpu(), fs.labels.cpu(), temperature)
    print(f"{name:34s} T={temperature:5.2f} | acc {r['acc']:.3f} | NLL {r['nll']:.3f} | Brier {r['brier']:.3f} | ECE {r['ece']:.3f}")
    return r

def head_of(name):
    head = MaskedPointerHead(HIDDEN).to(device)
    head.load_state_dict(variants[name]["head"])
    return head

val_feats = {"frozen": val_fs_frozen, "lora": val_fs_lora}
val_scores = {name: predict_scores(head_of(name), val_feats[name]) for name in variants}
print("Brier score of a uniform guesser on validation: %.3f\n" % uniform_brier(val_fs_frozen.mask.cpu()).item())

temps = {}
for name in variants:
    metrics_row(name + ", as trained", val_scores[name], val_feats[name])
    temps[name] = fit_temperature(val_scores[name], val_feats[name].mask.cpu(), val_feats[name].labels.cpu())
    metrics_row(name + ", temperature-scaled", val_scores[name], val_feats[name], temps[name])
''')

code(r'''
def val_brier_after_t(name):
    return calibration_report(val_scores[name], val_feats[name].mask.cpu(), val_feats[name].labels.cpu(), temps[name])["brier"]

chosen = min(variants, key=val_brier_after_t)
print("chosen on validation:", chosen)

if chosen == "frozen":
    # the LoRA adapters are still in the model; zero them out so the backbone is the original one again
    with torch.no_grad():
        for n, p in core.named_parameters():
            if p.requires_grad and n.endswith("lora_B.default.weight"):
                p.zero_()
test_fs = extract(core, renderer, build_requests(test_records, n_orders=4, seed=3), PAD, HIDDEN, batch_size=32)
head_chosen = head_of(chosen)
scores_test = predict_scores(head_chosen, test_fs)
mask_t, labels_t = test_fs.mask.cpu(), test_fs.labels.cpu()

report_by_question("test set (" + chosen + ")", head_chosen, test_fs)
print("\nBrier score of a uniform guesser on test: %.3f\n" % uniform_brier(mask_t).item())
metrics_row("test, as trained", scores_test, test_fs)
metrics_row("test, temperature from validation", scores_test, test_fs, temps[chosen])
print("\nReliability table on the test set, after temperature scaling:")
_, table = expected_calibration_error(probabilities(scores_test / temps[chosen], mask_t), labels_t)
print(format_reliability(table))
''')

md(r'''
## Wrap-up

What you now have is a complete vision decision model:
- **The image path, understood.** Patches, the vision tower, the 2 x 2 merge, the expanded `<|image_pad|>` run, and the three extra inputs (`pixel_values`, `image_grid_thw`, `mm_token_type_ids`) that carry an image alongside the token ids.
- **A renderer and encoder that accept a mixed state** of image and text parts, with the same delimiters, readout positions and sanitisation as the text-only pipeline.
- **A dataset with exact labels** and a control (mismatched images) that tells you whether a result comes from the picture.
- **The full set of tools**: a frozen-backbone head, LoRA on the language layers, calibration metrics with temperature scaling, and a test set touched once.

**Using your own images.** Write records whose `state` is `[{"type": "image", "path": "..."}, {"type": "text", "text": "..."}]` and whose `questions` follow the schema from notebook 4 (`id`, `prompt`, `type`, `options`, `label`). `MMRenderer(processor, image_root)` resolves paths relative to `image_root`. Everything else in this notebook runs unchanged. Keep an eye on image size: tokens grow with area, and 256 x 256 is the cheapest the processor allows.

**Natural next experiments:** adapt the vision tower too, add more scene complexity (overlapping shapes, more colours), try rank 16 or 32, run several seeds to put error bars on the frozen-vs-LoRA comparison, and repeat the mismatched-image control on every variant.
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
out = "/home/divye/Documents/Projects/create_jev/notebooks/09_vision.ipynb"
with open(out, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", out, "with", len(cells), "cells")
````
