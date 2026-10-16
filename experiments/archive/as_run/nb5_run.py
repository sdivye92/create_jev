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