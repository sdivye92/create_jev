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
