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
