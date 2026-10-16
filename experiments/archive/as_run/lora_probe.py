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
