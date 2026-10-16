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
