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
