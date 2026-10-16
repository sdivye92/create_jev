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
