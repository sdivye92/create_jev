# Experiment results log

All results from building the jev pipeline on Qwen3.5-0.8B, in the order they were measured. Everything trained here is a pointer head on a frozen backbone. LoRA and images came later and are reported in [RESULTS_LORA.md](RESULTS_LORA.md) and [RESULTS_VISION.md](RESULTS_VISION.md).

## Best configuration (pointer head on the frozen backbone)

**Recipe** (the model of notebook 7):

| Hyperparameter | Value |
|---|---|
| Training records | 800 |
| Option orders per question | 6, shuffled with the label moved |
| Input normalisation | RMS |
| Learning-rate schedule | One-cycle, peak 1e-3 |
| Epochs | 30 |
| Batch size | 64 |

| Metric | Validation (5 seeds) | Test (seed 0, used once) |
|---|---|---|
| Accuracy | 0.960 +- 0.001 | 0.959 |
| Loss | 0.126 | 0.131 |
| Order stability | 0.900 | 0.900 |
| Chance accuracy | 0.352 | 0.352 |

**Why these choices beat the alternatives (validation, from sections 3.4 to 3.6):**

| Choice | Result | Why it probably worked |
|---|---|---|
| One-cycle schedule instead of a constant learning rate | Accuracy 0.960 +- 0.001 against 0.921 +- 0.038, and the worst seed went from 0.847 to 0.959. | A constant rate keeps jumping around the minimum, so the result depended on which epoch you stopped at. The schedule ends with tiny steps and settles. |
| 30 epochs instead of 15 | 0.960 against 0.949 with the same peak rate. | The schedule needs enough steps to learn the harder questions. A peak of 3e-3 for 15 epochs reached 0.956, so a higher rate gets close, but 1e-3 for 30 epochs was the best overall. |
| Shuffled option orders (6) instead of one fixed order | Accuracy 0.949 against 0.887 and order stability 0.892 against 0.726. | The model reads options left to right, so position leaks into the answer. Shuffling with the label moved forces the head to use the content, and also acts as data augmentation. |
| Input normalisation on | The same accuracy (0.960 against 0.959), but loss 0.126 against 0.208. | The vectors have a large shared component, and normalising stops it from dominating. The probabilities are better, which matters for calibration. |
| Gradient clipping | Not used. 0.952 against 0.949 at 15 epochs, a difference inside the noise. | The schedule already keeps training stable. |
| 800 records instead of 8 | Held-out accuracy about 0.5 at 8 records (chance 0.39) and 0.96 at 800. | Data was the bottleneck, and the learning curve was still rising. |

LoRA and vision are not part of this recipe. Their best configurations are in [RESULTS_LORA.md](RESULTS_LORA.md) and [RESULTS_VISION.md](RESULTS_VISION.md).

## 0. How to read this file

### Setup (all experiments)
- Backbone: `Qwen/Qwen3.5-0.8B`, 852,985,920 parameters (vision tower 100,592,896, language model 752,393,024), loaded in bf16, frozen, `eval()` mode.
- Head: fp32. The notebook 5 and 6 head (`jev/model.py`) has 524,288 trainable parameters, which is 0.061% of the model. The notebook 7 head (`jev/head.py`) has the same size.
- Hardware and software: RTX 5060 Laptop GPU (8 GB), torch 2.14.1+cu130, transformers 5.19.0.dev0. The linear-attention fast kernels (`causal_conv1d`, `flash-linear-attention`) are not installed, so the reference PyTorch fallbacks run.
- Readout: `last_hidden_state` at each option's `<|box_end|>` token and at the final `<|fim_suffix|>` token. The loss is cross-entropy over the options of each question.

### Metric definitions
- **Accuracy**: fraction of examples where the highest-scoring option is the labelled one. Each (question, option order) pair is one example.
- **Chance accuracy**: the mean of 1 / (number of options) over examples, i.e. uniform random guessing.
- **Majority baseline** (notebook 6 only): for each question id, always predict the option text that was most common in the training records.
- **Order stability**: group examples by (record, question), and count a group as stable if the model picked the same option *text* in every order. Notebook 6 includes `score` questions in this number (they are never shuffled, so they are trivially stable and set a floor of one third). Notebook 7 excludes them. So **stability numbers are not comparable between notebooks 6 and 7.**
- **± values**: population standard deviation across seeds, where seeds change the head initialisation and the batch order.

### Where each number came from
- Numbers marked **(notebook)** were produced by running the notebook code itself, with the TODO cells filled with the reference solutions.
- Numbers marked **(prototype)** came from scratch scripts I ran while designing the notebooks. Those scripts are archived as they were run in [experiments/archive/as_run/](../experiments/archive/as_run/ALL_SCRIPTS.md), and the notebooks' final settings differ from them. They are included because they explain design choices.

---

## 1. Pre-training measurements

### 1.1 Cache reuse and bf16 (notebooks 2 and 3, and the investigation after)
The question: if the shared `state` prefix is computed once and its cache cloned per question, does it equal running `state + question` from scratch?

| dtype | max abs diff, question A | max abs diff, question B | `allclose(atol=1e-2)` |
|---|---|---|---|
| fp32 | 2.1e-5 | 1.4e-5 | True |
| bf16 | 0.375 | 0.250 | False |

- Final-layer hidden states have magnitude up to about 30. One bf16 step at magnitude 16-32 is 0.125, and at 32-64 it is 0.25, so the bf16 gap is 1-3 units of rounding.
- Against an fp32 ground truth: the bf16 baseline is 0.242 / 0.203 away, the bf16 cache-reuse path is 0.231 / 0.163 away, and the two bf16 paths differ from each other by 0.375 / 0.250. Cosine between them is 0.9999 and relative L2 is 1.4%.
- Repeating an identical call gives a max abs difference of exactly 0.0, so the model is deterministic.
- Conclusion: cache reuse is correct. The earlier suspicion of a bug was bf16 rounding, and an explicit `position_ids` made no difference because the model derives positions from the cache length. Files: `experiments/investigations/cache_reuse/cache_reuse_debug.py`, `experiments/investigations/cache_reuse/cache_reuse_findings.md`.
- Rows mode (one independent full pass per question) matched the reference exactly (`torch.equal` True).

### 1.2 The readout vectors before any training (notebook 5, one smoke question with 3 options)
- Vector norms: options 116.4 / 110.7 / 109.8, decide vector 125.1.
- Cosine between the three option vectors: 0.834, 0.821, 0.981 (pairwise). Cosine of each option with the decide vector: 0.596, 0.519, 0.529.
- Raw dot product of option and decide vectors: 8679, 7185, 7265. Divided by sqrt(1024): 271.2, 224.5, 227.0. Softmax of that: [1.0, 0.0, 0.0].
- `lm_head` is tied to the input embedding (same tensor object). `hidden_states[-1]` equals `last_hidden_state`.
- Logits tensor that `lm_head` would create: 1.59 GB in bf16 and 3.18 GB in fp32 at batch 8 and 400 tokens.
- Untrained head (seed 0): scores [1.546, -1.107, -0.847], probabilities [0.861, 0.061, 0.079].
- Option order sensitivity, untrained head: reversing the options and mapping scores back gives [0.880, 0.101, 1.233]. The top choice flipped from billing to sales. The same option's vector had cosine 0.818 / 0.988 / 0.817 with itself across the two orders.
- Gradient check: the head received gradients and the frozen backbone received none. Trainable parameters: 524,288 of 853,510,208.

---

## 2. Notebook 6: the 12-record smoke set

### 2.1 Data
`data/smoke_12.jsonl`: 12 hand-written customer-support records, three questions each (department with 3 options, escalate yes/no, frustration with 3 ordered levels). The first 8 records are for training and the last 4 are held out. The split is by record, so a held-out record's text is never seen in training.

| | department | escalate | frustration |
|---|---|---|---|
| train labels | billing 2, technical 4, sales 2 | no 6, yes 2 | calm 5, frustrated 1, furious 2 |
| held-out labels | billing 2, technical 1, sales 1 | no 2, yes 2 | calm 1, frustrated 2, furious 1 |

- Each training question is shown in 6 option orders (the original plus 5 shuffles; `score` questions are never shuffled), giving 144 training examples from only 24 distinct (text, question) pairs. Held-out has 72 examples (12 questions x 6 orders).
- Reference accuracies on held-out: chance 0.389, majority baseline 0.333.
- `shuffle_question` check: 811 of 1,800 shuffled copies differed from the original order, and every copy kept the label on the same option text.
- Feature caching: 144 train examples in 3.4-3.8 s, so about 26 ms per example one at a time.

### 2.2 First run with the original settings (notebook, lr 1e-3, 60 epochs, seed 0)
- Epoch-0 mean loss 26.6 (a model that knows nothing scores ln 3 = 1.10), 6.1 at epoch 10, 0.0000 from epoch 20.
- Train accuracy 1.000. Held-out accuracy 0.431 (choice 0.375, noul 0.417, score 0.500), order stability 0.333.
- With one order only (360 epochs): held-out accuracy 0.486, stability 0.667. This looked like shuffling hurt, but it was an artefact of training at an unstable learning rate (see 2.3).

### 2.3 Learning rate sweep (prototype, 3 seeds each, 6 orders vs 1 order)
Epoch counts: 40 epochs with 6 orders, 240 with 1 order (about equal gradient steps). Train accuracy was 1.000 in every row.

| lr | orders | epoch-0 loss (3-seed mean) | held-out accuracy | order stability |
|---|---|---|---|---|
| 1e-3 | 1 | 21.01 | 0.444 ± 0.034 | 0.722 |
| 1e-3 | 6 | 49.50 | 0.505 ± 0.062 | 0.667 |
| 3e-4 | 1 | 5.33 | 0.440 ± 0.056 | 0.528 |
| 3e-4 | 6 | 8.10 | 0.579 ± 0.105 | 0.861 |
| 1e-4 | 1 | 2.54 | 0.500 ± 0.030 | 0.528 |
| 1e-4 | 6 | 2.81 | 0.509 ± 0.017 | 0.806 |
| 3e-5 | 1 | 1.29 | 0.486 ± 0.030 | 0.528 |
| 3e-5 | 6 | 1.73 | 0.532 ± 0.052 | 0.778 |

- At `lr <= 3e-4` shuffling raises order stability from about 0.53 to 0.78-0.86 with no change in accuracy. At 1e-3 training is unstable and the comparison is not meaningful.
- Why 1e-3 fails: the vectors have norm about 110 and AdamW moves every weight by roughly `lr` per step, so early scores swing by whole units per step.

### 2.4 Final notebook 6 settings (notebook, lr 1e-4, 40 epochs, seed 0)
- Epoch-0 loss 1.76, then 0.0087 (epoch 10), 0.0019 (20), 0.0006 (30), 0.0003 (39). Training time 3.8 s.
- Train accuracy 1.000 and stability 1.000 (all types).
- Held-out accuracy 0.486 (choice 0.542, noul 0.417, score 0.500), order stability 0.667.
- Shuffling experiment (3 seeds): 1 order gives accuracy 0.500 (sd 0.030) and stability 0.528 (sd 0.039). 6 orders give accuracy 0.509 (sd 0.017) and stability 0.806 (sd 0.104).

### 2.5 Learning curve (prototype, 5 seeds, lr 1e-4, 40 epochs)
| training records | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| examples | 18 | 36 | 72 | 108 | 144 |
| train accuracy | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| held-out accuracy | 0.444 ± 0.034 | 0.481 ± 0.075 | 0.472 ± 0.028 | 0.617 ± 0.047 | 0.536 ± 0.047 |

Training accuracy is already 1.000 with a single record, so it carries no information about generalisation. The trend is upward but not monotone, and the spread is large.

### 2.6 Regularisation at 8 records (prototype, 5 seeds)
| setting | train accuracy | held-out accuracy |
|---|---|---|
| baseline (`d_head`=256, weight decay 0.01, 40 epochs) | 1.000 | 0.536 ± 0.047 |
| weight decay 1.0 | 0.997 | 0.497 ± 0.022 |
| `d_head`=8 | 1.000 | 0.547 ± 0.080 |
| early stop, 3 epochs | 0.924 | 0.519 ± 0.055 |
| early stop, 1 epoch | 0.747 | 0.453 ± 0.029 |

No regularisation moved held-out accuracy beyond the noise.

### 2.7 Conclusions from notebook 6
- Training accuracy 1.0 and loss near 0 showed only that the head can memorise 24 distinct facts with 524k parameters in 1024 dimensions.
- Held-out accuracy of about 0.5 against chance 0.39 is weak and rests on 12 independent questions. The honest reading is "more data needed", which notebook 7 tests.

---

## 3. Notebook 7: 800 synthetic records

### 3.1 Data
Generated by any2jev's synthetic ticket generator in a throwaway environment (seed 0, 3,000 records, split 80/10/10), then converted to our schema by `jev/data.py`. Raw files are in `data/synthetic_raw/`, converted files in `data/synthetic/`.

- Split: 2,400 train, 300 val, 300 test records. Notebook 7 uses the first 800 train records and the first 300 val records.
- Conversion check: 0 label mismatches across all 3,000 records when each converted label was compared with the original answer key.
- Train set composition: 7,281 questions (choice 3,625, score 1,805, noul 1,851). Questions per record: 1 -> 99, 2 -> 489, 3 -> 1,044, 4 -> 768. Options per choice question: 3 -> 1,817, 4 -> 1,558, 5 -> 250.
- Label counts in train: department technical 416, billing 411, returns 376, shipping 384, other 139, not_listed 44, none 38. Urgency: can wait 654, this week 611, today 540. Refund: no 1,286, yes 565. Tone: calm 919, frustrated 558, angry 340.
- Rendered length per question (first 400 records): minimum 31 tokens, median 52, maximum 106.
- Examples used: train 14,520 (800 records x 6 orders), val 3,668 (300 x 4 orders), test 3,628 (300 x 4 orders). Chance accuracy is 0.352 on both val and test.
- The text is template-based and the labels are encoded in keywords, so every accuracy below should be read as "the pipeline works", not "real tickets would score this".

### 3.2 Feature extraction (notebook)
- Batched extraction (64 questions per batch, sorted by length, right padding): train 112 s (7.7 ms per example), val 30 s. One at a time was about 26 ms per example, so about 3.3x faster.
- Batched versus one-at-a-time vectors: minimum cosine 0.99928 (decide vectors) and 0.99975 (option vectors), and 0 of 24 rows bit-identical. This is bf16 rounding that depends on batch shape. Padding slots are exactly zero.
- Cached features: 223 MB in bf16 on the GPU. The most options per question is 5.
- Masked loss check on a batch of 32: masked batch loss 1.138307, one example at a time 1.138307, and 1.754511 if the mask is forgotten.

### 3.3 First run (notebook, one-cycle schedule, lr 1e-3, 30 epochs, batch 64, seed 0)
Training time 5.5 s.

| epoch | 0 | 3 | 6 | 9 | 12 | 15 | 18 | 21 | 24 | 27 | 29 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| train loss | 0.9299 | 0.2876 | 0.1239 | 0.0644 | 0.0302 | 0.0110 | 0.0058 | 0.0042 | 0.0032 | 0.0028 | 0.0028 |
| val loss | 0.7471 | 0.3454 | 0.2347 | 0.1938 | 0.1636 | 0.1475 | 0.1289 | 0.1227 | 0.1275 | 0.1267 | 0.1268 |
| val accuracy | 0.680 | 0.862 | 0.917 | 0.930 | 0.945 | 0.955 | 0.957 | 0.960 | 0.959 | 0.959 | 0.959 |

Final validation: accuracy 0.959, loss 0.127, order stability 0.894. By question: department 0.864, refund 0.980, tone 0.998, urgency 0.996. The validation loss stops improving after about epoch 21 while the training loss keeps falling, so the head still memorises training tickets to some extent.

### 3.4 Experiment A: learning-rate schedule and input normalisation (notebook, 5 seeds)
| setting | val accuracy | worst seed | val loss |
|---|---|---|---|
| constant lr 1e-3, 12 epochs | 0.921 ± 0.038 | 0.847 | 0.332 |
| one-cycle lr 1e-3, 30 epochs | 0.960 ± 0.001 | 0.959 | 0.126 |
| one-cycle lr 1e-3, 30 epochs, no input normalisation | 0.959 ± 0.001 | 0.957 | 0.208 |

- The schedule removes the seed-to-seed spread and adds four points. Normalisation does not change accuracy once a schedule is used, but the loss is 0.208 instead of 0.126, so its probabilities are worse. This matters for calibration.

Extended schedule comparison (prototype, 5 seeds):

| setting | val accuracy | worst seed | val loss | order stability |
|---|---|---|---|---|
| constant lr 1e-3, 12 epochs | 0.921 ± 0.038 | 0.847 | 0.332 | 0.877 |
| one-cycle 1e-3, 15 epochs | 0.949 ± 0.001 | 0.948 | 0.147 | 0.893 |
| one-cycle 3e-3, 15 epochs | 0.956 ± 0.002 | 0.952 | 0.140 | 0.894 |
| one-cycle 1e-3 + grad clip 1.0, 15 epochs | 0.952 ± 0.002 | 0.950 | 0.142 | 0.893 |
| one-cycle 1e-3, 30 epochs | 0.960 ± 0.001 | 0.959 | 0.126 | 0.900 |

Earlier single-seed, 8-epoch, constant-lr comparison (prototype). Normalisation was needed for stability at high learning rates before a schedule existed:

| normalise | lr | epoch-0 train loss | final train loss | final val loss | final val acc | best val acc |
|---|---|---|---|---|---|---|
| yes | 1e-4 | 0.906 | 0.3374 | 0.4577 | 0.832 | 0.832 |
| yes | 3e-4 | 0.803 | 0.1481 | 0.3093 | 0.894 | 0.913 |
| yes | 1e-3 | 0.794 | 0.1528 | 0.2298 | 0.915 | 0.926 |
| no | 1e-4 | 0.822 | 0.1378 | 0.2854 | 0.910 | 0.910 |
| no | 3e-4 | 0.858 | 0.1070 | 0.5622 | 0.845 | 0.925 |
| no | 1e-3 | 1.173 | 0.4342 | 1.0531 | 0.873 | 0.873 |

Constant-lr epoch curve (prototype, val accuracy averaged over 3 seeds, losses from seed 0). It shows the noise the schedule removes:

| epoch | 0 | 1 | 2 | 3 | 5 | 8 | 12 | 16 | 20 | 25 | 29 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| train loss | 0.7944 | 0.4097 | 0.2729 | 0.2036 | 0.1222 | 0.0722 | 0.1626 | 0.0295 | 0.0042 | 0.2621 | 0.0027 |
| val loss | 0.5450 | 0.3798 | 0.3178 | 0.2277 | 0.2180 | 0.2030 | 0.1790 | 0.2042 | 0.1215 | 1.3540 | 0.1464 |
| val acc | 0.736 | 0.855 | 0.885 | 0.891 | 0.914 | 0.918 | 0.945 | 0.858 | 0.956 | 0.917 | 0.953 |

Val accuracy dipped to 0.858 at epoch 16 and the seed-0 val loss spiked to 1.354 at epoch 25. Best epoch by mean val accuracy was 24 (0.962) and by val loss was 21 (0.137).

### 3.5 Experiment B: learning curve (notebook, 3 seeds, one-cycle, 15 epochs)
| training records | 25 | 50 | 100 | 200 | 400 | 800 |
|---|---|---|---|---|---|---|
| examples | 414 | 846 | 1,758 | 3,564 | 7,272 | 14,520 |
| val accuracy | 0.563 ± 0.006 | 0.656 ± 0.002 | 0.761 ± 0.002 | 0.835 ± 0.000 | 0.919 ± 0.002 | 0.949 ± 0.001 |
| order stability | 0.650 | 0.651 | 0.736 | 0.759 | 0.832 | 0.892 |

- Gain per doubling: +0.093, +0.105, +0.074, +0.084, +0.030. The curve is slowing but has not flattened, and 2,400 training records exist.
- Earlier constant-lr version (prototype, 12 epochs): 0.558, 0.649, 0.741, 0.834, 0.902, 0.936 with stability 0.594, 0.624, 0.711, 0.751, 0.826, 0.880. Same shape, lower by about 0.01-0.02 at the top.

### 3.6 Experiment C: shuffling the options (notebook, 3 seeds)
| training | epochs | val accuracy | order stability |
|---|---|---|---|
| original order only | 90 | 0.887 | 0.726 |
| 6 orders | 15 | 0.949 | 0.892 |

Shuffling adds about six points of accuracy and seventeen of order stability. Part of the accuracy gain is data augmentation. The earlier constant-lr prototype gave 0.892 / 0.730 for one order and 0.936 / 0.880 for six.

### 3.7 Error analysis on validation (notebook, final model, `department` question)
| true answer | n | error rate | most common mistakes |
|---|---|---|---|
| returns | 204 | 22.1% | shipping 18, technical 17, billing 7 |
| other | 56 | 21.4% | returns 7, billing 2, technical 2 |
| billing | 208 | 11.5% | shipping 9, technical 6, not_listed 3 |
| shipping | 180 | 11.1% | technical 7, billing 5, not_listed 4 |
| not_listed | 40 | 10.0% | returns 4 |
| technical | 232 | 9.5% | billing 6, other 5, returns 4 |
| none | 20 | 5.0% | returns 1 |

`n` counts examples, and each ticket appears in 4 option orders. Per ticket: 66 of 235 tickets have at least one wrong order and only 4 are wrong in every order, so much of the error is order sensitivity. `other`, `none` and `not_listed` are the "none of the above" answers, and only the bare `other` is hard.

The ten worst tickets (wrong in 3 or 4 of 4 orders), abbreviated:

| wrong orders | true answer | text |
|---|---|---|
| 4/4 | shipping | "The tracking number shows no updates for a week. I want my m[oney back]..." |
| 4/4 | billing | "I need a proper receipt for my purchase. I need this fixed toda[y]..." |
| 4/4 | returns | "The screen was cracked when I opened it. ASAP please, my whole team is blocked." |
| 4/4 | returns | "The screen was cracked when I opened it. I need this fixed today." |
| 3/4 | returns | "The screen was cracked when I opened it. Whenever you get a chance." |
| 3/4 | returns | "The item came broken in the box. No rush on this." |
| 3/4 | technical | "Two-factor codes never arrive. I would like a refund." |
| 3/4 | shipping | "I expected better. My package has not arrived yet." |
| 3/4 | other | "The courier left my parcel at the wrong address." (shipping was removed from the options) |
| 3/4 | returns | "The screen was cracked when I opened it. Please refund me." |

- The cracked-screen phrase accounts for 4 of the 10 worst tickets, and its `returns` label is debatable (technical is plausible).
- When the true department is removed from the options, the model picks the nearest remaining one instead of `other`. For example, a misdelivered parcel with `shipping` missing went to `returns`.
- This error pattern depends on training quality. With the noisy constant learning rate (prototype), the bare `other` answer was wrong far more often (24 of 56) and urgency accuracy was 0.885. Both improved a lot with the schedule.

### 3.8 Test set, used once (notebook, same final model, seed 0)
| split | accuracy | loss | order stability | department | refund | tone | urgency |
|---|---|---|---|---|---|---|---|
| val | 0.959 | 0.127 | 0.894 | 0.864 | 0.980 | 0.998 | 0.996 |
| test | 0.959 | 0.131 | 0.900 | 0.874 | 0.990 | 0.982 | 0.991 |

Test chance accuracy is 0.352. Test and validation agree closely, which suggests the choices made on validation did not overfit it.

Earlier constant-lr model (prototype, 12 epochs, seed 0): val 0.920 (loss 0.258, stability 0.875), test 0.926 (loss 0.250, stability 0.882). Department 0.830 / 0.841, urgency 0.885 / 0.899.

---

## 4. Summary of findings
1. **Data was the bottleneck.** Held-out accuracy went from about 0.5 at 8 records (chance 0.39) to 0.96 at 800 records (chance 0.35), and the learning curve was still rising.
2. **Training stability mattered as much as the model.** A constant learning rate gave a ±0.038 spread across seeds, and a one-cycle schedule reduced it to ±0.001 while raising accuracy from 0.921 to 0.960.
3. **Shuffling options with moved labels works.** At 800 records it adds about six points of accuracy and seventeen of order stability.
4. **Regularisation tricks did not help at 8 records.** The problem there was too little information, not too much capacity.
5. **Input normalisation** lets high learning rates train without a schedule, and with a schedule it mainly improves the loss (0.126 vs 0.208), not accuracy.
6. **Remaining errors** are debatable template labels, order sensitivity, and the model choosing the nearest department when the right one has been removed.

## 5. Caveats
- The synthetic data has 8 phrases per department and fixed urgency and tone phrases. A near-perfect urgency score here is expected and says little about real text.
- Several results come from 3 seeds and the 12-question smoke held-out set is very small. Treat small differences there as noise.
- Prototype numbers were produced by the scratch scripts archived in [experiments/archive/as_run/](../experiments/archive/as_run/ALL_SCRIPTS.md). They agree with the notebooks in direction, and the notebooks' numbers take precedence where both exist.

## 6. Not yet measured
- Real-text test with the public datasets (boolq, ag_news, banking77, sst5).
- The full 2,400-record training set, and the cache-reuse optimisation in the serving path.

Calibration (expected calibration error, temperature scaling) and LoRA were measured later and are reported in [RESULTS_LORA.md](RESULTS_LORA.md). Images are reported in [RESULTS_VISION.md](RESULTS_VISION.md).
