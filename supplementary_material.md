# Build a Jev-Style Decision Model from an LLM: Supplement and Reference

Oct 15, 2026 · @Divye Singh

*A lookup page for the series. It repeats no story. Use it to find a recipe, a result or a term fast.*

## Every best recipe in one place

A **result** is a 3-seed average on the test set. The one exception is the frozen-head notebook run, which is a single seed used once. **Best observed, not confirmed** is the winner of single-seed or validation-only runs. Chance is 0.352 on the text tickets and 0.337 on the shapes.

### Tiny 12-ticket set (Part 3, best observed, not confirmed)

| Hyperparameter | Value |
| --- | --- |
| Training tickets | 8, split by ticket |
| Option orders per question | 6 |
| Head size and optimizer | 256, AdamW with weight decay 0.01 |
| Learning rate and epochs | 1e-4 and 40 |
| Held-out accuracy | 0.536 +- 0.047 over 5 seeds (chance 0.389) |
| Best single setting seen | Learning rate 3e-4, 0.579 +- 0.105 over 3 seeds |
| Order stability | 0.806 +- 0.104 over 3 seeds at learning rate 1e-4 (0.861 at 3e-4). It counts score questions, so it is not comparable with the 800-ticket numbers. |

### Text tickets, frozen backbone (Part 3)

| Hyperparameter | Value |
| --- | --- |
| Training tickets | 800 (14,520 examples) |
| Option orders per question | 6, shuffled with the label moved |
| Head size | 256 (524,288 trainable parameters) |
| Input normalization | RMS, applied before the projections |
| Learning-rate schedule | One-cycle, peak 1e-3, 681 warm-up steps out of 6,810 |
| Epochs and batch size | 30 and 64 |
| Gradient clipping | Not used |
| Calibration | Temperature scaling, T fitted on validation by minimising NLL, clamped to 0.05 to 20 |

| Measure | Notebook run: validation (5 seeds) and test (seed 0, once) | Result: test, mean of 3 seeds, retrained | With temperature scaling (T 1.41) |
| --- | --- | --- | --- |
| Accuracy | 0.960 +- 0.001 and 0.959 | 0.963 +- 0.005 | 0.963 |
| Order stability | 0.900 and 0.900 | 0.912 +- 0.012 | 0.912 |
| Loss (NLL) | 0.126 and 0.131 | 0.120 +- 0.008 | 0.110 +- 0.005 |
| Brier | not recorded | 0.056 +- 0.004 | 0.055 +- 0.003 |
| ECE | not recorded | 0.013 +- 0.004 | 0.009 +- 0.003 |

### LoRA on text (Part 4)

| Hyperparameter | Value |
| --- | --- |
| LoRA rank and alpha | 8 and 16 |
| LoRA target | Language layers only. The vision tower stays frozen. |
| LoRA learning rate and head learning rate | 2e-4 and 5e-4 |
| Head initialization | Warm start from the frozen head |
| Frozen head training | 30 epochs, one-cycle schedule, peak 1e-3, 6 orders |
| LoRA epochs and batch size | 2 and 16 |
| Training tickets | 800 |
| Loss | Cross-entropy + 1.0 x Brier |
| Trainable parameters, time, memory | 5,111,808, 272 seconds, 2.73 GB |

| Test, mean of 3 seeds | Frozen head | LoRA, cross-entropy | LoRA, with Brier (best) |
| --- | --- | --- | --- |
| Accuracy | 0.963 | 0.992 | 0.995 |
| Order stability | 0.912 | 0.977 | 0.981 |
| NLL | 0.120 | 0.028 | 0.018 |
| Brier | 0.056 | 0.013 | 0.009 |
| ECE | 0.013 | 0.004 | 0.003 |

### LoRA on shapes images (Part 4)

The same recipe as text, except the frozen head sees 4 option orders. Training takes 436 seconds and 2.84 GB.

| Test, mean of 3 seeds | Frozen head | LoRA, cross-entropy | LoRA, with Brier (best) |
| --- | --- | --- | --- |
| Accuracy | 0.946 | 0.970 | 0.972 |
| Order stability | 0.924 | 0.977 | 0.970 |
| NLL | 0.186 | 0.084 | 0.075 |
| Brier | 0.086 | 0.046 | 0.042 |
| ECE | 0.033 | 0.010 | 0.008 |

### Best observed, not confirmed

| Setting | Data | Single-seed validation result | Why it is not confirmed |
| --- | --- | --- | --- |
| LoRA learning rate 5e-4 | Text | Accuracy 0.999, Brier 0.003 | One seed, no test, found on validation |
| LoRA rank 16 | Text | Accuracy 0.997, Brier 0.004 | One seed, no test |
| LoRA learning rate 5e-4 | Shapes | Accuracy 0.981, Brier 0.027 | One seed, no test |
| LoRA rank 16 | Shapes | Accuracy 0.980, Brier 0.027 | One seed, no test |
| 4 epochs | Shapes | Accuracy 0.982, stability 0.987, Brier 0.023 | One seed, no test, twice the time |

## Glossary

| Term | Plain meaning | First in |
| --- | --- | --- |
| Token | A piece of text or an image patch group that the model reads as one unit. | Part 2 |
| Vector | A list of numbers. Ours have 1,024 numbers. | Part 2 |
| Parameter | One learned number inside a model. | Part 2 |
| Backbone | The pretrained language model underneath the head. | Part 3 |
| Frozen | Never changed by our training. | Part 3 |
| Logit | A raw score before the softmax turns scores into probabilities. | Part 1 |
| Softmax | Raises e (2.718) to each score and shares the total out, so the shares add up to 1. | Part 2 |
| bf16 | A 16-bit number format that keeps about three digits. Fast and small, but it rounds. | Part 2 |
| fp32 | A 32-bit number format that keeps about seven digits. | Part 2 |
| Attention shares | How much of its attention a token gives to each earlier token. | Part 2 |
| KV cache | Stored keys and values of earlier tokens, so a model need not recompute them. | Part 2 |
| Memory layer (DeltaNet) | A layer that keeps one fixed-size memory instead of a list of earlier tokens. | Part 2 |
| Marker token | A special one-token symbol at a fixed place in the sequence. We read a vector there. | Part 2 |
| Decide token | The final marker. Its vector stands for the whole question. | Part 2 |
| Choice, noul, score | The three question types. Choice picks one named option. Noul is yes or no. Score places an answer on an ordered scale. | Part 2 |
| Length of a vector | How big it is. The square root of the sum of its squared numbers. | Part 3 |
| Dot product | Multiply matching numbers of two vectors and add them up. | Part 3 |
| Cosine | Compares the direction of two vectors. 1 is the same direction and 0 is unrelated. | Part 3 |
| Projection | A learned table of numbers that turns a long vector into a shorter one. | Part 3 |
| Pointer head | A small trained scorer that rates each option's own vector against the decide vector. | Part 3 |
| Padding and mask | Empty option slots added to fill a batch and the step that stops them taking part in the softmax. | Part 3 |
| Option order | The order in which options are listed. It changes the vectors because the model reads left to right. | Part 3 |
| Order stability | Among choice and noul questions, the share that get the same option text in every order. | Part 3 |
| Chance accuracy | The accuracy of random guessing. The average of 1 over the number of options. | Part 3 |
| Held-out data | Tickets never seen in training. | Part 3 |
| Validation and test | Validation guides choices. Test is used once at the end for the honest number. | Part 3 |
| Synthetic data | Tickets made by a program from templates, with labels correct by construction. | Part 3 |
| Seed | A number that fixes the random choices of a run. Several seeds show how much is luck. | Part 3 |
| Epoch | One pass over the training data. | Part 3 |
| Learning rate | The step size of each update. | Part 3 |
| One-cycle schedule | The learning rate climbs to a peak and then falls smoothly to near zero. | Part 3 |
| Weight decay | A small pull of every trainable number toward zero. | Part 3 |
| Gradient clipping | A cap on the size of one update. | Part 3 |
| Regularization | Any restriction that stops a model from memorising. | Part 3 |
| RMS normalization | Dividing a vector by the square root of the average of its squared numbers. | Part 3 |
| NLL | The average of minus the log of the probability given to the true option. | Part 3 |
| Brier score | The squared error of the probabilities against the truth. 0 is perfect and 2 is worst. | Part 3 |
| ECE | The weighted average gap between stated confidence and achieved accuracy. | Part 3 |
| Calibration | Whether a stated probability matches how often the answer is right. | Part 3 |
| Temperature (T) | One number that divides all scores before the softmax. Above 1 flattens and below 1 sharpens. | Part 3 |
| NaN | Not a number. One NaN poisons a whole batch. | Part 3 |
| LoRA | Learning a small low-rank correction to a frozen matrix instead of changing the matrix. | Part 4 |
| Rank and alpha | Rank is how thin the correction matrices are. Alpha sets the scale of the correction. | Part 4 |
| Gradient checkpointing | Storing only layer inputs and recomputing the insides to save memory. | Part 4 |
| Patch | A 16 by 16 pixel square of an image. | Part 4 |
| Vision tower | The separate network that turns patches into feature vectors. | Part 4 |
| Swapped-image control | Give each record another record's image and keep its labels. Accuracy should fall to chance. | Part 4 |

## Mistakes we made and what each one taught us

| Part | What went wrong | What it taught us |
| --- | --- | --- |
| 2 | We suspected a bug in cache reuse. The vectors differed by 0.375. | It was bf16 rounding. At magnitude 32 to 64 one rounding step is 0.25. Compare in fp32 before declaring a bug. |
| 2 | We assumed that turning special tokens off stopped a user from forging a marker. | Fast tokenizers still recognise registered special-token strings in raw text. Sanitize before tokenizing. |
| 2 | We trusted an attribute name, a layer path and a mask parameter. | Each behaved differently on this model. Inspect the real object. |
| 2 | We assumed we could use a block-causal mask. | The recurrent layers keep one blended memory, so you cannot mask tokens out of it. |
| 3 | We expected a scaled dot product to give usable scores. | The vectors were nearly parallel and 110 to 125 units long. The softmax put 1.0 on one option. Look at raw vectors first. |
| 3 | We could have forgotten to mask the padded slots. | The loss is wrong but finite. Batched loss was 1.754511 against 1.138307 one question at a time. Test against a one-at-a-time loss. |
| 3 | A shuffle without moving the label trains on wrong answers. | Nothing crashes. Move the label with its option and check it. |
| 3 | Training accuracy was 1.000 with a single ticket. | Training accuracy on a tiny set carries no information. |
| 3 | The most-common-label baseline scored 0.333, below random guessing at 0.389. | Random guessing is the honest reference. |
| 3 | One seed at a learning rate of 1e-3 showed shuffling hurting stability. | The run was unstable. Average over seeds at a sane rate before reading an effect. |
| 3 | We predicted that all three "none of the above" answers would be hard. | Only `other` was hard. Read the actual errors. |
| 3 | Temperature scaling returned NaN on real data. | Minus infinity divided by a trainable T gives a NaN gradient. Replace padded scores with 0 first and test with padded tensors. |
| 3 | We compared order stability between two notebooks. | Notebook 6 counts score questions and notebook 7 excludes them. The numbers are not comparable. |
| 4 | LoRA training crashed with a cache of length 154 where 77 was expected. | Recomputation in checkpointing appended to the KV cache twice. Turn the cache off. Outputs stay bit-identical. |
| 4 | A batch of 8 failed to allocate memory. | Each sequence needs about 0.55 GB of activations. Gradient checkpointing brought a batch of 4 down to 2.0 GB in total. |
| 4 | The extraction time in the cost table understated the real cost. | Most runs reused cached features. Real extraction took about 180 seconds for text and 409 seconds for the first image run. |
| 4 | A single-seed ablation looked better than the default. | Do not claim it. Run three seeds and score on test first. |

## Takeaways and questions, part by part

The parts do not carry their own takeaway and question sections, so they are collected here. Part 1 is not repeated because it is already published.

### Part 2: Opening the Model

**What to take away**

- Inspect the real object before trusting an API, a tutorial or an attribute name. A tokenizer attribute, a layer path and a mask parameter all behaved differently from what we assumed.
- Hybrid models break assumptions that hold for pure transformers. You cannot mask tokens out of a recurrent state.
- Compare in fp32 before declaring a bug in bf16 code and think in units of the number format's rounding step.
- A marker must be exactly one token and user text must never be able to produce one. Assert both.

**Questions that came up**

| Question | Short answer |
| --- | --- |
| Where did the output head go when I counted parameters? | It shares its weights with the input embedding. Parameter listing reports a shared tensor once. |
| Why does a model with 24 layers return 25 hidden states? | The first is the embedding before any layer ran. |
| What do 6144 and 4 in the cache mean? | 6144 is the width of the query, key and value projections together (three of 2,048). 4 is the width of the short convolution in front of the recurrence. |
| Why can't we use a block-causal mask on this model? | The recurrent layers keep one blended memory, not a list of per-token entries. Their mask only blanks out padding. |
| Is a difference of 0.375 in a hidden vector large? | It depends on scale and number format. At magnitude 30 in bf16 the rounding step is 0.25, so it is about one and a half steps. Re-run in fp32 to be sure. |
| Why marker tokens and why one token each? | They give fixed places where option and question vectors are read. A marker that splits has no single token to read. |
| Why can a user forge a marker if special tokens are off? | The flag does not stop fast tokenizers from recognising registered special-token strings in raw text. Sanitize before tokenizing. |

### Part 3: Training the Jev Head, From Vectors to Probabilities You Can Trust

**What to take away**

- Look at the raw vectors before you design the head. Ours were nearly parallel and 110 to 125 units long. A scaled dot product would have put all of the probability on one option.
- A pointer head scores any number of options with one small set of numbers. Mask the padded slots and test the batched loss against a one-at-a-time loss.
- Shuffle the option orders in training and move the label with each option. Then measure order stability, because accuracy cannot see this problem.
- A training accuracy of 1.000 on a tiny set means nothing. Report seeds. A one-cycle learning-rate schedule cut the spread across seeds by a factor of 38.
- Accuracy does not say whether the probabilities are honest. Report NLL, Brier and ECE. Fit the temperature on validation and judge it on test.

**Questions that came up**

| Question | Short answer |
| --- | --- |
| Why do the raw dot products come out in the thousands? | The vectors are 110 to 125 units long and nearly parallel. Length times length times cosine is a big number whatever the model thinks. |
| Why not use a normal classifier with 3 outputs? | Options change with every question in number and in meaning. The pointer head scores each option from its own vector. |
| What does the padding mask do? | It sets the scores of empty slots to minus infinity so they get no share in the softmax. Without it the loss was 1.754511 instead of 1.138307 on a real batch. |
| Why is order stability lower than accuracy? | Stability asks for the same answer in every order. One wrong order breaks it. Accuracy counts each order on its own. |
| Is 0.959 accuracy impressive? | It shows the pipeline works. The tickets are templated and the labels sit in keywords, so it says little about real text. |
| Why was the test set used only once? | A test set that guides choices stops being a fair check. It agreed with validation, which is the evidence that tuning did not overfit. |
| Does temperature scaling change which option wins? | No. It divides every score by the same number, so the order of the scores stays the same. |
| Did adding Brier to the loss help? | Not for the frozen head. Part 4 shows a small gain for the adapted backbone. |

### Part 4: Letting the Backbone Learn, Then Giving It Eyes

**What to take away**

A frozen model plus a small pointer head reads a decision off one forward pass. LoRA lifts the accuracy from 0.963 to 0.995 on text. The same machinery reads images and the controls show that it looks at them. The takeaways below close the whole series. The engineering lessons from Parts 2 and 3 are listed above and are not repeated.

For business readers:

- Decision models are cheap and fast. One forward pass replaces a generated paragraph. The output is always a valid choice with a probability.
- The data needed is modest but it must be varied. Accuracy rose with every doubling of data. Tiny datasets produce numbers that look perfect and mean nothing. Insist on results measured on tickets the model has never seen.
- Probabilities have to be earned. Accuracy does not tell you whether "0.9 sure" is trustworthy. Ask for calibration measures such as the Brier score and the calibration error. Ask for a held-out test too. Our auto-close example showed how a miscalibrated model hides 200 extra wrong closures in every 1,000 tickets.
- Synthetic data is a smoke test and not a benchmark. If the labels are encoded in keywords, high accuracy proves little about real traffic.
- Check that the model uses the evidence. For image tasks the swapped-image test is a cheap and decisive question to ask any vendor or team.

For engineers:

- Check baselines such as random guessing. Report seeds. Split by the unit that shares information, which is the ticket and not the question.
- Write the unit test with data that looks like production. Our temperature-scaling bug needed minus-infinity padding to appear.
- Features can be cached while the backbone is frozen. That stops being true with LoRA, so plan the compute for it.

**Questions that came up**

| Question | Short answer |
| --- | --- |
| Why does LoRA start as a no-op? | One of the two small matrices starts at zero, so the correction is exactly zero until training moves it. |
| Why can't we cache features when using LoRA? | The output of the backbone changes at every step, so each epoch needs a fresh full pass. |
| Does LoRA always beat the frozen head? | No. With 100 or 200 training tickets it was worse. The crossover lies between 200 and 400 tickets. |
| What did LoRA actually fix? | Mostly the order sensitivity on the department question. Tickets wrong in at least one order fell from 64 of 235 to 11 of 235. |
| Why does the Brier term help LoRA but not the frozen head? | The frozen head has few parameters and its errors come from features it cannot change. The loss makes no difference there. The gain for LoRA is small. |
| How many tokens does an image cost? | Patches across times patches down, divided by 4 for the 2 by 2 merge. A 256 by 256 image costs 64 tokens and the processor will not go below 256 by 256. |
| How do we know the model really looks at the image? | Swap images between records and keep the labels. Accuracy fell from 0.972 to 0.342 for the adapted model. That is chance. |
| Why is the triangle question near 0.5 under the controls? | The answer is yes more often than no, so a model that ignores the image still scores above chance on it. |
| Can I use my own photos? | The pipeline needs no code change. An image part is a file path. You need records with labels and you should expect to retrain. |

## What has not been measured and where the full tables live

The data is synthetic, so every number shows that the pipeline works. It does not show that the model handles real tickets or photographs.

- A real-text test on public datasets such as boolq, ag\_news, banking77 and sst5.
- Real photographs and adapters inside the vision tower.
- Three seeds and a test score for the rank, learning-rate and epoch ablations.
- The full 2,400-ticket training set. The frozen-head learning curve was still rising at 800.
- LoRA data size separated from step count.
- A speed benchmark for cache reuse in a serving setting.

The full tables sit in three result files in the project's GitHub repository, in the results folder next to the jev package that runs everything:

| File | What it holds |
| --- | --- |
| [`RESULTS.md`](https://github.com/sdivye92/create_jev/blob/main/results/RESULTS.md) | The frozen head. The 12-ticket runs, the learning-rate sweep, the 800-ticket experiments, the error analysis and the one-time test. |
| [`RESULTS_LORA.md`](https://github.com/sdivye92/create_jev/blob/main/results/RESULTS_LORA.md) | LoRA on text. The main comparison on validation and test, cost, per-epoch accuracy, ablations, learning curve, errors and the reliability table. |
| [`RESULTS_VISION.md`](https://github.com/sdivye92/create_jev/blob/main/results/RESULTS_VISION.md) | LoRA on shapes images. The same tables plus the swapped-image and no-image controls. |
| [`jev/`](https://github.com/sdivye92/create_jev/tree/main/jev) | The package: rendering, feature caching, pointer head, LoRA, calibration and the command line. |

To rebuild the LoRA and vision files from the raw runs: `experiments/suite.sh lora`, `experiments/suite.sh vision` and `python experiments/make_reports.py`.
