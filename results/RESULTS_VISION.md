# Results: vision (images as input)

Measured on the synthetic shapes images, with a Qwen3.5-0.8B backbone and a pointer head. 14 runs, generated from `experiments/results/vision/*.json` by `experiments/make_reports.py`. Each run first trains a head on the frozen backbone (the starting point), then trains LoRA adapters on the language layers together with the head. Both stages are reported from the same run, so comparisons are paired.

## Best configuration

**Recommended recipe for the shapes images**, the `main_brier` runs. It is the same as for text:

| Hyperparameter | Value |
|---|---|
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA target | Language layers only (vision tower frozen) |
| LoRA learning rate | 2e-4 |
| Head learning rate | 5e-4 |
| Head initialisation | Warm start from the frozen head |
| Frozen head training | 30 epochs, one-cycle schedule, peak 1e-3 |
| Epochs | 2 |
| Batch size | 16 |
| Training records | 800 |
| Option orders per question (frozen head) | 4, shuffled with the label moved |
| Loss | Cross-entropy + 1.0 x Brier (Brier weight 1) |

Results are the mean of 3 seeds. The default recipe with a Brier weight of 0 (cross-entropy only) and the frozen head are shown for comparison. Chance is 0.337.

| Test set, mean of 3 seeds | Frozen head | LoRA, Brier weight 0 (cross-entropy only) | LoRA, Brier weight 1 (best) |
|---|---|---|---|
| Accuracy | 0.946 | 0.970 | 0.972 |
| Order stability | 0.924 | 0.977 | 0.970 |
| NLL | 0.186 | 0.084 | 0.075 |
| Brier | 0.086 | 0.046 | 0.042 |
| ECE | 0.033 | 0.010 | 0.008 |

On validation the gap between the two LoRA variants is larger: accuracy 0.980 against 0.972 and Brier 0.032 against 0.040, mostly from the leftmost question (0.974 against 0.944). Order stability is the one metric where the Brier variant is not better on test (0.970 against 0.977), and the difference is within the seed spread.

**Why this one, and why the other settings lost:**

| Setting | What happened | Why it probably worked or did not |
|---|---|---|
| LoRA instead of a frozen backbone | Test accuracy 0.946 to 0.970, order stability 0.924 to 0.977, Brier 0.086 to 0.046. The swapped-image and no-image controls stay at chance (about 0.34) for both, so the gains come from the picture. | The frozen backbone exposes the image content only as a small residual in the decide vector, which the head finds for easy questions (counting, triangle) but not for spatial ones. LoRA makes it easier to read, and the gain is mostly on leftmost and largest_color. |
| Brier weight 1 instead of 0 | Better on validation (Brier 0.032 against 0.040) and slightly better on test (Brier 0.042 against 0.046, ECE 0.008 against 0.010). | The Brier term trains on the whole probability vector and is bounded, so it improves probability quality directly. It helped more here than on text, probably because the vision task has more room left (the starting Brier is higher). The gain is small and the dataset is synthetic. |
| Brier weight on the frozen head | No effect (validation accuracy 0.947 against 0.948). | The head has too few parameters for the loss to matter. |
| Rank 8 instead of 4 | Same accuracy (0.973), but rank 4 has lower order stability (0.963 versus 0.973). | Capacity matters mostly for the order robustness. |
| Rank 8 instead of 16 | Rank 16 slightly better (0.980 accuracy, Brier 0.027) with twice the parameters. | A little extra capacity helps. The difference is a single seed. |
| LoRA learning rate 2e-4 instead of 1e-4 or 5e-4 | 1e-4 was worse (0.964). 5e-4 was better (0.981). | Two epochs is short, so a higher rate moves the backbone further. |
| 2 epochs instead of 4 | 4 epochs gave the best single run on validation (0.982 accuracy, stability 0.987, Brier 0.023) for double the time (873 s against 435 s). | Unlike text, the vision model was still improving after two epochs (0.981 after epoch 2 and 0.983 after epoch 4 on the quick check), so it benefits from more training. |
| Training size | LoRA beats the frozen head at every size from 100 records up (0.932 against 0.912 at 100). | The frozen head already starts high, and the adapters need little data to improve the hard spatial questions. |

**What was not done:** 4 epochs, a learning rate of 5e-4 and rank 16 each looked slightly better than the default in one seed on validation, but none was run with three seeds or scored on the test set, so they are not claimed as the best. The Brier variant is the best configuration that has three seeds and test evidence.

## 0. Before the tables: what was run, how runs are named, and what the columns mean

### What one run is

A run is one complete experiment on the synthetic shapes images. It first trains a pointer head on top of the frozen backbone, which is the starting point and the baseline (called frozen). It then trains LoRA adapters on the language layers together with the head, and that is the adapted model (called LoRA). Both stages are evaluated on the same validation records in the same run, so every frozen versus LoRA comparison in this file is paired.

Training uses 800 records. Each record has several questions, and every question is shown to the model in 4 different random option orders during training, because the head must not depend on the order in which options are listed. Validation and test use 4 fixed option orders per question, and the same records and orders in every run, so numbers are comparable between runs.

### Run names

Every result file is named `<what changes>_seed<N>`. For example `main_brier_seed1` is the run with the Brier term in the loss, using random seed 1.

| Part of the name | Meaning |
|---|---|
| main_crossentropy | The standard recipe. The loss is cross-entropy only. This is the reference run that the others are compared with. |
| main_brier | The standard recipe with a Brier term added to the loss (loss = cross-entropy + 1.0 x Brier). It tests whether training directly on the Brier score improves probability quality. |
| rank4, rank16 | LoRA rank 4 or 16 instead of 8. The rank sets how many numbers each adapter has. The scale alpha is set to twice the rank, so the strength of the correction stays comparable. |
| lr1e-4, lr5e-4 | LoRA learning rate 0.0001 or 0.0005 instead of the default 0.0002. |
| epochs4 | 4 passes over the training data instead of 2. |
| size100, size200, size400 | Train on only the first 100, 200 or 400 of the 800 training records. Used for the learning curve. |
| seed0, seed1, seed2 | The random seed, explained next. |

### Seeds

Training involves random choices: the starting values of the head, the order in which batches are drawn, and which random option orders are generated for the training questions. A seed is a number that fixes all of those choices. Running the same settings with the same seed gives the same result (up to tiny GPU rounding differences), and running with a different seed repeats the experiment with different random luck.

Why bother: a single run can look good or bad by chance, and one number cannot tell you which. Running the three main configurations with seeds 0, 1 and 2 gives three results for the same recipe, and the spread between them (the number after +-) shows how much of any difference is just luck. A difference between two variants that is smaller than that spread should not be trusted. The same seed numbers are used for the cross-entropy and Brier runs, so `main_crossentropy_seed1` and `main_brier_seed1` start from the same random draw and differ only in the loss.

Seeds do not change the records, the validation and test sets, or the validation option orders. Those are fixed, so the +- values measure training randomness only, not the variation you would get from a different dataset. The ablations and the learning curve use seed 0 only, so each of those numbers is a single draw and differences of about 0.005 in accuracy are within the noise.

### What the columns mean

| Column | Meaning | Better is |
|---|---|---|
| Accuracy | Share of (question, option order) examples where the option with the highest probability is the true one. The same question in 4 orders counts as 4 examples. | higher |
| Chance | The accuracy of guessing. It is the average of 1 / (number of options) over the questions. | - |
| Order stability | Share of choice and yes/no questions for which the model picks the same option text in every option order. A model can be accurate on average and still change its answer when the options are reshuffled, and this column catches that. Score questions are never shuffled and are excluded. | higher |
| NLL | Negative log-likelihood: the average of -log(probability given to the true option). It punishes confident wrong answers heavily. This is also the training loss. | lower |
| Brier | Mean squared error between the probabilities and the truth (1 for the true option, 0 for the others), summed over the options. It is bounded (the worst possible value is 2) and reads as a plain error rate for probabilities. A model that spreads probability equally over N options scores 1 - 1/N. | lower |
| ECE | Expected calibration error: predictions are grouped by confidence, and ECE is the average gap between confidence and actual accuracy in each group, weighted by group size. 0 means that when the model says 80% it is right 80% of the time. | lower |
| T | The temperature: one number fitted on the validation set that divides all scores before the softmax. T above 1 flattens the probabilities, so the model was overconfident. T below 1 sharpens them, so it was underconfident. T never changes which option wins, so accuracy is unaffected. | closer to 1 means less correction needed |
| NLL (T), Brier (T), ECE (T) | The same three measures after applying the temperature T. They show how good the probabilities become once the cheap post-hoc fix is applied, and so how much of the calibration problem was left over. On validation these are slightly optimistic, because T was fitted on that same data. On test, T from validation is used, so they are honest. | lower |
| +- (after a value) | Population standard deviation over the seeds that were run. It is a rough guide to run-to-run noise, based on only three runs. | - |


## 1. Main comparison (validation)

**What:** the central experiment. Four variants are compared: the frozen backbone and the LoRA-adapted backbone, each trained with cross-entropy alone and with cross-entropy plus the Brier term. Each variant is run with 3 seeds.

**Why:** it answers the two main questions. Does adapting the backbone beat reading it frozen, and does adding Brier to the loss change the probability quality?

**How to read:** compare rows in the same column. Accuracy and order stability should go up, NLL, Brier and ECE should go down. Check that a difference between two rows is larger than the +- spread before believing it. The columns marked (T) show the same measures after temperature scaling, and T itself tells you how overconfident the raw probabilities were. Validation values are the ones used to choose settings, so they are for comparing variants, and the test table below is the one to quote.

| Variant | Seeds | Accuracy | Order stability | NLL | Brier | ECE | NLL (T) | Brier (T) | ECE (T) | T |
|---|---|---|---|---|---|---|---|---|---|---|
| Frozen backbone, cross-entropy | 3 | 0.948 +- 0.001 | 0.926 +- 0.003 | 0.174 +- 0.005 | 0.079 +- 0.002 | 0.028 +- 0.001 | 0.138 +- 0.002 | 0.074 +- 0.001 | 0.008 +- 0.002 | 1.87 +- 0.03 |
| LoRA, cross-entropy | 3 | 0.972 +- 0.003 | 0.969 +- 0.004 | 0.074 +- 0.009 | 0.040 +- 0.004 | 0.010 +- 0.001 | 0.071 +- 0.009 | 0.039 +- 0.004 | 0.006 +- 0.002 | 1.10 +- 0.22 |
| Frozen backbone, cross-entropy + Brier | 3 | 0.947 +- 0.002 | 0.925 +- 0.003 | 0.166 +- 0.005 | 0.078 +- 0.002 | 0.025 +- 0.002 | 0.139 +- 0.003 | 0.074 +- 0.002 | 0.010 +- 0.001 | 1.73 +- 0.03 |
| LoRA, cross-entropy + Brier | 3 | 0.980 +- 0.001 | 0.973 +- 0.005 | 0.061 +- 0.002 | 0.032 +- 0.001 | 0.006 +- 0.001 | 0.061 +- 0.002 | 0.032 +- 0.001 | 0.006 +- 0.001 | 1.08 +- 0.05 |

Chance accuracy on validation: 0.337 +- 0.000.

### Accuracy by question (validation)

**Why:** the overall accuracy averages all questions, and the questions differ a lot in difficulty. This table shows which questions the gain comes from and which are already solved. Each cell is the mean accuracy over the seeds for that question.

| Variant | largest_color | has_triangle | count | leftmost |
|---|---|---|---|---|
| Frozen backbone, cross-entropy | 0.935 +- 0.004 | 0.990 +- 0.000 | 0.980 +- 0.000 | 0.886 +- 0.001 |
| LoRA, cross-entropy | 0.956 +- 0.008 | 1.000 +- 0.000 | 0.990 +- 0.004 | 0.944 +- 0.005 |
| Frozen backbone, cross-entropy + Brier | 0.933 +- 0.008 | 0.991 +- 0.002 | 0.980 +- 0.000 | 0.885 +- 0.003 |
| LoRA, cross-entropy + Brier | 0.956 +- 0.003 | 1.000 +- 0.000 | 0.988 +- 0.002 | 0.974 +- 0.003 |

## 2. Test set (used once per run)

**What:** the same four variants on records that were never used for training, for choosing settings, or for fitting T.

**Why:** validation was looked at while making decisions, so it is slightly flattered. The test set is touched once per main run, after everything is decided, and is the number to quote as the expected performance. If the test numbers are close to the validation numbers, the choices did not overfit validation.

**How to read:** the columns are as before, with the per-question accuracy at the end. The (T) columns use the temperature fitted on validation, which is how it would be used in practice.

| Variant | Seeds | Accuracy | Order stability | NLL | Brier | ECE | NLL (T) | Brier (T) | ECE (T) | largest_color | has_triangle | count | leftmost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Frozen backbone, cross-entropy | 3 | 0.946 +- 0.000 | 0.924 +- 0.004 | 0.186 +- 0.009 | 0.086 +- 0.003 | 0.033 +- 0.002 | 0.142 +- 0.004 | 0.080 +- 0.002 | 0.008 +- 0.001 | 0.924 +- 0.002 | 1.000 +- 0.000 | 0.978 +- 0.002 | 0.880 +- 0.003 |
| LoRA, cross-entropy | 3 | 0.970 +- 0.003 | 0.977 +- 0.006 | 0.084 +- 0.003 | 0.046 +- 0.003 | 0.010 +- 0.005 | 0.081 +- 0.000 | 0.046 +- 0.003 | 0.010 +- 0.000 | 0.954 +- 0.004 | 1.000 +- 0.000 | 0.980 +- 0.011 | 0.948 +- 0.006 |
| Frozen backbone, cross-entropy + Brier | 3 | 0.946 +- 0.002 | 0.922 +- 0.005 | 0.176 +- 0.006 | 0.085 +- 0.002 | 0.031 +- 0.002 | 0.142 +- 0.003 | 0.080 +- 0.002 | 0.007 +- 0.002 | 0.923 +- 0.004 | 1.000 +- 0.000 | 0.980 +- 0.000 | 0.882 +- 0.007 |
| LoRA, cross-entropy + Brier | 3 | 0.972 +- 0.002 | 0.970 +- 0.005 | 0.075 +- 0.006 | 0.042 +- 0.003 | 0.008 +- 0.002 | 0.074 +- 0.006 | 0.042 +- 0.003 | 0.006 +- 0.002 | 0.958 +- 0.002 | 1.000 +- 0.000 | 0.980 +- 0.008 | 0.950 +- 0.004 |

(T) = with the temperature fitted on validation.

## 3. Training cost

**Why:** the gain from LoRA has to be weighed against what it costs. The frozen backbone is read once and the features are cached, so the head trains in seconds. LoRA changes the backbone, so nothing can be cached, and every epoch needs a full forward and backward pass, which is why gradient checkpointing is used to fit in 8 GB.

| Quantity | Value (mean over main runs) |
|---|---|
| Trainable LoRA parameters (rank 8) | 5,111,808 |
| Frozen features: extraction time (near zero when the cache was reused, so the mean understates it) | 143 +- 156 s |
| Frozen head training | 4.9 +- 0.6 s |
| LoRA training (2 epochs) | 436 +- 1 s |
| Peak GPU memory during LoRA training | 2.84 +- 0.00 GB |

## 4. Learning during LoRA training

**What:** the validation accuracy measured after each epoch of LoRA training, for the default run (2 epochs) and for a run with 4 epochs.

**Why:** it shows whether training has converged, and whether the extra epochs of the 4-epoch run still help. The last column puts it next to the frozen head and the final full-validation number.

Quick validation accuracy (first 60 validation records, 2 orders) after each epoch.

| Run | Quick val accuracy after epoch 1, 2, ... | Frozen head (full val) | LoRA (full val) |
|---|---|---|---|
| main_crossentropy_seed0 | 0.967, 0.975 | 0.949 | 0.973 |
| epochs4_seed0 | 0.963, 0.981, 0.981, 0.983 | 0.949 | 0.982 |

## 5. Ablations (one seed, validation)

**What:** the default LoRA recipe, changed one setting at a time: the rank (size of the adapters), the LoRA learning rate, and the number of epochs.

**Why:** to see which settings matter and whether the default is a reasonable choice. Rank trades capacity for parameters, the learning rate trades speed for stability, and epochs trade accuracy for time.

**How to read:** every row is one run with seed 0, so differences of about 0.005 in accuracy are inside the noise (see the seed section). Look for consistent patterns across several columns rather than a single best number. The LoRA parameters and time columns give the price of each setting. These runs are scored on validation only, so a setting picked from here has not been confirmed on test.

| Setting | LoRA parameters | Accuracy | Order stability | NLL | Brier | ECE | LoRA time (s) |
|---|---|---|---|---|---|---|---|
| Default: rank 8, alpha 16, LoRA lr 2e-4, 2 epochs | 5,111,808 | 0.973 | 0.973 | 0.074 | 0.042 | 0.010 | 435 |
| Rank 4 (alpha 8) | 2,555,904 | 0.973 | 0.963 | 0.075 | 0.041 | 0.008 | 437 |
| Rank 16 (alpha 32) | 10,223,616 | 0.980 | 0.978 | 0.047 | 0.027 | 0.010 | 439 |
| LoRA lr 1e-4 | 5,111,808 | 0.964 | 0.967 | 0.106 | 0.054 | 0.015 | 437 |
| LoRA lr 5e-4 | 5,111,808 | 0.981 | 0.982 | 0.045 | 0.027 | 0.005 | 437 |
| 4 epochs | 5,111,808 | 0.982 | 0.987 | 0.038 | 0.023 | 0.008 | 873 |

## 6. Learning curve: frozen head versus LoRA by number of training records (seed 0, validation)

**What:** both the frozen head and LoRA are trained on only the first 100, 200, 400 or all 800 training records, and evaluated on the same validation set.

**Why:** it shows how much data each approach needs. LoRA has about 5 million trainable parameters against the head's much smaller number, so it may need more data before it pays off.

**How to read:** compare the frozen and LoRA columns in each row. Where LoRA falls below frozen, it is overfitting or under-trained for that amount of data. Note that the number of epochs is fixed, so fewer records also means fewer optimizer steps for both.

| Training records | Accuracy, frozen | Accuracy, LoRA | Stability, frozen | Stability, LoRA | Brier, frozen | Brier, LoRA | LoRA time (s) |
|---|---|---|---|---|---|---|---|
| 100 | 0.912 | 0.932 | 0.892 | 0.940 | 0.138 | 0.112 | 73 |
| 200 | 0.926 | 0.938 | 0.890 | 0.960 | 0.118 | 0.090 | 125 |
| 400 | 0.935 | 0.954 | 0.910 | 0.945 | 0.098 | 0.065 | 229 |
| 800 | 0.949 | 0.973 | 0.930 | 0.973 | 0.078 | 0.042 | 435 |

## 7. Controls: is the model using the image? (validation)

**Why:** high accuracy on an image task proves little until you rule out that the model is answering from the text or from label frequencies alone. The controls change the picture and keep everything else, so the drop in accuracy measures how much the answers depend on the picture.

`swapped` gives every record the image and caption of the next record, keeping its own questions and labels. `no image` removes the image and keeps the caption. Accuracy should fall toward chance if the answers come from the picture. Values are mean over the main cross-entropy runs. How to read: real images is the normal score, and the other two rows should be near chance (shown in the next table). If a question stays well above chance under the controls, part of its accuracy comes from something other than the image, such as a label that is usually the same. Here has_triangle answers yes more often than no, so a model that ignores the image still scores above 0.5 on it.

| Backbone | Condition | Overall accuracy | largest_color | has_triangle | count | leftmost |
|---|---|---|---|---|---|---|
| frozen | real images | 0.948 +- 0.001 | 0.935 +- 0.004 | 0.990 +- 0.000 | 0.980 +- 0.000 | 0.886 +- 0.001 |
| frozen | swapped images | 0.332 +- 0.001 | 0.207 +- 0.002 | 0.530 +- 0.000 | 0.305 +- 0.000 | 0.285 +- 0.005 |
| frozen | no image | 0.344 +- 0.001 | 0.259 +- 0.010 | 0.473 +- 0.004 | 0.285 +- 0.000 | 0.360 +- 0.008 |
| lora | real images | 0.972 +- 0.003 | 0.956 +- 0.008 | 1.000 +- 0.000 | 0.990 +- 0.004 | 0.944 +- 0.005 |
| lora | swapped images | 0.342 +- 0.004 | 0.223 +- 0.007 | 0.530 +- 0.000 | 0.307 +- 0.002 | 0.307 +- 0.009 |
| lora | no image | 0.360 +- 0.009 | 0.287 +- 0.014 | 0.541 +- 0.026 | 0.272 +- 0.019 | 0.339 +- 0.021 |

Overall chance on validation: 0.337 +- 0.000.

## 8. Errors

**What:** for the hardest questions, the errors broken down by the true answer. It uses seed 0 and the validation set, and counts all option orders.

**Why:** aggregate accuracy hides which answers fail. These tables show which classes the model confuses and whether LoRA fixed them or moved the errors around.

**How to read:** error rate is the share of examples whose true answer is that row and where the top option was wrong. The note under each table separates two kinds of failure. Records with at least one wrong option order are sensitive to the order of the options. Records wrong in every order are genuinely hard for the model, because no ordering rescues them. A good change turns the first kind into correct answers, and cannot be expected to fix the second kind.

### largest_color (seed 0, validation): errors by true answer

| True answer | Examples | Error rate, frozen | Error rate, LoRA |
|---|---|---|---|
| blue | 144 | 4.2% | 0.0% |
| green | 104 | 6.7% | 4.8% |
| orange | 128 | 12.5% | 8.6% |
| purple | 124 | 5.6% | 7.3% |
| red | 136 | 2.9% | 5.1% |
| yellow | 164 | 4.9% | 3.0% |

Records with at least one wrong option order: frozen 20 of 200, LoRA 12 of 200. Wrong in every order: frozen 5, LoRA 6.

### count (seed 0, validation): errors by true answer

| True answer | Examples | Error rate, frozen | Error rate, LoRA |
|---|---|---|---|
| four shapes | 228 | 3.5% | 1.8% |
| one shape | 196 | 0.0% | 0.0% |
| three shapes | 236 | 3.4% | 0.0% |
| two shapes | 140 | 0.0% | 0.0% |

Records with at least one wrong option order: frozen 4 of 200, LoRA 1 of 200. Wrong in every order: frozen 4, LoRA 1.

### leftmost (seed 0, validation): errors by true answer

| True answer | Examples | Error rate, frozen | Error rate, LoRA |
|---|---|---|---|
| circle | 288 | 11.1% | 2.1% |
| square | 232 | 11.2% | 9.5% |
| triangle | 280 | 12.1% | 6.8% |

Records with at least one wrong option order: frozen 36 of 200, LoRA 16 of 200. Wrong in every order: frozen 11, LoRA 6.

### has_triangle (seed 0, validation): errors by true answer

| True answer | Examples | Error rate, frozen | Error rate, LoRA |
|---|---|---|---|
| no | 332 | 0.0% | 0.0% |
| yes | 468 | 1.7% | 0.0% |

Records with at least one wrong option order: frozen 2 of 200, LoRA 0 of 200. Wrong in every order: frozen 2, LoRA 0.

## Reliability table (test, LoRA cross-entropy, seed 0, temperature from validation)

**What:** every test prediction is placed in a bin by the probability the model gave to its top option, and each bin reports how often that top option was correct.

**Why:** this is the table behind ECE. It shows directly whether a stated confidence can be trusted, which matters when downstream rules act on a threshold such as acting only above 90%.

**How to read:** in each row, mean confidence and accuracy should be about equal. Accuracy below confidence means overconfident, above means underconfident. Rows with only a few examples are noisy and should not be over-interpreted. The last row holds most predictions, so it dominates ECE.

| Confidence bin | Examples | Mean confidence | Accuracy |
|---|---|---|---|
| 0.0 to 0.1 | 0 | - | - |
| 0.1 to 0.2 | 0 | - | - |
| 0.2 to 0.3 | 0 | - | - |
| 0.3 to 0.4 | 0 | - | - |
| 0.4 to 0.5 | 5 | 0.480 | 0.200 |
| 0.5 to 0.6 | 31 | 0.554 | 0.742 |
| 0.6 to 0.7 | 50 | 0.657 | 0.480 |
| 0.7 to 0.8 | 77 | 0.762 | 0.675 |
| 0.8 to 0.9 | 94 | 0.854 | 0.809 |
| 0.9 to 1.0 | 2943 | 0.994 | 0.992 |

## Findings

- The model reads the image. With the real images the frozen head gets 0.948 and LoRA gets 0.972 on validation. With swapped or removed images accuracy falls to 0.33 to 0.36, which is the chance level of 0.337. The same holds after LoRA training, so LoRA did not learn to answer from the caption alone. Per question, has_triangle stays near 0.5 under the controls, because the labels are imbalanced (yes is the majority), and that is a prior, not vision.
- LoRA improves vision on top of the frozen head, but by less than on text. Accuracy goes from 0.948 to 0.972, stability from 0.926 to 0.969 and Brier from 0.079 to 0.040. Test results match (0.946 to 0.970). The gain is concentrated in leftmost (0.886 to 0.944) and largest_color (0.935 to 0.956). has_triangle and count were already at 0.98 or above.
- The remaining errors are spatial and perceptual. leftmost is still the weakest question and its errors are concentrated on squares and triangles. largest_color confuses similar colours, and orange and purple are the worst. LoRA does not fix every class: for largest_color the error rate for red and purple went up, from 2.9% to 5.1% and from 5.6% to 7.3%, while blue went to 0%. With 200 validation records per question these per-class rates are based on a few dozen examples, so the movements are only suggestive.
- Brier as a loss term helps on vision more than on text. With LoRA, validation accuracy is 0.980 against 0.972 and Brier is 0.032 against 0.040, and the gain is on leftmost (0.974 against 0.944). This is consistent over three seeds (0.001 spread for the Brier variant), but it is still a synthetic task with one dataset, so I treat it as a hint.
- Temperature scaling matters for the frozen head (T is 1.87, so it was overconfident, ECE falls from 0.028 to 0.008) and barely matters after LoRA (T is about 1.1). LoRA training made the probabilities better calibrated without any post-hoc step. The reliability table has few examples below 0.9 confidence, so the lower bins are noisy.
- Ablations (one seed, validation): rank 16 gave 0.980 against 0.973, lr 5e-4 gave 0.981, lr 1e-4 gave 0.964 and four epochs gave 0.982 at twice the time. Rank 4 matches the default accuracy with half the parameters but has a lower order stability. As for text, a higher learning rate helped, and the single-seed differences of under 0.01 are within noise.
- Unlike text, LoRA already helps with few images. At 100 training records LoRA reaches 0.932 against 0.912 for the frozen head, and the lead grows to 0.973 against 0.949 at 800. Order stability is better at every size. The visual features need fewer examples to adapt than the text task did, most likely because the frozen head's start point is already high at 0.91.
- Cost: LoRA training takes about 436 seconds for two epochs on 800 records, which is 1.6 times the text cost because each example carries 64 image tokens. Peak GPU memory is 2.84 GB. Extraction of frozen features, which includes the vision tower, took 409 seconds for the first run.

Caveats. The images are rendered by a small generator with six colours and three shapes on a plain background, so this shows that the pipeline and the LoRA adaptation work, not that the model handles natural images. The labels are exact. The controls are applied to validation only. Three seeds do not cover the dataset variance, since the records are fixed.

## Reproduce

```
experiments/suite.sh vision                 # all configurations, resumable
python experiments/vision/run.py --tag main_crossentropy_seed0 --seed 0 --test   # one configuration
python experiments/make_reports.py            # rebuild this file
```
