# Results: LoRA on text tickets

Measured on the synthetic text tickets, with a Qwen3.5-0.8B backbone and a pointer head. 14 runs, generated from `experiments/results/lora/*.json` by `experiments/make_reports.py`. Each run first trains a head on the frozen backbone (the starting point), then trains LoRA adapters on the language layers together with the head. Both stages are reported from the same run, so comparisons are paired.

## Best configuration

**Recommended recipe for the text tickets**, the `main_brier` runs:

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
| Option orders per question (frozen head) | 6, shuffled with the label moved |
| Loss | Cross-entropy + 1.0 x Brier (Brier weight 1) |

Results are the mean of 3 seeds. The default recipe with a Brier weight of 0 (cross-entropy only) and the frozen head are shown for comparison.

| Test set, mean of 3 seeds | Frozen head | LoRA, Brier weight 0 (cross-entropy only) | LoRA, Brier weight 1 (best) |
|---|---|---|---|
| Accuracy | 0.963 | 0.992 | 0.995 |
| Order stability | 0.912 | 0.977 | 0.981 |
| NLL | 0.120 | 0.028 | 0.018 |
| Brier | 0.056 | 0.013 | 0.009 |
| ECE | 0.013 | 0.004 | 0.003 |

**Why this one, and why the other settings lost:**

| Setting | What happened | Why it probably worked or did not |
|---|---|---|
| LoRA instead of a frozen backbone | Largest effect by far. Test accuracy 0.963 to 0.992, order stability 0.912 to 0.977, Brier 0.056 to 0.013. | The frozen backbone cannot be changed, so the head can only use what is already easy to read at the decide token. The department question was stuck near 0.88 for that reason. LoRA lets the backbone put the needed information there, and it also removes most of the sensitivity to option order. |
| Brier weight 1 instead of 0 | Small gain on test: NLL 0.028 to 0.018, Brier 0.013 to 0.009, accuracy 0.992 to 0.995. On validation the two are equal (0.993 accuracy, Brier 0.011 versus 0.012). | Cross-entropy only looks at the probability of the true option, and a confident mistake costs it without limit. The Brier term looks at the whole probability vector and is bounded, so it pushes the model toward better calibrated probabilities. The evidence is weak, because validation does not separate the two and the test gain is a few thousandths. It is the best configuration with test evidence, not a proven one. |
| Brier weight on the frozen head | No effect (test Brier 0.056 versus 0.055). | The frozen head has few parameters and the remaining errors come from features it cannot change, so the loss function makes no difference. |
| Rank 8 instead of 4 | Rank 4 had lower order stability (0.966 versus 0.988) at the same accuracy level (0.988 versus 0.992). | Half the capacity is enough for most of the gain but not for removing all of the order sensitivity. |
| Rank 8 instead of 16 | Rank 16 was slightly better on validation (0.997 versus 0.992) with twice the parameters. | The extra capacity helps a little. The difference is a single seed, so it is not clear it is more than noise. Rank 8 was kept for its smaller size. |
| LoRA learning rate 2e-4 instead of 1e-4 or 5e-4 | 1e-4 was worse (0.987). 5e-4 was the best single run of all (0.999 validation accuracy, Brier 0.003). | With only two epochs, a low rate does not move the backbone far enough. A higher rate moved it further and did better. |
| 2 epochs instead of 4 | 4 epochs gave 0.994 against 0.992 for double the time (541 s against 273 s). | The model had already converged after two epochs, as the per-epoch accuracy shows in section 4. |
| 800 training records | LoRA overtakes the frozen head between 200 and 400 records. | Five million trainable parameters overfit a few hundred examples. |

**What was not done:** the learning rate of 5e-4 and rank 16 looked better than the default in one seed on validation, but they were not run with three seeds or scored on the test set, so they are not claimed as the best. A natural next step is to run them with three seeds and the test set, and to try them together with the Brier term.

## 0. Before the tables: what was run, how runs are named, and what the columns mean

### What one run is

A run is one complete experiment on the synthetic text tickets. It first trains a pointer head on top of the frozen backbone, which is the starting point and the baseline (called frozen). It then trains LoRA adapters on the language layers together with the head, and that is the adapted model (called LoRA). Both stages are evaluated on the same validation records in the same run, so every frozen versus LoRA comparison in this file is paired.

Training uses 800 records. Each record has several questions, and every question is shown to the model in 6 different random option orders during training, because the head must not depend on the order in which options are listed. Validation and test use 4 fixed option orders per question, and the same records and orders in every run, so numbers are comparable between runs.

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
| Frozen backbone, cross-entropy | 3 | 0.962 +- 0.002 | 0.906 +- 0.007 | 0.122 +- 0.004 | 0.059 +- 0.002 | 0.014 +- 0.002 | 0.112 +- 0.003 | 0.058 +- 0.002 | 0.006 +- 0.002 | 1.41 +- 0.02 |
| LoRA, cross-entropy | 3 | 0.993 +- 0.001 | 0.984 +- 0.003 | 0.027 +- 0.003 | 0.011 +- 0.001 | 0.003 +- 0.000 | 0.025 +- 0.002 | 0.012 +- 0.001 | 0.002 +- 0.000 | 1.30 +- 0.05 |
| Frozen backbone, cross-entropy + Brier | 3 | 0.959 +- 0.000 | 0.901 +- 0.005 | 0.118 +- 0.005 | 0.059 +- 0.002 | 0.014 +- 0.001 | 0.112 +- 0.003 | 0.058 +- 0.002 | 0.007 +- 0.002 | 1.32 +- 0.03 |
| LoRA, cross-entropy + Brier | 3 | 0.993 +- 0.002 | 0.981 +- 0.005 | 0.024 +- 0.007 | 0.012 +- 0.004 | 0.004 +- 0.001 | 0.022 +- 0.007 | 0.011 +- 0.003 | 0.002 +- 0.000 | 1.41 +- 0.05 |

Chance accuracy on validation: 0.352 +- 0.000.

### Accuracy by question (validation)

**Why:** the overall accuracy averages all questions, and the questions differ a lot in difficulty. This table shows which questions the gain comes from and which are already solved. Each cell is the mean accuracy over the seeds for that question.

| Variant | department | refund | tone | urgency |
|---|---|---|---|---|
| Frozen backbone, cross-entropy | 0.877 +- 0.008 | 0.987 +- 0.001 | 0.995 +- 0.002 | 0.993 +- 0.004 |
| LoRA, cross-entropy | 0.977 +- 0.004 | 0.998 +- 0.002 | 1.000 +- 0.000 | 0.999 +- 0.002 |
| Frozen backbone, cross-entropy + Brier | 0.871 +- 0.007 | 0.984 +- 0.002 | 0.995 +- 0.002 | 0.991 +- 0.006 |
| LoRA, cross-entropy + Brier | 0.973 +- 0.009 | 0.999 +- 0.002 | 0.999 +- 0.001 | 1.000 +- 0.000 |

## 2. Test set (used once per run)

**What:** the same four variants on records that were never used for training, for choosing settings, or for fitting T.

**Why:** validation was looked at while making decisions, so it is slightly flattered. The test set is touched once per main run, after everything is decided, and is the number to quote as the expected performance. If the test numbers are close to the validation numbers, the choices did not overfit validation.

**How to read:** the columns are as before, with the per-question accuracy at the end. The (T) columns use the temperature fitted on validation, which is how it would be used in practice.

| Variant | Seeds | Accuracy | Order stability | NLL | Brier | ECE | NLL (T) | Brier (T) | ECE (T) | department | refund | tone | urgency |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Frozen backbone, cross-entropy | 3 | 0.963 +- 0.005 | 0.912 +- 0.012 | 0.120 +- 0.008 | 0.056 +- 0.004 | 0.013 +- 0.004 | 0.110 +- 0.005 | 0.055 +- 0.003 | 0.009 +- 0.003 | 0.882 +- 0.012 | 0.992 +- 0.002 | 0.985 +- 0.003 | 0.994 +- 0.002 |
| LoRA, cross-entropy | 3 | 0.992 +- 0.001 | 0.977 +- 0.004 | 0.028 +- 0.005 | 0.013 +- 0.001 | 0.004 +- 0.001 | 0.027 +- 0.004 | 0.013 +- 0.002 | 0.003 +- 0.001 | 0.970 +- 0.002 | 1.000 +- 0.001 | 0.999 +- 0.001 | 0.999 +- 0.002 |
| Frozen backbone, cross-entropy + Brier | 3 | 0.965 +- 0.004 | 0.914 +- 0.011 | 0.116 +- 0.008 | 0.055 +- 0.004 | 0.012 +- 0.004 | 0.110 +- 0.006 | 0.055 +- 0.004 | 0.007 +- 0.001 | 0.885 +- 0.009 | 0.995 +- 0.004 | 0.985 +- 0.001 | 0.994 +- 0.002 |
| LoRA, cross-entropy + Brier | 3 | 0.995 +- 0.001 | 0.981 +- 0.004 | 0.018 +- 0.002 | 0.009 +- 0.001 | 0.003 +- 0.001 | 0.017 +- 0.003 | 0.009 +- 0.001 | 0.003 +- 0.001 | 0.980 +- 0.003 | 1.000 +- 0.001 | 1.000 +- 0.000 | 1.000 +- 0.000 |

(T) = with the temperature fitted on validation.

## 3. Training cost

**Why:** the gain from LoRA has to be weighed against what it costs. The frozen backbone is read once and the features are cached, so the head trains in seconds. LoRA changes the backbone, so nothing can be cached, and every epoch needs a full forward and backward pass, which is why gradient checkpointing is used to fit in 8 GB.

| Quantity | Value (mean over main runs) |
|---|---|
| Trainable LoRA parameters (rank 8) | 5,111,808 |
| Frozen features: extraction time (near zero when the cache was reused, so the mean understates it) | 70 +- 72 s |
| Frozen head training | 5.8 +- 0.7 s |
| LoRA training (2 epochs) | 272 +- 1 s |
| Peak GPU memory during LoRA training | 2.73 +- 0.00 GB |

## 4. Learning during LoRA training

**What:** the validation accuracy measured after each epoch of LoRA training, for the default run (2 epochs) and for a run with 4 epochs.

**Why:** it shows whether training has converged, and whether the extra epochs of the 4-epoch run still help. The last column puts it next to the frozen head and the final full-validation number.

Quick validation accuracy (first 60 validation records, 2 orders) after each epoch.

| Run | Quick val accuracy after epoch 1, 2, ... | Frozen head (full val) | LoRA (full val) |
|---|---|---|---|
| main_crossentropy_seed0 | 0.997, 0.997 | 0.961 | 0.992 |
| epochs4_seed0 | 0.964, 0.997, 0.992, 0.992 | 0.961 | 0.994 |

## 5. Ablations (one seed, validation)

**What:** the default LoRA recipe, changed one setting at a time: the rank (size of the adapters), the LoRA learning rate, and the number of epochs.

**Why:** to see which settings matter and whether the default is a reasonable choice. Rank trades capacity for parameters, the learning rate trades speed for stability, and epochs trade accuracy for time.

**How to read:** every row is one run with seed 0, so differences of about 0.005 in accuracy are inside the noise (see the seed section). Look for consistent patterns across several columns rather than a single best number. The LoRA parameters and time columns give the price of each setting. These runs are scored on validation only, so a setting picked from here has not been confirmed on test.

| Setting | LoRA parameters | Accuracy | Order stability | NLL | Brier | ECE | LoRA time (s) |
|---|---|---|---|---|---|---|---|
| Default: rank 8, alpha 16, LoRA lr 2e-4, 2 epochs | 5,111,808 | 0.992 | 0.988 | 0.028 | 0.013 | 0.004 | 273 |
| Rank 4 (alpha 8) | 2,555,904 | 0.988 | 0.966 | 0.041 | 0.020 | 0.007 | 271 |
| Rank 16 (alpha 32) | 10,223,616 | 0.997 | 0.988 | 0.008 | 0.004 | 0.002 | 272 |
| LoRA lr 1e-4 | 5,111,808 | 0.987 | 0.966 | 0.039 | 0.020 | 0.005 | 272 |
| LoRA lr 5e-4 | 5,111,808 | 0.999 | 0.996 | 0.005 | 0.003 | 0.001 | 271 |
| 4 epochs | 5,111,808 | 0.994 | 0.988 | 0.024 | 0.010 | 0.004 | 541 |

## 6. Learning curve: frozen head versus LoRA by number of training records (seed 0, validation)

**What:** both the frozen head and LoRA are trained on only the first 100, 200, 400 or all 800 training records, and evaluated on the same validation set.

**Why:** it shows how much data each approach needs. LoRA has about 5 million trainable parameters against the head's much smaller number, so it may need more data before it pays off.

**How to read:** compare the frozen and LoRA columns in each row. Where LoRA falls below frozen, it is overfitting or under-trained for that amount of data. Note that the number of epochs is fixed, so fewer records also means fewer optimizer steps for both.

| Training records | Accuracy, frozen | Accuracy, LoRA | Stability, frozen | Stability, LoRA | Brier, frozen | Brier, LoRA | LoRA time (s) |
|---|---|---|---|---|---|---|---|
| 100 | 0.773 | 0.692 | 0.762 | 0.629 | 0.351 | 0.459 | 40 |
| 200 | 0.854 | 0.817 | 0.771 | 0.710 | 0.228 | 0.260 | 72 |
| 400 | 0.932 | 0.964 | 0.850 | 0.922 | 0.111 | 0.061 | 139 |
| 800 | 0.961 | 0.992 | 0.899 | 0.988 | 0.062 | 0.013 | 273 |

## 7. Errors

**What:** for the hardest questions, the errors broken down by the true answer. It uses seed 0 and the validation set, and counts all option orders.

**Why:** aggregate accuracy hides which answers fail. These tables show which classes the model confuses and whether LoRA fixed them or moved the errors around.

**How to read:** error rate is the share of examples whose true answer is that row and where the top option was wrong. The note under each table separates two kinds of failure. Records with at least one wrong option order are sensitive to the order of the options. Records wrong in every order are genuinely hard for the model, because no ordering rescues them. A good change turns the first kind into correct answers, and cannot be expected to fix the second kind.

### department (seed 0, validation): errors by true answer

| True answer | Examples | Error rate, frozen | Error rate, LoRA |
|---|---|---|---|
| billing | 208 | 10.1% | 2.4% |
| none | 20 | 5.0% | 5.0% |
| not_listed | 40 | 12.5% | 0.0% |
| other | 56 | 19.6% | 5.4% |
| returns | 204 | 19.6% | 3.4% |
| shipping | 180 | 12.8% | 2.2% |
| technical | 232 | 10.3% | 1.7% |

Records with at least one wrong option order: frozen 64 of 235, LoRA 11 of 235. Wrong in every order: frozen 2, LoRA 3.

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
| 0.4 to 0.5 | 3 | 0.439 | 0.667 |
| 0.5 to 0.6 | 13 | 0.538 | 0.231 |
| 0.6 to 0.7 | 17 | 0.637 | 0.647 |
| 0.7 to 0.8 | 24 | 0.754 | 0.667 |
| 0.8 to 0.9 | 28 | 0.856 | 0.929 |
| 0.9 to 1.0 | 3543 | 0.997 | 0.998 |

## Findings

- LoRA clearly beats the frozen backbone at the full training size. Validation accuracy goes from 0.962 to 0.993, order stability from 0.906 to 0.984, and Brier from 0.059 to 0.011, over three seeds with small spread. The test set tells the same story (0.963 to 0.992 accuracy, stability 0.912 to 0.977).
- Almost the whole gain is on the hard question. The frozen head reaches only 0.877 on department, LoRA reaches 0.977. The other three questions were already at 0.99 or above, so there was little left to gain there.
- The mistakes that remain are no longer about option order. The number of records with at least one wrong order drops from 64 of 235 to 11 of 235, while the number wrong in every order stays at 2 versus 3. What LoRA removed was order sensitivity, not genuinely hard records.
- Calibration comes with accuracy. NLL falls from 0.122 to 0.027 and ECE from 0.014 to 0.003. The fitted temperature stays between 1.3 and 1.4, so the model is slightly overconfident before scaling in both cases, and temperature scaling changes little after LoRA. The reliability table shows 3543 of 3628 test predictions in the top confidence bin at 0.997 mean confidence and 0.998 accuracy. The lower bins hold few examples, so they are noisy.
- Adding the Brier term helps a little for LoRA on test (Brier 0.009 versus 0.013, NLL 0.018 versus 0.028) and does nothing measurable for the frozen head. On validation the two LoRA variants are within the seed spread, so this is a small effect that I would not oversell.
- The ablations use one seed, so differences of about 0.005 in accuracy are inside the noise. The pattern is still consistent. A higher learning rate (5e-4) and a higher rank (16) both did better than the default and both reached a validation Brier below 0.005, while a lower learning rate (1e-4) and rank 4 did worse. Four epochs gave 0.994 against 0.992 for two epochs at twice the time, which is not worth it. The best setting here (lr 5e-4) was found on validation, so its 0.999 is optimistic and I did not confirm it on test.
- With little data LoRA is worse than the frozen head. At 100 and 200 training records LoRA is below the frozen head (0.692 versus 0.773 and 0.817 versus 0.854) and its Brier is worse. The crossover is between 200 and 400 records, and at 800 records LoRA is far ahead. Five million trainable parameters overfit a few hundred examples, and the head alone does not. The number of epochs was fixed, so with fewer records LoRA also gets proportionally fewer optimizer steps, and the two effects are not separated here.
- Cost: LoRA adds about 272 seconds for two epochs on 800 records, with a peak GPU memory of 2.73 GB thanks to gradient checkpointing. The frozen head trains in under 6 seconds on cached features. If the frozen head is good enough, its training step is about fifty times cheaper, although extracting the features still takes minutes.

Caveats. The data is synthetic and templated, so the near-perfect numbers show that LoRA can learn the task, not that it will generalise to real tickets. Three seeds measure only the head initialisation, batch order and option shuffles, because the records and validation orders are fixed. The extraction time row in the cost table is a mean over runs that mostly reused cached features, so it understates the real cost. Extraction of 800 records in 6 option orders took about 180 seconds.

## Reproduce

```
experiments/suite.sh lora                 # all configurations, resumable
python experiments/lora/run.py --tag main_crossentropy_seed0 --seed 0 --test   # one configuration
python experiments/make_reports.py            # rebuild this file
```
