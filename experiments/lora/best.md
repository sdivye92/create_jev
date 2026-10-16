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
