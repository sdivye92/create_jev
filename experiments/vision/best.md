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
