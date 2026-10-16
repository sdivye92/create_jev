# create_jev

A from-scratch exploration of how to turn a small vision-language model (Qwen3.5-0.8B) into a decision engine. Show it a situation (text, an image, or both) and a typed question with fixed options, and one forward pass returns a calibrated probability for every option. No text is generated, so the output is always valid and fast to compute.

The project is both a working package and a worked example of how it was figured out. Everything runs on a single 8 GB laptop GPU.

## What is in here

```
jev/                  the package: rendering, feature caching, pointer head, LoRA, calibration, CLI   (see jev/README.md)
notebooks/            nine step-by-step notebooks, from opening the model to answering questions about images
data/                 datasets: synthetic text tickets (converted and raw), hand-written examples, generated image data
experiments/          the LoRA and vision experiment suites, report generator, side investigations, and the archive of the pointer-head scripts   (see experiments/README.md)
results/              the written reports:
  RESULTS.md            pointer-head results on the frozen backbone, starting with its best configuration
  RESULTS_LORA.md       LoRA versus the frozen backbone on text tickets: best configuration, terms and columns explained, three seeds, ablations, learning curve, errors
  RESULTS_VISION.md     the same study on rendered shapes images, with controls that check the image is used
requirements.txt, pyproject.toml
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You need a CUDA build of PyTorch that supports your GPU (CUDA 12.8 or newer for recent NVIDIA cards). Confirm with `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_arch_list())"`.

Train a model on the bundled text tickets, evaluate it, and ask it a question:

```bash
python -m jev train --train data/synthetic/train.jsonl --val data/synthetic/val.jsonl --out runs/text_head --max-train-records 800
python -m jev evaluate --model runs/text_head --data data/synthetic/test.jsonl
python -m jev predict --model runs/text_head --request request.json
```

Train a model that looks at images:

```bash
python -m jev make-shapes --out data/shapes
python -m jev train --train data/shapes/train.jsonl --val data/shapes/val.jsonl --out runs/shapes_lora --lora
```

All flags, the data format, the request format and the output format are in [jev/README.md](jev/README.md).

## How it works, in short

1. The state and the question are written into one token sequence with marker tokens: one after each option, and a final "decide" marker.
2. The backbone reads the sequence once. The hidden vector at each option marker represents that option, and the vector at the decide marker represents the question.
3. A small pointer head scores each option vector against the decide vector. Softmax turns the scores into probabilities.
4. Optionally, LoRA adapters let the backbone learn what to put into those vectors. A temperature fitted on held-out data makes the probabilities honest.

## Learning path

The notebooks build the system in order. Each one asks questions, runs the experiments, and explains what the output means.

| Notebook | Topic |
|---|---|
| `01_explore_architecture` | Open the model: modules, the 24-layer hybrid pattern, parameter counts, tied embeddings. |
| `02_forward_pass_and_caching` | Hidden states and caches. Why attention layers keep a growing list and the memory layers keep a fixed-size state. |
| `03_rows_mode_dry_run` | Why questions cannot share a masked sequence here, and a cache-reuse test (including the bf16 false alarm). |
| `04_jev_data_format` | The record schema, marker tokens, rendering, and protection against forged markers. |
| `05_model_surgery` | Dropping the output head, choosing the readout, and designing the pointer head from the raw vectors. |
| `06_training_the_head` | First training run on a tiny set: memorisation, learning rates, shuffling options. |
| `07_scaling_up` | Training on 800 records: schedule, normalisation, learning curve, error analysis, the test set once. |
| `08_lora_and_calibration` | LoRA with `peft`, gradient checkpointing, Brier score, calibration error and temperature scaling. |
| `09_vision` | Images as input: how the model sees them, a shapes dataset, controls that check the image is really used. |

Notebooks 8 and 9 have a `SMOKE = True` switch for a short dry run that only checks they execute. They use the same code as the command line.

## Reproducing the LoRA and vision studies

```bash
experiments/suite.sh lora        # text tickets: 14 configurations, about 1.5 hours on an 8 GB laptop GPU
experiments/suite.sh vision      # shapes images: 14 configurations, about 2.5 hours
python experiments/make_reports.py   # rebuilds results/RESULTS_LORA.md and results/RESULTS_VISION.md from the result files
```

The suites are resumable, and every configuration is named `<what changes>_seed<N>`, where the seed fixes the random choices in training so that three seeds show the run-to-run spread. [experiments/README.md](experiments/README.md) explains the layout, and each report explains its own columns and tables.

## Where the numbers come from

Every pointer-head number is in [RESULTS.md](results/RESULTS.md), tagged as coming from a notebook run or from a prototype script. The LoRA and vision numbers come from the suites in [experiments/](experiments/README.md) and are rebuilt from their result files by `experiments/make_reports.py`. The pointer-head scripts are archived verbatim in [experiments/archive/as_run/](experiments/archive/as_run/), and [experiments/archive/as_run/ALL_SCRIPTS.md](experiments/archive/as_run/ALL_SCRIPTS.md) lists them in the order they were written, with what each one produced and the full code.

Treat those scripts as a record, not a kit. They contain hard-coded paths, and two of them (`lora_probe.py`, `lora_mem.py`) import a hand-written LoRA module that was later replaced by `peft`. The supported way to train and evaluate is the command line in [jev/README.md](jev/README.md).

## Best configuration of each experiment

| Experiment | Best configuration | Test metrics |
|---|---|---|
| Pointer head, frozen backbone (text) | 800 records, 6 shuffled option orders, input normalisation, one-cycle schedule (peak 1e-3, 30 epochs) | accuracy 0.959, order stability 0.900 (seed 0, chance 0.352) |
| LoRA (text) | Rank 8, LoRA learning rate 2e-4, 2 epochs, cross-entropy plus 1.0 x Brier | accuracy 0.995, stability 0.981, Brier 0.009, ECE 0.003 (3 seeds) |
| Vision (shapes images) | The same LoRA recipe, with the Brier term | accuracy 0.972, stability 0.970, Brier 0.042, ECE 0.008 (3 seeds, chance 0.337) |

Each "Best configuration" section has a one-row-per-hyperparameter table of the recipe, its metrics, and the reasons each setting beat the alternatives. They are at the top of [RESULTS.md](results/RESULTS.md), [RESULTS_LORA.md](results/RESULTS_LORA.md) and [RESULTS_VISION.md](results/RESULTS_VISION.md). Single-seed ablations (rank 16, a LoRA learning rate of 5e-4, 4 epochs) scored slightly higher on validation, but they were not run with three seeds or on the test set, so they are not claimed as best.

## Results so far

The frozen-backbone results are in [RESULTS.md](results/RESULTS.md). The LoRA and vision results are in [RESULTS_LORA.md](results/RESULTS_LORA.md) and [RESULTS_VISION.md](results/RESULTS_VISION.md), each with three seeds, ablations and a test set that was used once per run.

- On the bundled synthetic text tickets, a head trained on the frozen backbone with 800 training records reaches 0.959 accuracy on the test set (chance 0.352), and the test result agrees with validation.
- A learning-rate schedule cut the seed-to-seed spread from 0.038 to 0.001, and shuffling the option order added about six points of accuracy and seventeen of order stability.
- Those tickets are template-based and the labels live in keywords, so this shows the pipeline works. It does not show how real tickets would score.
- LoRA on the text tickets (rank 8, 5.1M trainable parameters, 800 training records) lifts test accuracy from 0.963 to 0.992 and order stability from 0.912 to 0.977, and it improves calibration (test Brier 0.056 to 0.013). With 100 to 200 training records it is worse than the frozen head, and the crossover is between 200 and 400 records.
- On rendered shapes images the frozen head reaches 0.946 on test and LoRA reaches 0.970. Swapping the image or removing it drops accuracy to chance (about 0.34), so the answers do come from the picture. Adding a Brier term to the loss gave a small further gain on images.
- Both studies use synthetic data, three seeds and one validation set, so differences of about 0.005 in the ablations are within noise. See the caveats at the end of each report.

## Notes

- Hardware used: RTX 5060 laptop GPU (8 GB), PyTorch 2.14 with CUDA 13.0 wheels, transformers built from its development branch (Qwen3.5 is newer than the last release).
- `requirements.txt` lists `dotenv`; the package that provides `load_dotenv` is `python-dotenv`.
- `datasets` is listed for upcoming experiments on public datasets and is not used by the package yet.
- Keep secrets such as `HF_TOKEN` in `.env` (ignored by git).
- Generated data and trained models (`data/shapes*/`, `runs/`) are ignored by git.
