# jev

A decision model built on a vision-language backbone (Qwen3.5-0.8B). You give it a state (text, an image, or both) and typed questions with fixed options. One forward pass per question returns a probability for every option. No text is generated, so there is nothing to parse, the answer is always one of your options, and the probabilities can be calibrated.

Everything runs from the project root, using the project virtual environment. Commands are `python -m jev ...`.

## Contents

- [What is in the package](#what-is-in-the-package)
- [Data format](#data-format)
- [Setup](#setup)
- [Train](#train)
- [Evaluate](#evaluate)
- [Run (predict)](#run-predict)
- [Python API](#python-api)
- [Model directory](#model-directory)
- [Troubleshooting](#troubleshooting)
- [Where the reported numbers come from](#where-the-reported-numbers-come-from)
- [Limits](#limits)

## What is in the package

| File | Role |
|---|---|
| `cli.py`, `__main__.py` | The command line: `make-shapes`, `train`, `evaluate`, `predict`. |
| `pipeline.py` | `fit` (train and save), `JevPredictor` (load and answer), `evaluate_file`, `load_backbone`. |
| `render.py` | Turns a (state, question) pair into token ids with the marker tokens and the readout positions. Sanitises user text so it cannot forge a marker. |
| `vision.py` | `MMRenderer`: the same, but a state may contain images, which become the model's own image tokens plus pixel data. |
| `features.py` | Runs the backbone over rendered questions in padded batches and caches the readout vectors (`encode`, `extract`, `FeatureSet`). |
| `head.py` | `MaskedPointerHead`: scores each option's vector against the decision vector, for a padded batch. |
| `train.py` | Losses (masked cross-entropy, optional Brier term), head training, evaluation, LoRA helpers built on `peft`, end-to-end adapter training. |
| `calibration.py` | NLL, Brier score, expected calibration error, reliability tables, temperature scaling. |
| `data.py` | Converting an external record format to ours, option reordering with the label moved, JSONL loading. |
| `shapes.py` | Generator for a synthetic image dataset with exact labels. |
| `model.py` | The single-question head and wrapper used in the step-by-step tutorials. The pipeline uses `head.py`, not this. |

## Data format

One JSON object per line. A record is a state plus a list of questions.

```json
{
  "state": "My payouts failed three days in a row. Fix this ASAP.",
  "questions": [
    {"id": "department", "prompt": "Which team should handle this?", "type": "choice",
     "options": ["billing", "technical", "sales"], "label": 0},
    {"id": "escalate", "prompt": "Does this need urgent human attention?", "type": "noul",
     "options": ["no", "yes"], "label": 1},
    {"id": "frustration", "prompt": "How frustrated is the customer?", "type": "score",
     "options": ["calm", "frustrated", "furious"], "label": 2}
  ]
}
```

- `state` is a string, or a list of parts: `{"type": "text", "text": "..."}` and `{"type": "image", "path": "images/a.png"}`. Image paths are relative to the image root (see the flags below) or absolute.
- `type` is `choice` (any number of unordered options), `noul` (yes/no, options `["no", "yes"]`) or `score` (ordered options, listed from lowest to highest). Score options are never shuffled during training, because their order is the meaning.
- `label` is the index of the correct option. It is required for training and evaluation and ignored by `predict`.
- Options must be unique within a question and there must be at least two. Each question is scored independently of the others.

Converting records from another format: `jev.data.convert_file(src, dst)` converts the format used by the open-source any2jev generator and checks every label (duplicate options, out-of-range labels and similar raise an error with the file and line number).

Datasets in this project:

| Path | What it is |
|---|---|
| `data/synthetic/{train,val,test}.jsonl` | 3,000 synthetic support tickets (2,400 / 300 / 300), text only, converted to this format. The labels are encoded in keywords, so accuracy here shows the pipeline works, not how real tickets would score. |
| `data/synthetic_raw/{train,val,test}.jsonl` | The same tickets in the generator's original format, before conversion. `jev.data.convert_file` turns these into the format above. Kept so the conversion can be audited. |
| `data/smoke_12.jsonl` | 12 hand-written tickets (8 for training, 4 held out) for quick experiments. |
| `data/smoke.jsonl` | A single hand-written ticket, used to check rendering. |
| `data/shapes/` | Created by `python -m jev make-shapes`. Images of coloured shapes with four questions each. |

## Setup

From the project root, with the virtual environment active:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

This needs a CUDA build of PyTorch that supports your GPU. On a recent NVIDIA card use the CUDA 12.8 or newer wheels (the requirements file already points at the PyTorch index). Check with `python -c "import torch; print(torch.cuda.get_arch_list())"` and look for your architecture, for example `sm_120`.

Optional: `pip install -e .` installs the package and adds a `jev` command, so `jev train ...` works from anywhere. Without it, use `python -m jev ...` from the project root.

The first run downloads the model (about 1.7 GB in bf16) into the Hugging Face cache.

## Train

Training has two stages. The first always runs, the second is optional.

1. **Frozen backbone, trained head.** The backbone runs once over every training question, each in several shuffled option orders, and the readout vectors are cached. A small pointer head is trained on the cache. This takes seconds after the caching step.
2. **LoRA (`--lora`).** Small low-rank adapters are added to the language layers and trained together with the head, end to end through the backbone. There is no cache in this stage, so it is much slower and needs more GPU memory. The head starts from the stage-one head.

Then a temperature is fitted on the validation set so the probabilities are better calibrated, and everything is saved.

```bash
# text only, frozen backbone
python -m jev train --train data/synthetic/train.jsonl --val data/synthetic/val.jsonl --out runs/text_head --max-train-records 800

# text only, with LoRA
python -m jev train --train data/synthetic/train.jsonl --val data/synthetic/val.jsonl --out runs/text_lora --max-train-records 800 --lora

# images: generate the dataset, then train with LoRA
python -m jev make-shapes --out data/shapes --train 800 --val 200 --test 200
python -m jev train --train data/shapes/train.jsonl --val data/shapes/val.jsonl --out runs/shapes_lora --lora

# a quick check that everything works (about two minutes)
python -m jev train --train data/synthetic/train.jsonl --val data/synthetic/val.jsonl --out runs/dry_run --max-train-records 16 --max-val-records 12 --n-orders 2 --head-epochs 2
```

Flags for `train`:

| Flag | Default | Meaning |
|---|---|---|
| `--train`, `--val` | required | Training and validation JSONL. The validation file is also used to fit the temperature, so keep it separate from the data you finally test on. |
| `--out` | required | Directory to write the model to. |
| `--image-root` | folder of `--train` | Folder that image paths are relative to. |
| `--max-train-records`, `--max-val-records` | all | Use only the first N records. |
| `--n-orders` | 6 | Option orders per training question for the frozen head. More orders gives more order robustness and a longer caching step. |
| `--head-epochs`, `--head-lr` | 30, 1e-3 | Frozen-head training. A one-cycle schedule is used. |
| `--lora` | off | Run the LoRA stage. |
| `--lora-epochs` | 2 | Each epoch shows every question once, in a fresh random option order. |
| `--lora-batch` | 16 | Questions per step. Lower it if you run out of GPU memory. |
| `--rank`, `--alpha` | 8, 16 | LoRA rank and scale. About 5 million trainable parameters (0.6% of the model) at rank 8. |
| `--brier-weight` | 0 | Adds `weight x Brier score` to the cross-entropy loss. |
| `--seed` | 0 | Fixes the random choices in training: the head's starting values, the batch order and the option shuffles. The same seed gives the same run, and different seeds show how much a result changes from luck alone. |
| `--model-id` | `Qwen/Qwen3.5-0.8B` | The backbone. |

What to expect, from measurements on an 8 GB laptop GPU:

- Caching runs at about 8 ms per example for text and about 25 ms for images (an image adds 64 tokens to every question).
- Training the head takes seconds to a minute.
- LoRA with gradient checkpointing peaked at about 2.7 GB of GPU memory for text and 2.8 GB for images at the default batch of 16. Without checkpointing, backpropagation costs about 0.55 GB per sequence. On 800 training records, LoRA took about 136 s per epoch for text and 217 s per epoch for images, and the command prints elapsed time and peak GPU memory as it goes.
- Close other programs that hold the GPU (including idle notebook kernels, which keep the model in memory, about 2 GB each).

The command prints validation accuracy, chance level, order stability and the fitted temperature. The full metrics are saved to `metrics.json`.

## Evaluate

```bash
python -m jev evaluate --model runs/text_head --data data/synthetic/test.jsonl
python -m jev evaluate --model runs/shapes_lora --data data/shapes/test.jsonl
```

Each question is shown in several option orders (`--n-orders`, default 4). The output is JSON:

- `acc`, and `by_question`: accuracy overall and per question id. Compare with `chance`.
- `stability`: for choice and yes/no questions, the share of questions where the model picked the same option text in every order. Score questions are excluded because they are never shuffled.
- `calibration` and `calibration_temperature_scaled`: `acc`, `nll`, `brier`, `ece`, as trained and with the fitted temperature. Lower is better for the last three. A Brier score of `1 - 1/N` is what always answering uniformly would get.
- `temperature`: above 1 means the raw scores were overconfident. It is limited to the range 0.05 to 20; sitting at a limit means the scores carry almost no usable signal.

Use a test file that was not used for training or for fitting the temperature, and look at it once.

## Run (predict)

A request is a state and the questions, without labels:

```json
{
  "state": [{"type": "image", "path": "images/test_00000.png"}, {"type": "text", "text": "Photo uploaded by a user."}],
  "questions": [
    {"id": "largest_color", "prompt": "What color is the largest shape?", "type": "choice", "options": ["red", "yellow", "blue"]},
    {"id": "has_triangle", "prompt": "Is there a triangle in the image?", "type": "noul", "options": ["no", "yes"]},
    {"id": "count", "prompt": "How many shapes are in the image?", "type": "score", "options": ["one shape", "two shapes", "three shapes", "four shapes"]}
  ]
}
```

```bash
python -m jev predict --model runs/shapes_lora --request request.json --image-root data/shapes
```

The answer is a list with one entry per question:

```json
[{"id": "has_triangle", "type": "noul", "options": ["no", "yes"], "probabilities": [0.16, 0.84],
  "prediction": "yes", "confidence": 0.84}]
```

For `score` questions the entry also has `expected_level`, the probability-weighted level index (0 is the first option). `--image-root` defaults to the folder of the request file. Probabilities already include the fitted temperature.

## Python API

```python
from jev import JevPredictor, fit, evaluate_file
from jev.data import load_jsonl

metrics = fit(load_jsonl("data/synthetic/train.jsonl")[:800], load_jsonl("data/synthetic/val.jsonl"),
              out_dir="runs/text_head", lora=False)

model = JevPredictor.from_pretrained("runs/text_head")
answers = model.predict("My payouts failed three days in a row.",
                        [{"id": "department", "type": "choice", "options": ["billing", "technical", "sales"],
                          "prompt": "Which team should handle this?"}])
print(answers[0]["prediction"], answers[0]["confidence"])

report = evaluate_file("runs/text_head", "data/synthetic/test.jsonl")
```

Loading the model once and calling `predict` repeatedly is the efficient pattern. Each question costs one forward pass through the backbone.

Lower-level pieces (`Renderer`, `MMRenderer`, `extract`, `train_head`, `train_lora`, `add_lora`, the calibration functions) can be used directly, as the tutorials do.

## Model directory

```
runs/<name>/
  config.json    backbone id, head size, whether LoRA was used, rank, alpha, fitted temperature
  head.pt        the pointer head weights
  adapter/       the LoRA adapter (only with --lora), in the standard peft format
  metrics.json   validation metrics at the end of training
```

The backbone itself is not copied. It is loaded from the Hugging Face cache by id.

## Troubleshooting

- **Out of GPU memory during LoRA.** Lower `--lora-batch` (try 8, then 4), and close other processes using the GPU. Check with `nvidia-smi`.
- **`image 'images/...' not found`.** Image paths are relative to the image root. Set `--image-root` to the folder that contains the `images/` directory, or use absolute paths.
- **Temperature at 20 or 0.05.** The model is at chance or nearly so on the validation set. Look at accuracy and `chance` first.
- **Accuracy is high on the synthetic text data but you doubt it.** You should. The tickets are template-based and the labels are encoded in keywords. Test on your own data before trusting a number.
- **Warnings about missing `causal_conv1d` or `flash-linear-attention`.** These are speed notes. The model runs on a PyTorch fallback with identical results, only slower.
- **Numbers differ slightly between runs.** The backbone runs in bf16, and batch shapes change the low-order bits. Vectors agree to a cosine similarity above 0.997. Compare in fp32 before treating a small difference as a bug.

## Where the reported numbers come from

The full LoRA and vision studies (three seeds, ablations, learning curves, controls) are run with `experiments/suite.sh`, which uses the same pipeline code but adds the experiment bookkeeping. The measured results for this package are in `results/RESULTS.md` (pointer head), `results/RESULTS_LORA.md` and `results/RESULTS_VISION.md`, produced by the suites in `experiments/` (see `experiments/README.md`). The scripts that produced them are archived, as they were run, in `experiments/archive/as_run/`, and `experiments/archive/as_run/ALL_SCRIPTS.md` explains each one. They are a record of the exploration. For training and evaluating today, use the commands above.

## Limits

- The backbone is a 0.85B-parameter model. The vision tower is always frozen; LoRA only adapts the language layers.
- Each question is answered in its own row, with the state repeated. The model has hybrid memory layers, so questions cannot share one packed sequence with a mask.
- Option order slightly changes the answer because the model reads left to right. Training shuffles options and moves the labels to reduce this, and `stability` measures what is left.
- Validation and test numbers on the bundled synthetic data are a pipeline check, not a benchmark.
