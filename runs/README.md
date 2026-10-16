# runs

Everything the training code writes lands here: cached backbone features and saved models. All of it can be regenerated, so none of it is source.

The folder and its subfolders are created automatically the first time a run needs them, so a fresh clone does not contain `cache/` and you never create anything here by hand.

```
runs/
  cache/          cached readout vectors from the frozen backbone (about 1.2 GB)
  lora_text/      saved adapters and heads from the text LoRA suite, one folder per configuration
  vision/         the same for the vision suite (created when a vision configuration is saved)
  <name>/         a model directory written by `python -m jev train --out runs/<name>`
```

## cache/

The experiment suites run the frozen backbone once per question and keep the readout vectors, so that the frozen-head baseline and repeated runs do not pay for extraction again. Each file is a `FeatureSet` saved with `torch.save`.

| File name | What it holds |
|---|---|
| `<data>_train<N>_o<K>_s<seed>.pt` | Training features: N records, each question shown in K shuffled option orders, with the random choices fixed by the seed. |
| `<data>_val<N>_frozen.pt`, `<data>_test<N>_frozen.pt` | Validation and test features, in four fixed option orders. |
| `shapes_val<N>_swapped_frozen.pt`, `shapes_val<N>_noimage_frozen.pt` | The vision controls: the same validation questions with the image swapped for another one, and with the image removed. |

`<data>` is `text` for the support tickets and `shapes` for the images, so `text_train800_o6_s0.pt` is 800 text records in 6 orders with seed 0.

A cache file is found by its name only. Nothing checks that the data or the model behind it is unchanged. Delete the files in `cache/` after you change the dataset, the backbone, the rendering code or the number of options, otherwise a run silently reuses old vectors.

## lora_text/ and vision/

Each configuration run with saving on gets a folder named after its tag, for example `lora_text/main_crossentropy_seed0/`. It holds the LoRA adapter (`adapter_model.safetensors`, `adapter_config.json`) and the trained pointer head (`head.pt`). The suites normally run with `--no-save`, so only the main runs you asked to keep appear here. See [lora_text/main_crossentropy_seed0/README.md](lora_text/main_crossentropy_seed0/README.md) for what one of them contains and how to load it.

These folders are experiment checkpoints, not full model directories. They have no `config.json` and no fitted temperature, and the adapter is not in an `adapter/` subfolder, so `python -m jev evaluate` and `JevPredictor` cannot read them as they are. The README in that folder shows the small rearrangement that makes one loadable.

## Model directories

`python -m jev train --out runs/<name>` writes a complete model that the command line and `JevPredictor` can load:

```
runs/<name>/
  config.json    backbone id, head size, whether LoRA was used, rank, alpha, fitted temperature
  head.pt        the pointer head weights
  adapter/       the LoRA adapter (only with --lora), in the standard peft format
  metrics.json   validation metrics at the end of training
```

The backbone itself is never copied. It is loaded from the Hugging Face cache by id. The format is described further in [jev/README.md](../jev/README.md#model-directory).

## Git

Only `runs/cache/` is ignored. The cache files are 40 to 180 MB each, and GitHub rejects any file over 100 MB, so they cannot be committed. `*.pt` is ignored as well, with an exception for `head.pt`, so a saved adapter folder (about 22 MB) can be committed with its head.

Everything else under `runs/` is tracked, including any model you save with `python -m jev train --out runs/<name>`. Add the folder to `.gitignore` if you do not want it shared.
