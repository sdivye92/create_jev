# Experiments

Everything that produced a reported number lives here, grouped by purpose.

```
experiments/
  engine.py            shared experiment code: datasets, feature caching, evaluation suite, the LoRA run itself
  lora/                text LoRA study (synthetic tickets)
    run.py             one configuration per call
    configs.txt        the configuration matrix used by the suite
    best.md            hand-written best configuration and why it won, placed at the top of results/RESULTS_LORA.md
    findings.md        hand-written commentary that is appended to results/RESULTS_LORA.md
  vision/              vision study (synthetic shapes images, with image controls)
    run.py
    configs.txt
    best.md            best configuration, at the top of results/RESULTS_VISION.md
    findings.md        appended to results/RESULTS_VISION.md
  suite.sh             runs every configuration of a group, resumable
  make_reports.py      builds results/RESULTS_LORA.md and results/RESULTS_VISION.md from the result JSON files plus best.md and findings.md (no number in the generated tables is typed by hand)
  results/             one JSON per configuration and the run logs
  investigations/      side questions answered along the way (cache reuse, bf16 versus fp32)
  archive/as_run/      the pointer-head scripts exactly as they were run, kept for traceability (see archive/README.md)
```

## Running

```bash
experiments/suite.sh lora      # text LoRA suite: 14 configurations, about 1.5 hours in total on an 8 GB laptop GPU
experiments/suite.sh vision    # vision suite: 14 configurations, about 2.5 hours in total
python experiments/make_reports.py
```

A single configuration can be run directly, for example `python experiments/lora/run.py --tag try --seed 0 --rank 16 --alpha 32 --no-save`.
Pass `--smoke` for a tiny run that only checks the plumbing.
Configurations whose result file already exists are skipped by the suite, so an interrupted suite can simply be started again.
After editing `best.md` or `findings.md`, rerun `python experiments/make_reports.py` to refresh the reports; it does not touch the GPU.

## Run names and seeds

A result file is named `<what changes>_seed<N>`, for example `main_brier_seed1` or `size200_seed0`.
The first part says what differs from the standard recipe, and `configs.txt` in each group lists every name with a one-line meaning.
The seed fixes the random choices in training (the head's starting values, the batch order and the random option orders), so running the main configurations with seeds 0, 1 and 2 shows how much a result moves from luck alone.
The ablations use seed 0 only.
The reports explain all of this again next to the tables, together with the meaning of every column.

## Rules the experiments follow

- The frozen head is trained first, then LoRA and the head are trained together, and both are evaluated on the same validation set.
- The test set is evaluated once per main run, with the temperature fitted on validation. Ablations report validation only.
- Feature extraction for the frozen baseline is cached under `runs/cache/`, so repeated runs on the same data and seed reuse it.

The reports are [RESULTS_LORA.md](../results/RESULTS_LORA.md) and [RESULTS_VISION.md](../results/RESULTS_VISION.md), next to the pointer-head report [RESULTS.md](../results/RESULTS.md).
