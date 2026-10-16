# main_crossentropy_seed0

The saved LoRA adapter and pointer head from the main text-ticket run of the [LoRA study](../../../results/RESULTS_LORA.md): seed 0, cross-entropy loss only (Brier weight 0).

It is a worked example of what the experiment suite leaves behind, not a general-purpose model. It was trained on 800 synthetic support tickets, so it shows that the pipeline works and says nothing about how real tickets would score.

## Files

| File | What it is | Size |
|---|---|---|
| `adapter_model.safetensors` | The LoRA weights: 300 tensors on the language layers of the backbone, 5,111,808 trainable parameters. | 20 MB |
| `adapter_config.json` | The PEFT configuration: rank 8, alpha 16, no dropout, targets `in_proj_qkv`, `in_proj_z`, `out_proj`, `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj` and `down_proj` inside `language_model.layers` (the vision tower is not touched). | 1 KB |
| `head.pt` | The pointer head trained together with the adapter: `opt_proj` and `dec_proj`, each a 1024 to 256 projection. | 2 MB |

The backbone is not stored. It is loaded by id (`Qwen/Qwen3.5-0.8B`) from the Hugging Face cache.

## How it was trained

| Setting | Value |
|---|---|
| Backbone | `Qwen/Qwen3.5-0.8B`, bf16 |
| Data | `data/synthetic/train.jsonl`, first 800 records |
| Frozen head first | 30 epochs, one-cycle schedule, peak 1e-3, 6 shuffled option orders per question |
| LoRA | rank 8, alpha 16, LoRA learning rate 2e-4, head learning rate 5e-4, 2 epochs, batch 16 |
| Loss | cross-entropy only (Brier weight 0) |
| Cost | 273 s of LoRA training, 2.73 GB peak GPU memory, 459 s in total on an 8 GB laptop GPU |

The best recipe in the study adds a Brier term (`main_brier_*`). This run is the cross-entropy baseline it is compared with.

## Results

Accuracy, order stability and calibration of this one run, on the synthetic tickets (chance accuracy is 0.352).

| | Frozen head | This adapter |
|---|---|---|
| Validation accuracy | 0.961 | 0.992 |
| Test accuracy | 0.957 | 0.991 |
| Test order stability | 0.897 | 0.982 |
| Test Brier score | 0.061 | 0.012 |
| Test ECE | 0.019 | 0.003 |

Test accuracy per question: department 0.971, refund 1.000, tone 0.998, urgency 0.996. The test set was scored once. Single-seed numbers move by a few thousandths between seeds, so the three-seed means in [RESULTS_LORA.md](../../../results/RESULTS_LORA.md) are the ones to quote.

## Using it

This folder is the raw output of `experiments/lora/run.py`, not a model directory written by `python -m jev train`. It has no `config.json` and the adapter sits at the top level instead of in `adapter/`, so `JevPredictor.from_pretrained` cannot read it as it is.

To make it loadable, copy the folder and rearrange it:

```bash
cp -r runs/lora_text/main_crossentropy_seed0 runs/text_lora_ce
cd runs/text_lora_ce
mkdir adapter && mv adapter_model.safetensors adapter_config.json adapter/
cat > config.json <<'EOF'
{"model_id": "Qwen/Qwen3.5-0.8B", "hidden_size": 1024, "d_head": 256, "normalize": true,
 "lora": true, "rank": 8, "alpha": 16, "temperature": 1.2681443691253662}
EOF
```

The temperature is the one fitted on the validation set for this run. Then use it like any trained model:

```bash
python -m jev evaluate --model runs/text_lora_ce --data data/synthetic/test.jsonl
```

See [jev/README.md](../../../jev/README.md) for the request format and the Python API.

## Reproducing it

```bash
python experiments/lora/run.py --tag main_crossentropy_seed0 --seed 0 --test
```

The command writes this folder and `experiments/results/lora/main_crossentropy_seed0.json`. Training is seeded, but the backbone runs in bf16, so a rerun can differ in the low-order bits.

## Versions

PEFT 0.21.2, with the transformers development branch listed in `requirements.txt` (Qwen3.5 is newer than the last release).
