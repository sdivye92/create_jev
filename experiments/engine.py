"""Shared engine for the LoRA and vision experiments.

One run = one configuration. It trains a pointer head on the frozen backbone, then LoRA adapters plus the head, and records
validation metrics (accuracy, order stability, calibration, errors, controls) for both stages in a JSON file.
Frozen-backbone features are cached on disk, so repeated runs on the same data only pay for extraction once.
"""
import argparse
import copy
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch  # noqa: E402

from jev.calibration import calibration_report, expected_calibration_error, fit_temperature, probabilities  # noqa: E402
from jev.data import load_jsonl  # noqa: E402
from jev.features import build_requests, extract  # noqa: E402
from jev.head import MaskedPointerHead  # noqa: E402
from jev.pipeline import load_backbone  # noqa: E402
from jev.shapes import generate_dataset  # noqa: E402
from jev.train import add_lora, chance_accuracy, evaluate, predict_scores, save_run, train_head, train_lora  # noqa: E402
from jev.vision import MMRenderer  # noqa: E402

DATASETS = {
    "text": {"dir": "data/synthetic", "orders": 6, "error_qids": ["department"], "sizes": {"train": 800, "val": 300, "test": 300}},
    "shapes": {"dir": "data/shapes", "orders": 4, "error_qids": ["largest_color", "count", "leftmost", "has_triangle"],
               "sizes": {"train": 800, "val": 200, "test": 200}},
}
EXTRACT_BATCH = 32


def add_common_args(parser):
    parser.add_argument("--tag", required=True, help="name of this run (used for the result file)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=800, help="training records used (a prefix of the file)")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--lora-lr", type=float, default=2e-4)
    parser.add_argument("--head-lr", type=float, default=5e-4, help="learning rate of the head during the LoRA stage")
    parser.add_argument("--epochs", type=int, default=2, help="LoRA epochs")
    parser.add_argument("--batch", type=int, default=16, help="LoRA batch size")
    parser.add_argument("--brier", type=float, default=0.0, help="weight of the Brier term in the loss")
    parser.add_argument("--test", action="store_true", help="also evaluate the test set (once, with the validation temperature)")
    parser.add_argument("--no-save", action="store_true", help="do not save the adapter")
    parser.add_argument("--smoke", action="store_true", help="tiny sizes, only checks that everything runs")
    return parser


def cached(name, builder):
    path = ROOT / "runs" / "cache" / f"{name}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    obj = builder()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)
    return obj


def error_analysis(head, fs, qids):
    """Confusions by option text, and how many tickets (records) are wrong in some or all option orders."""
    preds = predict_scores(head, fs).argmax(-1).tolist()
    labels = fs.labels.cpu().tolist()
    out = {}
    for qid in qids:
        confusion, per_record = defaultdict(lambda: defaultdict(int)), defaultdict(lambda: [0, 0])
        for q, r, opts, p, y in zip(fs.meta["qid"], fs.meta["rec"], fs.meta["options"], preds, labels):
            if q != qid:
                continue
            true, got = opts[y].split(": ")[0], opts[p].split(": ")[0]
            confusion[true][got] += 1
            per_record[r][1] += 1
            per_record[r][0] += true != got
        out[qid] = {"confusion": {t: dict(v) for t, v in confusion.items()}, "records": len(per_record),
                    "records_with_error": sum(w > 0 for w, _ in per_record.values()),
                    "records_always_wrong": sum(w == n for w, n in per_record.values())}
    return out


def suite(head, fs, temperature=None, qids=()):
    """Accuracy, order stability, calibration as trained and temperature-scaled. Fit the temperature only on validation data."""
    r = evaluate(head, fs)
    scores, mask, labels = predict_scores(head, fs), fs.mask.cpu(), fs.labels.cpu()
    fitted = temperature is None
    if fitted:
        temperature = fit_temperature(scores, mask, labels)
    _, table = expected_calibration_error(probabilities(scores / temperature, mask), labels)
    out = {"acc": r["acc"], "loss": r["loss"], "by_question": r["by_question"], "stability": r["stability"], "chance": chance_accuracy(fs),
           "temperature": temperature, "temperature_fitted_here": fitted, "cal": calibration_report(scores, mask, labels),
           "cal_scaled": calibration_report(scores, mask, labels, temperature), "reliability_scaled": table}
    if qids:
        out["errors"] = error_analysis(head, fs, qids)
    return out


def with_next_state(records):
    states = [r["state"] for r in records]
    return [{**r, "state": states[(i + 1) % len(records)]} for i, r in enumerate(records)]


def without_images(records):
    return [{**r, "state": [p for p in r["state"] if p["type"] != "image"] if isinstance(r["state"], list) else r["state"]} for r in records]


def run(cfg, dataset):
    spec = DATASETS[dataset]
    root = ROOT / spec["dir"]
    if dataset == "shapes" and not (root / "train.jsonl").exists():
        generate_dataset(root, spec["sizes"], seed=0)
    sizes = {"train": 24, "val": 12, "test": 12} if cfg.smoke else spec["sizes"]
    n_train = min(cfg.n_train, sizes["train"])
    train_all = load_jsonl(root / "train.jsonl")[: sizes["train"]]
    train_records = train_all[:n_train]
    val_records = load_jsonl(root / "val.jsonl")[: sizes["val"]]
    test_records = load_jsonl(root / "test.jsonl")[: sizes["test"]]
    qids = spec["error_qids"]
    orders = 2 if cfg.smoke else spec["orders"]
    head_epochs = 2 if cfg.smoke else 30
    key = f"{dataset}{'_smoke' if cfg.smoke else ''}"
    out_dir = ROOT / "experiments" / "results" / ("lora" if dataset == "text" else "vision")
    result_path = out_dir / f"{cfg.tag}.json"

    def log(msg):
        print(f"[{cfg.tag} {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    result = {"config": {**vars(cfg), "dataset": dataset, "n_train_used": n_train, "orders": orders, "head_epochs": head_epochs}}
    processor, core, hidden = load_backbone()
    pad = processor.tokenizer.pad_token_id
    device = next(core.parameters()).device
    renderer = MMRenderer(processor, root)
    val_requests = build_requests(val_records, n_orders=4, seed=2)

    # ---- frozen backbone: features (cached), head
    t0 = time.time()
    train_full = cached(f"{key}_train{len(train_all)}_o{orders}_s{cfg.seed}",
                        lambda: extract(core, renderer, build_requests(train_all, orders, cfg.seed + 1), pad, hidden, EXTRACT_BATCH))
    train_fs = train_full.where_rec_below(n_train) if n_train < len(train_all) else train_full
    val_frozen = cached(f"{key}_val{len(val_records)}_frozen", lambda: extract(core, renderer, val_requests, pad, hidden, EXTRACT_BATCH))
    controls_frozen = {}
    if dataset == "shapes":
        controls_frozen["swapped"] = cached(f"{key}_val{len(val_records)}_swapped_frozen", lambda: extract(
            core, renderer, build_requests(with_next_state(val_records), 4, 2), pad, hidden, EXTRACT_BATCH))
        controls_frozen["no_image"] = cached(f"{key}_val{len(val_records)}_noimage_frozen", lambda: extract(
            core, renderer, build_requests(without_images(val_records), 4, 2), pad, hidden, EXTRACT_BATCH))
    test_frozen = None
    if cfg.test:
        test_frozen = cached(f"{key}_test{len(test_records)}_frozen", lambda: extract(
            core, renderer, build_requests(test_records, 4, 3), pad, hidden, EXTRACT_BATCH))
    result["timing_extract_s"] = round(time.time() - t0, 1)
    log(f"frozen features ready: {len(train_fs)} train, {len(val_frozen)} val examples ({result['timing_extract_s']}s)")

    t0 = time.time()
    head_f, _ = train_head(train_fs, val_frozen, hidden, epochs=head_epochs, lr=1e-3, seed=cfg.seed, brier_weight=cfg.brier)
    result["timing_head_s"] = round(time.time() - t0, 1)
    frozen = {"val": suite(head_f, val_frozen, qids=qids)}
    T_frozen = frozen["val"]["temperature"]
    frozen["controls"] = {name: suite(head_f, fs, temperature=T_frozen) for name, fs in controls_frozen.items()}
    if test_frozen is not None:
        frozen["test"] = suite(head_f, test_frozen, temperature=T_frozen, qids=qids)
    result["frozen"] = frozen
    log(f"frozen head: val acc {frozen['val']['acc']:.3f}, stability {frozen['val']['stability']:.3f}")

    # ---- LoRA on the language layers, trained end to end with the head
    peft_model = add_lora(core, rank=cfg.rank, alpha=cfg.alpha)
    n_lora = sum(p.numel() for p in core.parameters() if p.requires_grad)
    head_l = MaskedPointerHead(hidden).to(device)
    head_l.load_state_dict(head_f.state_dict())
    quick = build_requests(val_records[: min(60, len(val_records))], 2, 7)
    quick_hist = []

    def monitor(epoch):
        r = evaluate(head_l, extract(core, renderer, quick, pad, hidden, EXTRACT_BATCH))
        quick_hist.append({"epoch": epoch, "quick_val_acc": r["acc"], "quick_val_loss": r["loss"]})
        log(f"  end of epoch {epoch}: quick val acc {r['acc']:.3f}")
        return {}

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    hist = train_lora(core, head_l, renderer, train_records, pad, epochs=1 if cfg.smoke else cfg.epochs, batch_size=4 if cfg.smoke else cfg.batch,
                      lora_lr=cfg.lora_lr, head_lr=cfg.head_lr, brier_weight=cfg.brier, seed=cfg.seed, log_every=50, on_epoch_end=monitor)
    result["timing_lora_s"] = round(time.time() - t0, 1)
    result["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    result["lora_params"] = n_lora
    result["lora_train_history"] = hist
    result["quick_val_history"] = quick_hist
    log(f"LoRA trained in {result['timing_lora_s']}s, peak GPU {result['peak_gpu_gb']} GB, {n_lora:,} trainable parameters")

    val_lora = extract(core, renderer, val_requests, pad, hidden, EXTRACT_BATCH)
    lora = {"val": suite(head_l, val_lora, qids=qids)}
    T_lora = lora["val"]["temperature"]
    lora["controls"] = {}
    if dataset == "shapes":
        lora["controls"]["swapped"] = suite(head_l, extract(core, renderer, build_requests(with_next_state(val_records), 4, 2), pad, hidden, EXTRACT_BATCH), temperature=T_lora)
        lora["controls"]["no_image"] = suite(head_l, extract(core, renderer, build_requests(without_images(val_records), 4, 2), pad, hidden, EXTRACT_BATCH), temperature=T_lora)
    if cfg.test:
        test_lora = extract(core, renderer, build_requests(test_records, 4, 3), pad, hidden, EXTRACT_BATCH)
        lora["test"] = suite(head_l, test_lora, temperature=T_lora, qids=qids)
    result["lora"] = lora
    log(f"LoRA: val acc {lora['val']['acc']:.3f} (frozen {frozen['val']['acc']:.3f}), stability {lora['val']['stability']:.3f}")

    if not cfg.no_save and not cfg.smoke:
        save_run(ROOT / "runs" / ("lora_text" if dataset == "text" else "vision") / cfg.tag, peft_model, head_l)
    result["total_s"] = round(result["timing_extract_s"] + result["timing_head_s"] + result["timing_lora_s"], 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ("smoke_" + result_path.name if cfg.smoke else result_path.name)
    path.write_text(json.dumps(result, indent=1))
    log(f"wrote {path.relative_to(ROOT)}")
    return result
