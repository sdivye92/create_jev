"""Command line: python -m jev {make-shapes,train,evaluate,predict} ..."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def _make_shapes(a):
    from .shapes import generate_dataset

    print(generate_dataset(a.out, {"train": a.train, "val": a.val, "test": a.test}, seed=a.seed))


def _train(a):
    from .data import load_jsonl
    from .pipeline import fit

    image_root = a.image_root or Path(a.train).parent
    metrics = fit(load_jsonl(a.train)[: a.max_train_records], load_jsonl(a.val)[: a.max_val_records], a.out, image_root=image_root,
                  model_id=a.model_id, n_orders=a.n_orders, head_epochs=a.head_epochs, head_lr=a.head_lr, lora=a.lora,
                  lora_epochs=a.lora_epochs, lora_batch=a.lora_batch, rank=a.rank, alpha=a.alpha, brier_weight=a.brier_weight, seed=a.seed)
    print(json.dumps({k: metrics[k] for k in ("acc", "chance", "stability", "temperature")}, indent=2))


def _evaluate(a):
    from .pipeline import evaluate_file

    print(json.dumps(evaluate_file(a.model, a.data, a.image_root, a.n_orders, max_records=a.max_records), indent=2))


def _predict(a):
    from .pipeline import JevPredictor

    request = json.loads(Path(a.request).read_text())
    predictor = JevPredictor.from_pretrained(a.model, a.image_root or Path(a.request).resolve().parent)
    print(json.dumps(predictor.predict(request["state"], request["questions"]), indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="jev", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("make-shapes", help="generate the synthetic image dataset")
    p.add_argument("--out", default="data/shapes")
    p.add_argument("--train", type=int, default=800)
    p.add_argument("--val", type=int, default=200)
    p.add_argument("--test", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_make_shapes)

    p = sub.add_parser("train", help="train a head on the frozen backbone, optionally followed by LoRA")
    p.add_argument("--train", required=True, help="training JSONL")
    p.add_argument("--val", required=True, help="validation JSONL (also used to fit the temperature)")
    p.add_argument("--out", required=True, help="output directory for the model")
    p.add_argument("--image-root", help="folder that image paths are relative to (default: the folder of the training file)")
    p.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--max-train-records", type=int)
    p.add_argument("--max-val-records", type=int)
    p.add_argument("--n-orders", type=int, default=6, help="option orders per training question for the frozen head")
    p.add_argument("--head-epochs", type=int, default=30)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--lora", action="store_true", help="after the frozen head, train LoRA adapters and the head together")
    p.add_argument("--lora-epochs", type=int, default=2)
    p.add_argument("--lora-batch", type=int, default=16, help="lower this if you run out of GPU memory")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--brier-weight", type=float, default=0.0, help="weight of the Brier term added to the cross-entropy loss")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_train)

    p = sub.add_parser("evaluate", help="accuracy, order stability and calibration on a labelled JSONL")
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--image-root")
    p.add_argument("--n-orders", type=int, default=4)
    p.add_argument("--max-records", type=int)
    p.set_defaults(func=_evaluate)

    p = sub.add_parser("predict", help="answer the questions in a request JSON file")
    p.add_argument("--model", required=True)
    p.add_argument("--request", required=True, help='JSON file: {"state": ..., "questions": [...]}')
    p.add_argument("--image-root", help="folder that image paths are relative to (default: the folder of the request file)")
    p.set_defaults(func=_predict)

    args = parser.parse_args(argv)
    args.func(args)
