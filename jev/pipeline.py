"""End-to-end training, saving, loading and prediction for a jev decision model.

A trained model is a directory containing config.json, head.pt, metrics.json and, when LoRA was used, adapter/.
"""
import json
import time
from pathlib import Path

import torch

from .calibration import calibration_report, fit_temperature, probabilities
from .data import load_jsonl
from .features import build_requests, encode, extract, render_requests
from .head import MaskedPointerHead
from .train import add_lora, chance_accuracy, evaluate, predict_scores, train_head, train_lora
from .vision import MMRenderer

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-0.8B"
EXTRACT_BATCH = 32


def load_backbone(model_id=DEFAULT_MODEL_ID, device_map="auto"):
    """Load the vision-language model, freeze it, and return (processor, core, hidden_size). core is the part that makes hidden vectors."""
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForMultimodalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device_map).eval()
    core = model.model
    for p in core.parameters():
        p.requires_grad_(False)
    return processor, core, model.config.text_config.hidden_size


def fit(train_records, val_records, out_dir, image_root=".", model_id=DEFAULT_MODEL_ID, n_orders=6, head_epochs=30, head_lr=1e-3,
        lora=False, lora_epochs=2, lora_batch=16, rank=8, alpha=16, brier_weight=0.0, seed=0, log=print):
    """Train a pointer head on the frozen backbone, optionally followed by LoRA on the language layers, then fit a temperature on val.

    Returns the validation metrics and writes the model directory to out_dir.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    processor, core, hidden = load_backbone(model_id)
    pad = processor.tokenizer.pad_token_id
    device = next(core.parameters()).device
    renderer = MMRenderer(processor, image_root)

    t0 = time.time()
    val_requests = build_requests(val_records, n_orders=4, seed=seed + 2)
    train_fs = extract(core, renderer, build_requests(train_records, n_orders=n_orders, seed=seed + 1), pad, hidden, EXTRACT_BATCH)
    val_fs = extract(core, renderer, val_requests, pad, hidden, EXTRACT_BATCH)
    log(f"cached {len(train_fs)} train and {len(val_fs)} val examples in {time.time() - t0:.0f}s")

    head, _ = train_head(train_fs, val_fs, hidden, epochs=head_epochs, lr=head_lr, seed=seed, brier_weight=brier_weight,
                         log_every=max(head_epochs // 5, 1))
    log("frozen-backbone head: val accuracy %.3f" % evaluate(head, val_fs)["acc"])

    if lora:
        peft_model = add_lora(core, rank=rank, alpha=alpha)
        lora_head = MaskedPointerHead(hidden, d_head=head.opt_proj.out_features, normalize=head.normalize).to(device)
        lora_head.load_state_dict(head.state_dict())
        train_lora(core, lora_head, renderer, train_records, pad, epochs=lora_epochs, batch_size=lora_batch, brier_weight=brier_weight,
                   seed=seed, log_every=25)
        peft_model.save_pretrained(out / "adapter")
        head = lora_head
        val_fs = extract(core, renderer, val_requests, pad, hidden, EXTRACT_BATCH)
        log("LoRA backbone + head: val accuracy %.3f" % evaluate(head, val_fs)["acc"])

    scores, mask, labels = predict_scores(head, val_fs), val_fs.mask.cpu(), val_fs.labels.cpu()
    temperature = fit_temperature(scores, mask, labels)
    metrics = evaluate(head, val_fs)
    metrics["chance"] = chance_accuracy(val_fs)
    metrics["calibration"] = calibration_report(scores, mask, labels)
    metrics["calibration_temperature_scaled"] = calibration_report(scores, mask, labels, temperature)
    metrics["temperature"] = temperature

    torch.save(head.state_dict(), out / "head.pt")
    config = {"model_id": model_id, "hidden_size": hidden, "d_head": head.opt_proj.out_features, "normalize": head.normalize,
              "lora": lora, "rank": rank, "alpha": alpha, "temperature": temperature}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log(f"saved model to {out} (temperature {temperature:.2f})")
    return metrics


class JevPredictor:
    """A trained decision model. predict() answers typed questions about a state (text, images, or both) with one forward pass per question."""

    def __init__(self, processor, core, head, temperature=1.0, image_root="."):
        self.core, self.head, self.temperature = core, head.eval(), temperature
        self.pad = processor.tokenizer.pad_token_id
        self.hidden = head.opt_proj.in_features
        self.renderer = MMRenderer(processor, image_root)

    @classmethod
    def from_pretrained(cls, path, image_root=".", device_map="auto"):
        path = Path(path)
        cfg = json.loads((path / "config.json").read_text())
        processor, core, hidden = load_backbone(cfg["model_id"], device_map)
        if cfg["lora"]:
            from peft import PeftModel

            PeftModel.from_pretrained(core, path / "adapter")
        device = next(core.parameters()).device
        head = MaskedPointerHead(hidden, d_head=cfg["d_head"], normalize=cfg["normalize"]).to(device)
        head.load_state_dict(torch.load(path / "head.pt", map_location=device))
        return cls(processor, core, head, cfg["temperature"], image_root)

    @torch.no_grad()
    def predict(self, state, questions, batch_size=16):
        """state: a string, or a list of {"type": "text", "text": ...} / {"type": "image", "path": ...} parts.

        questions: a list of {"id", "type", "options", optionally "prompt"}. Returns one dict per question.
        """
        requests = [{"state": state, "question": {**q, "label": 0}, "type": q["type"], "qid": q["id"], "rec": 0} for q in questions]
        rendered = render_requests(self.renderer, requests)
        results = []
        for start in range(0, len(requests), batch_size):
            idx = list(range(start, min(start + batch_size, len(requests))))
            opt, dec, mask = encode(self.core, rendered, idx, self.pad, max(len(rendered[i][1]) for i in idx))
            scores = self.head(opt.float(), dec.float(), mask).float().cpu()
            probs = probabilities(scores / self.temperature, mask.cpu())
            for row, i in enumerate(idx):
                q = questions[i]
                p = probs[row, : len(q["options"])].tolist()
                best = max(range(len(p)), key=p.__getitem__)
                result = {"id": q["id"], "type": q["type"], "options": q["options"], "probabilities": p,
                          "prediction": q["options"][best], "confidence": p[best]}
                if q["type"] == "score":
                    result["expected_level"] = sum(k * pk for k, pk in enumerate(p))
                results.append(result)
        return results


def evaluate_file(model_dir, data_path, image_root=None, n_orders=4, seed=3, max_records=None):
    """Evaluate a saved model on a labelled JSONL file. Each question is shown in n_orders option orders."""
    records = load_jsonl(data_path)[:max_records]
    predictor = JevPredictor.from_pretrained(model_dir, image_root or Path(data_path).parent)
    fs = extract(predictor.core, predictor.renderer, build_requests(records, n_orders, seed), predictor.pad, predictor.hidden, EXTRACT_BATCH)
    result = evaluate(predictor.head, fs)
    scores, mask, labels = predict_scores(predictor.head, fs), fs.mask.cpu(), fs.labels.cpu()
    result["chance"] = chance_accuracy(fs)
    result["calibration"] = calibration_report(scores, mask, labels)
    result["calibration_temperature_scaled"] = calibration_report(scores, mask, labels, predictor.temperature)
    result["temperature"] = predictor.temperature
    return result
