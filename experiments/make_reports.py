"""Build results/RESULTS_LORA.md and results/RESULTS_VISION.md from the JSON files in experiments/results/. No number in the reports is typed by hand.

Hand-written text lives next to the runners: best.md (the best configuration, placed at the top) and findings.md (commentary, appended at the end), in experiments/lora/ and experiments/vision/.
Usage: python experiments/make_reports.py
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "lora": {"file": "RESULTS_LORA.md", "dataset": "synthetic text tickets", "questions": ["department", "refund", "tone", "urgency"]},
    "vision": {"file": "RESULTS_VISION.md", "dataset": "synthetic shapes images", "questions": ["largest_color", "has_triangle", "count", "leftmost"]},
}


def load(group):
    return {p.stem: json.loads(p.read_text()) for p in sorted((ROOT / "experiments" / "results" / group).glob("*.json"))}


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def ms(vals, nd=3):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "n/a"
    return f"{statistics.mean(vals):.{nd}f}" + (f" +- {statistics.pstdev(vals):.{nd}f}" if len(vals) > 1 else "")


def get(r, *path):
    for p in path:
        if r is None or p not in r:
            return None
        r = r[p]
    return r


def seeds(res, prefix):
    return [res[f"{prefix}_seed{s}"] for s in range(5) if f"{prefix}_seed{s}" in res]


def variants(res):
    return [("Frozen backbone, cross-entropy", "main_crossentropy", "frozen"), ("LoRA, cross-entropy", "main_crossentropy", "lora"),
            ("Frozen backbone, cross-entropy + Brier", "main_brier", "frozen"), ("LoRA, cross-entropy + Brier", "main_brier", "lora")]


def glossary(group, cfg):
    text = group == "lora"
    orders = 6 if text else 4
    out = []
    out.append("## 0. Before the tables: what was run, how runs are named, and what the columns mean\n")
    out.append("### What one run is\n")
    out.append(f"A run is one complete experiment on the {cfg['dataset']}. It first trains a pointer head on top of the frozen backbone, which is the starting point and the baseline (called frozen). It then trains LoRA adapters on the language layers together with the head, and that is the adapted model (called LoRA). Both stages are evaluated on the same validation records in the same run, so every frozen versus LoRA comparison in this file is paired.\n")
    out.append(f"Training uses 800 records. Each record has several questions, and every question is shown to the model in {orders} different random option orders during training, because the head must not depend on the order in which options are listed. Validation and test use 4 fixed option orders per question, and the same records and orders in every run, so numbers are comparable between runs.\n")
    out.append("### Run names\n")
    out.append("Every result file is named `<what changes>_seed<N>`. For example `main_brier_seed1` is the run with the Brier term in the loss, using random seed 1.\n")
    out.append(table(["Part of the name", "Meaning"], [
        ["main_crossentropy", "The standard recipe. The loss is cross-entropy only. This is the reference run that the others are compared with."],
        ["main_brier", "The standard recipe with a Brier term added to the loss (loss = cross-entropy + 1.0 x Brier). It tests whether training directly on the Brier score improves probability quality."],
        ["rank4, rank16", "LoRA rank 4 or 16 instead of 8. The rank sets how many numbers each adapter has. The scale alpha is set to twice the rank, so the strength of the correction stays comparable."],
        ["lr1e-4, lr5e-4", "LoRA learning rate 0.0001 or 0.0005 instead of the default 0.0002."],
        ["epochs4", "4 passes over the training data instead of 2."],
        ["size100, size200, size400", "Train on only the first 100, 200 or 400 of the 800 training records. Used for the learning curve."],
        ["seed0, seed1, seed2", "The random seed, explained next."]]))
    out.append("")
    out.append("### Seeds\n")
    out.append("Training involves random choices: the starting values of the head, the order in which batches are drawn, and which random option orders are generated for the training questions. A seed is a number that fixes all of those choices. Running the same settings with the same seed gives the same result (up to tiny GPU rounding differences), and running with a different seed repeats the experiment with different random luck.\n")
    out.append("Why bother: a single run can look good or bad by chance, and one number cannot tell you which. Running the three main configurations with seeds 0, 1 and 2 gives three results for the same recipe, and the spread between them (the number after +-) shows how much of any difference is just luck. A difference between two variants that is smaller than that spread should not be trusted. The same seed numbers are used for the cross-entropy and Brier runs, so `main_crossentropy_seed1` and `main_brier_seed1` start from the same random draw and differ only in the loss.\n")
    out.append("Seeds do not change the records, the validation and test sets, or the validation option orders. Those are fixed, so the +- values measure training randomness only, not the variation you would get from a different dataset. The ablations and the learning curve use seed 0 only, so each of those numbers is a single draw and differences of about 0.005 in accuracy are within the noise.\n")
    out.append("### What the columns mean\n")
    out.append(table(["Column", "Meaning", "Better is"], [
        ["Accuracy", "Share of (question, option order) examples where the option with the highest probability is the true one. The same question in 4 orders counts as 4 examples.", "higher"],
        ["Chance", "The accuracy of guessing. It is the average of 1 / (number of options) over the questions.", "-"],
        ["Order stability", "Share of choice and yes/no questions for which the model picks the same option text in every option order. A model can be accurate on average and still change its answer when the options are reshuffled, and this column catches that. Score questions are never shuffled and are excluded.", "higher"],
        ["NLL", "Negative log-likelihood: the average of -log(probability given to the true option). It punishes confident wrong answers heavily. This is also the training loss.", "lower"],
        ["Brier", "Mean squared error between the probabilities and the truth (1 for the true option, 0 for the others), summed over the options. It is bounded (the worst possible value is 2) and reads as a plain error rate for probabilities. A model that spreads probability equally over N options scores 1 - 1/N.", "lower"],
        ["ECE", "Expected calibration error: predictions are grouped by confidence, and ECE is the average gap between confidence and actual accuracy in each group, weighted by group size. 0 means that when the model says 80% it is right 80% of the time.", "lower"],
        ["T", "The temperature: one number fitted on the validation set that divides all scores before the softmax. T above 1 flattens the probabilities, so the model was overconfident. T below 1 sharpens them, so it was underconfident. T never changes which option wins, so accuracy is unaffected.", "closer to 1 means less correction needed"],
        ["NLL (T), Brier (T), ECE (T)", "The same three measures after applying the temperature T. They show how good the probabilities become once the cheap post-hoc fix is applied, and so how much of the calibration problem was left over. On validation these are slightly optimistic, because T was fitted on that same data. On test, T from validation is used, so they are honest.", "lower"],
        ["+- (after a value)", "Population standard deviation over the seeds that were run. It is a rough guide to run-to-run noise, based on only three runs.", "-"]]))
    out.append("")
    return "\n".join(out) + "\n"


def report(group, res):
    cfg = GROUPS[group]
    qs = cfg["questions"]
    L = [f"# Results: {'LoRA on text tickets' if group == 'lora' else 'vision (images as input)'}\n"]
    n_runs = len(res)
    L.append(f"Measured on the {cfg['dataset']}, with a Qwen3.5-0.8B backbone and a pointer head. {n_runs} runs, generated from `experiments/results/{group}/*.json` by `experiments/make_reports.py`. "
             "Each run first trains a head on the frozen backbone (the starting point), then trains LoRA adapters on the language layers together with the head. Both stages are reported from the same run, so comparisons are paired.\n")
    best = ROOT / "experiments" / group / "best.md"
    if best.exists():
        L.append(best.read_text().strip() + "\n")
    L.append(glossary(group, cfg))

    # 1 main comparison
    L.append("## 1. Main comparison (validation)\n")
    L.append("**What:** the central experiment. Four variants are compared: the frozen backbone and the LoRA-adapted backbone, each trained with cross-entropy alone and with cross-entropy plus the Brier term. Each variant is run with 3 seeds.\n")
    L.append("**Why:** it answers the two main questions. Does adapting the backbone beat reading it frozen, and does adding Brier to the loss change the probability quality?\n")
    L.append("**How to read:** compare rows in the same column. Accuracy and order stability should go up, NLL, Brier and ECE should go down. Check that a difference between two rows is larger than the +- spread before believing it. The columns marked (T) show the same measures after temperature scaling, and T itself tells you how overconfident the raw probabilities were. Validation values are the ones used to choose settings, so they are for comparing variants, and the test table below is the one to quote.\n")
    rows = []
    for name, prefix, stage in variants(res):
        rs = seeds(res, prefix)
        if not rs:
            continue
        g = lambda *p: [get(r, stage, "val", *p) for r in rs]
        rows.append([name, len(rs), ms(g("acc")), ms(g("stability")), ms(g("cal", "nll")), ms(g("cal", "brier")), ms(g("cal", "ece")),
                     ms(g("cal_scaled", "nll")), ms(g("cal_scaled", "brier")), ms(g("cal_scaled", "ece")), ms(g("temperature"), 2)])
    L.append(table(["Variant", "Seeds", "Accuracy", "Order stability", "NLL", "Brier", "ECE", "NLL (T)", "Brier (T)", "ECE (T)", "T"], rows))
    ch = [get(r, "frozen", "val", "chance") for r in seeds(res, "main_crossentropy")]
    L.append(f"\nChance accuracy on validation: {ms(ch)}.\n")

    L.append("### Accuracy by question (validation)\n")
    L.append("**Why:** the overall accuracy averages all questions, and the questions differ a lot in difficulty. This table shows which questions the gain comes from and which are already solved. Each cell is the mean accuracy over the seeds for that question.\n")
    rows = []
    for name, prefix, stage in variants(res):
        rs = seeds(res, prefix)
        if rs:
            rows.append([name] + [ms([get(r, stage, "val", "by_question", q) for r in rs]) for q in qs])
    L.append(table(["Variant"] + qs, rows))

    # 2 test
    L.append("\n## 2. Test set (used once per run)\n")
    L.append("**What:** the same four variants on records that were never used for training, for choosing settings, or for fitting T.\n")
    L.append("**Why:** validation was looked at while making decisions, so it is slightly flattered. The test set is touched once per main run, after everything is decided, and is the number to quote as the expected performance. If the test numbers are close to the validation numbers, the choices did not overfit validation.\n")
    L.append("**How to read:** the columns are as before, with the per-question accuracy at the end. The (T) columns use the temperature fitted on validation, which is how it would be used in practice.\n")
    rows = []
    for name, prefix, stage in variants(res):
        rs = [r for r in seeds(res, prefix) if get(r, stage, "test")]
        if not rs:
            continue
        g = lambda *p: [get(r, stage, "test", *p) for r in rs]
        rows.append([name, len(rs), ms(g("acc")), ms(g("stability")), ms(g("cal", "nll")), ms(g("cal", "brier")), ms(g("cal", "ece")),
                     ms(g("cal_scaled", "nll")), ms(g("cal_scaled", "brier")), ms(g("cal_scaled", "ece"))] + [ms([get(r, stage, "test", "by_question", q) for r in rs]) for q in qs])
    L.append(table(["Variant", "Seeds", "Accuracy", "Order stability", "NLL", "Brier", "ECE", "NLL (T)", "Brier (T)", "ECE (T)"] + qs, rows))
    L.append("\n(T) = with the temperature fitted on validation.\n")

    # 3 cost
    L.append("## 3. Training cost\n")
    L.append("**Why:** the gain from LoRA has to be weighed against what it costs. The frozen backbone is read once and the features are cached, so the head trains in seconds. LoRA changes the backbone, so nothing can be cached, and every epoch needs a full forward and backward pass, which is why gradient checkpointing is used to fit in 8 GB.\n")
    rs = seeds(res, "main_crossentropy") + seeds(res, "main_brier")
    if rs:
        L.append(table(["Quantity", "Value (mean over main runs)"], [
            ["Trainable LoRA parameters (rank 8)", f"{rs[0]['lora_params']:,}"],
            ["Frozen features: extraction time (near zero when the cache was reused, so the mean understates it)", f"{ms([r['timing_extract_s'] for r in rs], 0)} s"],
            ["Frozen head training", f"{ms([r['timing_head_s'] for r in rs], 1)} s"],
            ["LoRA training (2 epochs)", f"{ms([r['timing_lora_s'] for r in rs], 0)} s"],
            ["Peak GPU memory during LoRA training", f"{ms([r['peak_gpu_gb'] for r in rs], 2)} GB"]]))
        L.append("")

    # 4 epochs
    L.append("## 4. Learning during LoRA training\n")
    L.append("**What:** the validation accuracy measured after each epoch of LoRA training, for the default run (2 epochs) and for a run with 4 epochs.\n")
    L.append("**Why:** it shows whether training has converged, and whether the extra epochs of the 4-epoch run still help. The last column puts it next to the frozen head and the final full-validation number.\n")
    rows = []
    for tag in ["main_crossentropy_seed0", "epochs4_seed0"]:
        if tag in res:
            h = res[tag]["quick_val_history"]
            rows.append([tag, ", ".join(f"{x['quick_val_acc']:.3f}" for x in h), f"{res[tag]['frozen']['val']['acc']:.3f}", f"{res[tag]['lora']['val']['acc']:.3f}"])
    if rows:
        L.append("Quick validation accuracy (first 60 validation records, 2 orders) after each epoch.\n")
        L.append(table(["Run", "Quick val accuracy after epoch 1, 2, ...", "Frozen head (full val)", "LoRA (full val)"], rows))
        L.append("")

    # 5 ablations
    L.append("## 5. Ablations (one seed, validation)\n")
    L.append("**What:** the default LoRA recipe, changed one setting at a time: the rank (size of the adapters), the LoRA learning rate, and the number of epochs.\n")
    L.append("**Why:** to see which settings matter and whether the default is a reasonable choice. Rank trades capacity for parameters, the learning rate trades speed for stability, and epochs trade accuracy for time.\n")
    L.append("**How to read:** every row is one run with seed 0, so differences of about 0.005 in accuracy are inside the noise (see the seed section). Look for consistent patterns across several columns rather than a single best number. The LoRA parameters and time columns give the price of each setting. These runs are scored on validation only, so a setting picked from here has not been confirmed on test.\n")
    ab = [("main_crossentropy_seed0", "Default: rank 8, alpha 16, LoRA lr 2e-4, 2 epochs"), ("rank4_seed0", "Rank 4 (alpha 8)"), ("rank16_seed0", "Rank 16 (alpha 32)"),
          ("lr1e-4_seed0", "LoRA lr 1e-4"), ("lr5e-4_seed0", "LoRA lr 5e-4"), ("epochs4_seed0", "4 epochs")]
    rows = []
    for tag, name in ab:
        if tag in res:
            r = res[tag]
            rows.append([name, f"{r['lora_params']:,}", f"{r['lora']['val']['acc']:.3f}", f"{r['lora']['val']['stability']:.3f}", f"{r['lora']['val']['cal']['nll']:.3f}",
                         f"{r['lora']['val']['cal']['brier']:.3f}", f"{r['lora']['val']['cal']['ece']:.3f}", f"{r['timing_lora_s']:.0f}"])
    if rows:
        L.append(table(["Setting", "LoRA parameters", "Accuracy", "Order stability", "NLL", "Brier", "ECE", "LoRA time (s)"], rows))
        L.append("")

    # 6 data size
    L.append("## 6. Learning curve: frozen head versus LoRA by number of training records (seed 0, validation)\n")
    L.append("**What:** both the frozen head and LoRA are trained on only the first 100, 200, 400 or all 800 training records, and evaluated on the same validation set.\n")
    L.append("**Why:** it shows how much data each approach needs. LoRA has about 5 million trainable parameters against the head's much smaller number, so it may need more data before it pays off.\n")
    L.append("**How to read:** compare the frozen and LoRA columns in each row. Where LoRA falls below frozen, it is overfitting or under-trained for that amount of data. Note that the number of epochs is fixed, so fewer records also means fewer optimizer steps for both.\n")
    rows = []
    for tag, n in [("size100_seed0", 100), ("size200_seed0", 200), ("size400_seed0", 400), ("main_crossentropy_seed0", 800)]:
        if tag in res:
            r = res[tag]
            rows.append([n, f"{r['frozen']['val']['acc']:.3f}", f"{r['lora']['val']['acc']:.3f}", f"{r['frozen']['val']['stability']:.3f}", f"{r['lora']['val']['stability']:.3f}",
                         f"{r['frozen']['val']['cal']['brier']:.3f}", f"{r['lora']['val']['cal']['brier']:.3f}", f"{r['timing_lora_s']:.0f}"])
    if rows:
        L.append(table(["Training records", "Accuracy, frozen", "Accuracy, LoRA", "Stability, frozen", "Stability, LoRA", "Brier, frozen", "Brier, LoRA", "LoRA time (s)"], rows))
        L.append("")

    # 7 controls (vision)
    if group == "vision":
        L.append("## 7. Controls: is the model using the image? (validation)\n")
        L.append("**Why:** high accuracy on an image task proves little until you rule out that the model is answering from the text or from label frequencies alone. The controls change the picture and keep everything else, so the drop in accuracy measures how much the answers depend on the picture.\n")
        L.append("`swapped` gives every record the image and caption of the next record, keeping its own questions and labels. `no image` removes the image and keeps the caption. Accuracy should fall toward chance if the answers come from the picture. Values are mean over the main cross-entropy runs. How to read: real images is the normal score, and the other two rows should be near chance (shown in the next table). If a question stays well above chance under the controls, part of its accuracy comes from something other than the image, such as a label that is usually the same. Here has_triangle answers yes more often than no, so a model that ignores the image still scores above 0.5 on it.\n")
        rs = seeds(res, "main_crossentropy")
        rows = []
        if rs:
            for stage in ("frozen", "lora"):
                for ctrl in ("real", "swapped", "no_image"):
                    vals = {q: [get(r, stage, "val", "by_question", q) if ctrl == "real" else get(r, stage, "controls", ctrl, "by_question", q) for r in rs] for q in qs}
                    allv = [get(r, stage, "val", "acc") if ctrl == "real" else get(r, stage, "controls", ctrl, "acc") for r in rs]
                    rows.append([stage, {"real": "real images", "swapped": "swapped images", "no_image": "no image"}[ctrl], ms(allv)] + [ms(vals[q]) for q in qs])
            chance_q = {}
            fs = rs[0]
            L.append(table(["Backbone", "Condition", "Overall accuracy"] + qs, rows))
            L.append("\nOverall chance on validation: " + ms([get(r, 'frozen', 'val', 'chance') for r in rs]) + ".\n")

    # 8 per question chance + errors
    L.append(f"## {8 if group == 'vision' else 7}. Errors\n")
    rs = seeds(res, "main_crossentropy")
    if rs:
        L.append("**What:** for the hardest questions, the errors broken down by the true answer. It uses seed 0 and the validation set, and counts all option orders.\n")
        L.append("**Why:** aggregate accuracy hides which answers fail. These tables show which classes the model confuses and whether LoRA fixed them or moved the errors around.\n")
        L.append("**How to read:** error rate is the share of examples whose true answer is that row and where the top option was wrong. The note under each table separates two kinds of failure. Records with at least one wrong option order are sensitive to the order of the options. Records wrong in every order are genuinely hard for the model, because no ordering rescues them. A good change turns the first kind into correct answers, and cannot be expected to fix the second kind.\n")
        r = rs[0]
        for qid in r["frozen"]["val"].get("errors", {}):
            L.append(f"### {qid} (seed 0, validation): errors by true answer\n")
            rows = []
            fe, le = r["frozen"]["val"]["errors"][qid], r["lora"]["val"]["errors"][qid]
            keys = sorted(set(fe["confusion"]) | set(le["confusion"]))

            def rate(e, k):
                row = e["confusion"].get(k, {})
                n = sum(row.values())
                return (n, (n - row.get(k, 0)) / n if n else 0)

            for k in keys:
                nf, ef = rate(fe, k)
                nl, el = rate(le, k)
                rows.append([k, nf, f"{ef:.1%}", f"{el:.1%}"])
            L.append(table(["True answer", "Examples", "Error rate, frozen", "Error rate, LoRA"], rows))
            L.append(f"\nRecords with at least one wrong option order: frozen {fe['records_with_error']} of {fe['records']}, LoRA {le['records_with_error']} of {le['records']}. "
                     f"Wrong in every order: frozen {fe['records_always_wrong']}, LoRA {le['records_always_wrong']}.\n")

    # reliability
    stage_r = [r for r in seeds(res, "main_crossentropy") if get(r, "lora", "test")]
    if stage_r:
        L.append("## Reliability table (test, LoRA cross-entropy, seed 0, temperature from validation)\n")
        L.append("**What:** every test prediction is placed in a bin by the probability the model gave to its top option, and each bin reports how often that top option was correct.\n")
        L.append("**Why:** this is the table behind ECE. It shows directly whether a stated confidence can be trusted, which matters when downstream rules act on a threshold such as acting only above 90%.\n")
        L.append("**How to read:** in each row, mean confidence and accuracy should be about equal. Accuracy below confidence means overconfident, above means underconfident. Rows with only a few examples are noisy and should not be over-interpreted. The last row holds most predictions, so it dominates ECE.\n")
        rows = [[f"{b['range'][0]:.1f} to {b['range'][1]:.1f}", b["n"], "-" if b["n"] == 0 else f"{b['confidence']:.3f}", "-" if b["n"] == 0 else f"{b['accuracy']:.3f}"]
                for b in stage_r[0]["lora"]["test"]["reliability_scaled"]]
        L.append(table(["Confidence bin", "Examples", "Mean confidence", "Accuracy"], rows))
        L.append("")

    f = ROOT / "experiments" / group / "findings.md"
    if f.exists():
        L.append(f.read_text().strip() + "\n")
    L.append("## Reproduce\n")
    L.append(f"```\nexperiments/suite.sh {group}                 # all configurations, resumable\npython experiments/{group}/run.py --tag main_crossentropy_seed0 --seed 0 --test   # one configuration\npython experiments/make_reports.py            # rebuild this file\n```\n")
    (ROOT / "results" / cfg["file"]).write_text("\n".join(L))
    print("wrote", cfg["file"], "from", len(res), "runs")


if __name__ == "__main__":
    for g in GROUPS:
        res = load(g)
        if res:
            report(g, res)
