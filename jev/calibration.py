"""Probability-quality metrics (NLL, Brier, ECE) and temperature scaling."""
import torch
import torch.nn.functional as F


def probabilities(scores, mask):
    """Softmax over the real options only. Padded slots get probability exactly 0."""
    return scores.masked_fill(~mask, float("-inf")).softmax(-1)


def nll(probs, labels):
    return -torch.log(probs.gather(1, labels[:, None]).squeeze(1).clamp_min(1e-12)).mean()


def brier_score(probs, labels, mask):
    """Mean over examples of the sum over options of (p_k - y_k)^2, where y is the one-hot true answer.

    0 is perfect, and a model that always says "uniform over N options" scores 1 - 1/N. Works on tensors that
    require grad, so it can also be used as a loss.
    """
    onehot = F.one_hot(labels, probs.shape[-1]).to(probs.dtype)
    return (((probs - onehot) ** 2) * mask).sum(-1).mean()


def uniform_brier(mask):
    """The Brier score of always answering with equal probability on every option."""
    return (1 - 1 / mask.sum(-1)).float().mean()


def expected_calibration_error(probs, labels, n_bins=10):
    """Bin examples by top-option confidence; ECE is the average gap between confidence and accuracy, weighted by bin size."""
    conf, pred = probs.max(-1)
    correct = (pred == labels).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    ece, table = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        n = int(in_bin.sum())
        if n == 0:
            table.append({"range": (lo.item(), hi.item()), "n": 0, "confidence": None, "accuracy": None})
            continue
        c, a = conf[in_bin].mean().item(), correct[in_bin].mean().item()
        ece += n / len(labels) * abs(c - a)
        table.append({"range": (lo.item(), hi.item()), "n": n, "confidence": c, "accuracy": a})
    return ece, table


def fit_temperature(scores, mask, labels, max_iter=100):
    """Find T > 0 minimising the NLL of softmax(scores / T) on this set. One number, fitted on held-out data."""
    scores = scores.masked_fill(~mask, 0.0)  # padded slots hold -inf, and -inf / T has a NaN gradient; they are masked out again below
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.5, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = nll(probabilities(scores / log_t.exp(), mask), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return min(max(log_t.exp().item(), 0.05), 20.0)  # a limit is only a sanity guard: hitting one means the scores carry almost no usable signal


def calibration_report(scores, mask, labels, temperature=1.0):
    probs = probabilities(scores / temperature, mask)
    ece, _ = expected_calibration_error(probs, labels)
    return {
        "acc": (probs.argmax(-1) == labels).float().mean().item(),
        "nll": nll(probs, labels).item(),
        "brier": brier_score(probs, labels, mask).item(),
        "ece": ece,
    }


def format_reliability(table):
    lines = [f"{'confidence bin':16s} {'n':>6s} {'mean conf':>10s} {'accuracy':>9s}"]
    for row in table:
        lo, hi = row["range"]
        if row["n"] == 0:
            lines.append(f"{lo:.1f} - {hi:.1f}        {0:6d} {'-':>10s} {'-':>9s}")
        else:
            lines.append(f"{lo:.1f} - {hi:.1f}        {row['n']:6d} {row['confidence']:10.3f} {row['accuracy']:9.3f}")
    return "\n".join(lines)
