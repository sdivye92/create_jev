"""Record schema helpers: convert an external format to ours, option reordering with the label moved, JSONL loading."""
import json


def render_state(state, indent=0):
    """Flatten a string, dict or list state into plain text."""
    pad = "  " * indent
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        lines = []
        for key, value in state.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.append(render_state(value, indent + 1))
            else:
                lines.append(f"{pad}{key}: {value}")
        return "\n".join(lines)
    if isinstance(state, list):
        lines = []
        for item in state:
            if isinstance(item, dict) and set(item) == {"role", "content"}:
                lines.append(f"{pad}{item['role']}: {item['content']}")
            elif isinstance(item, (dict, list)):
                lines.append(f"{pad}-")
                lines.append(render_state(item, indent + 1))
            else:
                lines.append(f"{pad}- {item}")
        return "\n".join(lines)
    return str(state)


def render_prompt(instructions):
    """Instructions are a string, or a dict such as {"question": ..., "focus": ...}."""
    if isinstance(instructions, str):
        return instructions
    if isinstance(instructions, dict):
        return " ".join(str(v) for v in instructions.values())
    return str(instructions)


def with_description(key, description):
    return key if not description else f"{key}: {description}"


def convert_question(qid, q):
    kind = q["type"]
    prompt = render_prompt(q["instructions"])
    if kind == "choice":
        keys = list(q["criteria"])
        options = [with_description(k, q["criteria"][k]) for k in keys]
        label = keys.index(q["label"])
    elif kind == "noul":
        criteria = q.get("criteria") or {}
        options = [with_description("no", criteria.get("false")), with_description("yes", criteria.get("true"))]
        label = 1 if q["label"] else 0
    elif kind == "score":
        options = list(q["criteria"])
        label = q["label"]
    else:
        raise ValueError(f"unknown question type {kind!r}")
    out = {"id": qid, "prompt": prompt, "type": kind, "options": options, "label": label}
    validate_question(out)
    return out


def validate_question(q):
    n = len(q["options"])
    if n < 2:
        raise ValueError(f"question {q['id']!r} has fewer than 2 options")
    if len(set(q["options"])) != n:
        raise ValueError(f"question {q['id']!r} has duplicate options: {q['options']}")
    if not isinstance(q["label"], int) or isinstance(q["label"], bool) or not 0 <= q["label"] < n:
        raise ValueError(f"question {q['id']!r} has label {q['label']!r} outside 0..{n - 1}")
    if any(not o.strip() for o in q["options"]):
        raise ValueError(f"question {q['id']!r} has an empty option")


def convert_record(rec):
    return {
        "state": render_state(rec["state"]),
        "questions": [convert_question(qid, q) for qid, q in rec["questions"].items()],
    }


def convert_file(src, dst):
    count = 0
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, 1):
            try:
                converted = convert_record(json.loads(line))
            except (ValueError, KeyError) as e:
                raise ValueError(f"{src}:{line_no}: {e}") from e
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def random_order(n, rng):
    return rng.sample(range(n), n)


def reorder_question(q, order):
    """Present the options as q["options"][order[0]], q["options"][order[1]], ... and move the label with its option."""
    if q["type"] == "score":
        return q
    return {**q, "options": [q["options"][i] for i in order], "label": order.index(q["label"])}
