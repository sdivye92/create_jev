"""Token-level rendering of a (state, question) pair, with the marker tokens and user-text sanitising."""
import re

DELIMITERS = {
    "state": "<|fim_prefix|>",
    "q": "<|fim_middle|>",
    "opt": "<|box_start|>",
    "opt_end": "<|box_end|>",
    "decide": "<|fim_suffix|>",
}

_SPECIAL_RE = re.compile(r"<\|([^|<>]{1,64})\|>")


def sanitize_user_text(text):
    return _SPECIAL_RE.sub(lambda m: f"<¦{m.group(1)}¦>", text)


class Renderer:
    """Turns a (state, question) pair into token ids plus the positions the head reads."""

    def __init__(self, tokenizer, delimiters=DELIMITERS):
        self.tokenizer = tokenizer
        self.delimiters = delimiters
        self.delim_ids = {self._ids(s)[0] for s in delimiters.values()}

    def _ids(self, text):
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def delim(self, name):
        return self._ids(self.delimiters[name])

    def tok_user(self, text):
        return [i for i in self._ids(sanitize_user_text(text)) if i not in self.delim_ids]

    def render_question(self, record, qi):
        q = record["questions"][qi]
        ids, opt_end_positions = [], []
        ids += self.delim("state")
        ids += self.tok_user(record["state"])
        ids += self.delim("q")
        ids += self.tok_user(q.get("prompt", q["id"]))
        for opt in q["options"]:
            ids += self.delim("opt")
            ids += self.tok_user(opt)
            ids += self.delim("opt_end")
            opt_end_positions.append(len(ids) - 1)
        ids += self.delim("decide")
        return ids, opt_end_positions, len(ids) - 1
