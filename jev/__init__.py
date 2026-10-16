"""jev: decision models on top of a vision-language backbone.

One forward pass per question returns a probability for every option, with no text generated. See jev/README.md.
The names below load lazily, so `import jev.render` stays cheap.
"""
import importlib

__version__ = "0.1.0"

_EXPORTS = {
    "Renderer": "render", "DELIMITERS": "render", "MMRenderer": "vision", "MaskedPointerHead": "head",
    "fit": "pipeline", "JevPredictor": "pipeline", "evaluate_file": "pipeline", "load_backbone": "pipeline",
    "generate_dataset": "shapes", "load_jsonl": "data", "convert_file": "data",
}


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(importlib.import_module(f".{_EXPORTS[name]}", __name__), name)
    raise AttributeError(f"module 'jev' has no attribute {name!r}")
