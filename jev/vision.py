"""Rendering for states that contain images: an image part becomes the model's own image tokens plus pixel data."""
from pathlib import Path

import torch
from PIL import Image

from .render import DELIMITERS, Renderer

IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


class MMRenderer(Renderer):
    """Like Renderer, but the state may be a list of parts: {"type": "text", "text": ...} or {"type": "image", "path": ...}.

    An image becomes the model's own image tokens (vision_start, N image_pad tokens, vision_end) where the processor decides N.
    The pixel data travels alongside the token ids in a fourth element, which features.encode passes to the backbone.
    """

    def __init__(self, processor, image_root, delimiters=DELIMITERS):
        super().__init__(processor.tokenizer, delimiters)
        self.processor = processor
        self.image_root = Path(image_root)
        self._images = {}

    def _image_block(self, path):
        if path not in self._images:
            location = self.image_root / path
            if not location.exists():
                raise FileNotFoundError(f"image {str(path)!r} not found at {location}. Image paths are relative to the image root; pass --image-root, or use absolute paths.")
            image = Image.open(location).convert("RGB")
            out = self.processor(text=[IMAGE_PLACEHOLDER], images=[image], return_tensors="pt")
            self._images[path] = (out["input_ids"][0].tolist(), out["mm_token_type_ids"][0].tolist(),
                                  out["pixel_values"].to(torch.bfloat16), out["image_grid_thw"])
        return self._images[path]

    def render_question(self, record, qi):
        state = record["state"]
        parts = state if isinstance(state, list) else [{"type": "text", "text": state}]
        q = record["questions"][qi]
        ids, types, pixels, grids, ends = [], [], [], [], []

        def add_text_ids(new):
            ids.extend(new)
            types.extend([0] * len(new))

        add_text_ids(self.delim("state"))
        for part in parts:
            if part["type"] == "text":
                add_text_ids(self.tok_user(part["text"]))
            elif part["type"] == "image":
                block_ids, block_types, pix, grid = self._image_block(part["path"])
                ids.extend(block_ids)
                types.extend(block_types)
                pixels.append(pix)
                grids.append(grid)
            else:
                raise ValueError(f"unknown state part type {part['type']!r}")
        add_text_ids(self.delim("q"))
        add_text_ids(self.tok_user(q.get("prompt", q["id"])))
        for opt in q["options"]:
            add_text_ids(self.delim("opt"))
            add_text_ids(self.tok_user(opt))
            add_text_ids(self.delim("opt_end"))
            ends.append(len(ids) - 1)
        add_text_ids(self.delim("decide"))
        return ids, ends, len(ids) - 1, {"types": types, "pixels": pixels, "grids": grids}
