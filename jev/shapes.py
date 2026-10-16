"""Synthetic image dataset: scenes of coloured shapes with exact labels."""
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

COLORS = {
    "red": (220, 40, 40), "green": (40, 160, 70), "blue": (40, 80, 220),
    "yellow": (240, 210, 40), "purple": (140, 60, 180), "orange": (240, 140, 30),
}
SHAPES = ("circle", "square", "triangle")
COUNT_LEVELS = ["one shape", "two shapes", "three shapes", "four shapes"]
CAPTIONS = ["Photo uploaded by a user.", "Image attached to the ticket.", "Screenshot from the app.", "Picture sent from a phone."]
BACKGROUND = (236, 236, 236)


def _draw_shape(draw, shape, color, cx, cy, side):
    half = side / 2
    box = [cx - half, cy - half, cx + half, cy + half]
    if shape == "circle":
        draw.ellipse(box, fill=color)
    elif shape == "square":
        draw.rectangle(box, fill=color)
    else:
        draw.polygon([(cx, cy - half), (cx - half, cy + half), (cx + half, cy + half)], fill=color)


def make_scene(rng, size=256):
    """1-4 non-overlapping shapes of different sizes on a plain background. Returns (image, shapes)."""
    n = rng.randint(1, 4)
    while True:
        sides = sorted(rng.sample(range(36, 101, 8), n), reverse=True)
        placed = []
        for side in sides:
            for _ in range(200):
                cx = rng.randint(side // 2 + 4, size - side // 2 - 4)
                cy = rng.randint(side // 2 + 4, size - side // 2 - 4)
                if all(abs(cx - p["cx"]) >= (side + p["side"]) / 2 + 6 or abs(cy - p["cy"]) >= (side + p["side"]) / 2 + 6 for p in placed) \
                        and all(cx != p["cx"] for p in placed):
                    placed.append({"shape": rng.choice(SHAPES), "color": rng.choice(list(COLORS)), "side": side, "cx": cx, "cy": cy})
                    break
            else:
                break
        if len(placed) == n:
            break
    image = Image.new("RGB", (size, size), BACKGROUND)
    draw = ImageDraw.Draw(image)
    for p in placed:
        _draw_shape(draw, p["shape"], COLORS[p["color"]], p["cx"], p["cy"], p["side"])
    return image, placed


def make_questions(shapes, rng):
    largest = max(shapes, key=lambda p: p["side"])["color"]
    colors = rng.sample([c for c in COLORS if c != largest], rng.randint(2, 4)) + [largest]
    rng.shuffle(colors)
    leftmost = min(shapes, key=lambda p: p["cx"])["shape"]
    kinds = list(SHAPES)
    rng.shuffle(kinds)
    return [
        {"id": "largest_color", "prompt": "What color is the largest shape?", "type": "choice", "options": colors, "label": colors.index(largest)},
        {"id": "has_triangle", "prompt": "Is there a triangle in the image?", "type": "noul", "options": ["no", "yes"],
         "label": int(any(p["shape"] == "triangle" for p in shapes))},
        {"id": "count", "prompt": "How many shapes are in the image?", "type": "score", "options": list(COUNT_LEVELS), "label": len(shapes) - 1},
        {"id": "leftmost", "prompt": "Which shape is furthest to the left?", "type": "choice", "options": kinds, "label": kinds.index(leftmost)},
    ]


def generate_dataset(out_dir, sizes, seed=0):
    """Write PNGs and one JSONL per split. The state is a list of parts: an image, and sometimes an uninformative caption."""
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    counts = {}
    for split_idx, (split, n) in enumerate(sizes.items()):
        rng = random.Random(seed * 100 + split_idx)
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for i in range(n):
                image, shapes = make_scene(rng)
                rel = f"images/{split}_{i:05d}.png"
                image.save(out_dir / rel)
                state = [{"type": "image", "path": rel}]
                if rng.random() < 0.5:
                    state.append({"type": "text", "text": rng.choice(CAPTIONS)})
                f.write(json.dumps({"state": state, "questions": make_questions(shapes, rng)}, ensure_ascii=False) + "\n")
        counts[split] = n
    return counts
