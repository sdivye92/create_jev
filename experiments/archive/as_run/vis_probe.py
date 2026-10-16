import os, sys
os.environ["HF_HUB_DISABLE_XET"] = "1"; os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForMultimodalLM
processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B")
model = AutoModelForMultimodalLM.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16, device_map="auto").eval()
core = model.model; dev = next(core.parameters()).device
ip = processor.image_processor
print("image_processor:", type(ip).__name__, {k: getattr(ip, k) for k in ("patch_size", "merge_size", "temporal_patch_size", "min_pixels", "max_pixels") if hasattr(ip, k)}, "| size:", getattr(ip, "size", None))
vc = model.config.vision_config
print("vision config:", {k: getattr(vc, k) for k in ("hidden_size", "depth", "num_heads", "patch_size", "spatial_merge_size", "out_hidden_size") if hasattr(vc, k)})

def img(color, shape="circle", size=224):
    im = Image.new("RGB", (size, size), "white"); d = ImageDraw.Draw(im)
    box = [size * 0.25, size * 0.25, size * 0.75, size * 0.75]
    d.ellipse(box, fill=color) if shape == "circle" else d.rectangle(box, fill=color)
    return im

tok = processor.tokenizer
text = "<|vision_start|><|image_pad|><|vision_end|>What color is the shape?"
inp = processor(text=[text], images=[img("red")], return_tensors="pt")
print("keys:", list(inp.keys()))
for k, v in inp.items(): print(f"  {k}: {tuple(v.shape)} {v.dtype}")
n_pad = (inp["input_ids"] == tok.convert_tokens_to_ids("<|image_pad|>")).sum().item()
print("image_pad tokens after expansion:", n_pad, "| image_grid_thw:", inp["image_grid_thw"].tolist())
if "mm_token_type_ids" in inp: print("mm_token_type_ids values:", sorted(set(inp["mm_token_type_ids"].flatten().tolist())))

inp = {k: v.to(dev) for k, v in inp.items()}
with torch.no_grad():
    h_red = core(**inp).last_hidden_state
    inp2 = dict(inp); inp2["pixel_values"] = processor(text=[text], images=[img("blue", "square")], return_tensors="pt")["pixel_values"].to(dev)
    h_blue = core(**inp2).last_hidden_state
print("hidden:", tuple(h_red.shape), "| last-token cosine red-circle vs blue-square: %.4f" % torch.nn.functional.cosine_similarity(h_red[0, -1].float(), h_blue[0, -1].float(), dim=0).item())

chat = [{"role": "user", "content": [{"type": "image", "image": img("red")}, {"type": "text", "text": "What color is the shape? Answer with one word."}]}]
gi = processor.apply_chat_template(chat, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(dev)
out = model.generate(**gi, max_new_tokens=6, do_sample=False)
print("generate:", repr(processor.decode(out[0][gi["input_ids"].shape[-1]:], skip_special_tokens=True)))
