import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import copy, torch
from transformers import AutoProcessor, AutoModelForMultimodalLM
import transformers.models.qwen3_5.modeling_qwen3_5 as M

processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B")
model = AutoModelForMultimodalLM.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16, device_map="auto").eval()
tok = processor.tokenizer
DEV = next(model.parameters()).device

state_text = "Customer ticket: payouts have failed three days in a row, and the bank says nothing is wrong on their end."
question_a = " Question: which team should handle this — billing, technical, or sales?"

state_ids = torch.tensor([tok(state_text, add_special_tokens=False)["input_ids"]], device=DEV)
L = state_ids.shape[1]
full = torch.tensor([tok(state_text+question_a, add_special_tokens=False)["input_ids"]], device=DEV)
q_only = full[:, L:]
qn = q_only.shape[1]

# monkeypatch TextModel.forward to log position_ids fed
orig = M.Qwen3_5TextModel.forward
logs = {}
def patched(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, use_cache=None, **kw):
    logs.setdefault('pos', []).append(None if position_ids is None else (tuple(position_ids.shape), position_ids.reshape(position_ids.shape[0], -1)[:, :3].tolist(), position_ids.reshape(position_ids.shape[0], -1)[:, -3:].tolist()))
    return orig(self, input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, **kw)
M.Qwen3_5TextModel.forward = patched

with torch.no_grad():
    ref = model(input_ids=full, output_hidden_states=True, use_cache=False)
    print("BASELINE pos fed:", logs['pos'][-1])
    so = model(input_ids=state_ids, use_cache=True)
    cache = so.past_key_values
    cl = copy.deepcopy(cache)
    print("clone get_seq_length:", cl.get_seq_length())
    logs['pos'].clear()
    cont = model(input_ids=q_only, past_key_values=cl, use_cache=True, output_hidden_states=True)
    print("CONT pos fed:", logs['pos'][-1])

rl = ref.hidden_states[-1][:, -qn:]
clast = cont.hidden_states[-1]
print("last-layer per-position max-abs-diff:")
for j in range(qn):
    d = (rl[0,j]-clast[0,j]).abs().max().item()
    n = rl[0,j].abs().max().item()
    print(f"  qpos {j}: diff={d:.4f} ref_absmax={n:.3f}")
print("last-token rel (mean):", ((rl[0,-1]-clast[0,-1]).abs()/ (rl[0,-1].abs()+1e-6)).mean().item())

# which dims blow up at last token
dd = (rl[0,-1]-clast[0,-1]).abs()
top = torch.topk(dd, 5)
print("top-5 diff dims at last token:", top.values.tolist(), top.indices.tolist())
print("  ref at those dims:", rl[0,-1][top.indices].tolist())
print("  cont at those dims:", clast[0,-1][top.indices].tolist())
