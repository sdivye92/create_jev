import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import copy, torch
from transformers import AutoProcessor, AutoModelForMultimodalLM

processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B")
model = AutoModelForMultimodalLM.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16, device_map="auto").eval()
tok = processor.tokenizer
DEV = next(model.parameters()).device
lm = model.model.language_model

state_text = "Customer ticket: payouts have failed three days in a row, and the bank says nothing is wrong on their end."
qa = " Question: which team should handle this — billing, technical, or sales?"
qb = " Question: is this urgent and does it need escalation?"
state_ids = torch.tensor([tok(state_text, add_special_tokens=False)["input_ids"]], device=DEV)
L = state_ids.shape[1]

with torch.no_grad():
    so = model(input_ids=state_ids, use_cache=True)
    cache = so.past_key_values
    for name, q in [("A", qa), ("B", qb)]:
        full = torch.tensor([tok(state_text+q, add_special_tokens=False)["input_ids"]], device=DEV)
        q_only = full[:, L:]
        ref = model(input_ids=full, output_hidden_states=True, use_cache=False)
        cl = copy.deepcopy(cache)
        cont = model(input_ids=q_only, past_key_values=cl, use_cache=True, output_hidden_states=True)
        # pre-norm
        rp = ref.hidden_states[-1][:,-1]; cp = cont.hidden_states[-1][:,-1]
        # post-norm
        rn = lm.norm(ref.hidden_states[-1])[:,-1]; cnn = lm.norm(cont.hidden_states[-1])[:,-1]
        print(f"Q{name}: prenorm diff={ (rp-cp).abs().max().item():.5f} (mag {rp.abs().max().item():.1f}) | postnorm diff={(rn-cnn).abs().max().item():.5f} (mag {rn.abs().max().item():.2f})")
        # argmax token match via lm_head
        lr = model.lm_head(rn.to(model.lm_head.weight.dtype)) if hasattr(model,'lm_head') else None
