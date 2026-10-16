import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import copy, torch
from transformers import AutoProcessor, AutoModelForMultimodalLM

processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B")

def runtest(dtype, label):
    model = AutoModelForMultimodalLM.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=dtype, device_map="auto").eval()
    tok = processor.tokenizer
    DEV = next(model.parameters()).device
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
            d = (ref.hidden_states[-1][:,-1] - cont.hidden_states[-1][:,-1]).abs().max().item()
            print(f"[{label}] Q{name}: last-token hs[-1] max-abs-diff={d:.6f} refmax={ref.hidden_states[-1][:,-1].abs().max().item():.3f}")
    del model
    torch.cuda.empty_cache()

runtest(torch.float32, "fp32")
