"""Is the bf16 cache-reuse gap just rounding? Compare both bf16 paths to an fp32 reference."""
import os, copy, sys
os.environ["HF_HUB_DISABLE_XET"]="1"; os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]="1"
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM
sys.path.insert(0, os.path.dirname(__file__))
M="Qwen/Qwen3.5-0.8B"
S="Customer ticket: payouts have failed three days in a row, and the bank says nothing is wrong on their end."
QS={"A":" Question: which team should handle this — billing, technical, or sales?","B":" Question: is this urgent and does it need escalation?"}
tok=AutoProcessor.from_pretrained(M).tokenizer
ids=lambda t: torch.tensor([tok(t,add_special_tokens=False)["input_ids"]],device="cuda")
@torch.no_grad()
def run(dtype):
    m=AutoModelForMultimodalLM.from_pretrained(M,dtype=dtype).to("cuda").eval().model
    s=ids(S); L=s.shape[1]; c=m(input_ids=s,use_cache=True).past_key_values
    out={}
    for k,q in QS.items():
        f=ids(S+q)
        base=m(input_ids=f,output_hidden_states=True,use_cache=False).hidden_states[-1][0,-1].float().cpu()
        cont=m(input_ids=f[:,L:],past_key_values=copy.deepcopy(c),use_cache=True,output_hidden_states=True).hidden_states[-1][0,-1].float().cpu()
        out[k]=(base,cont)
    del m,c; torch.cuda.empty_cache(); return out
r32=run(torch.float32); r16=run(torch.bfloat16)
for k in QS:
    ref=r32[k][0]; b,c=r16[k]
    cos=lambda x,y: torch.nn.functional.cosine_similarity(x,y,dim=0).item()
    print(f"Q{k}: bf16base-vs-fp32 maxdiff={(b-ref).abs().max():.3f} | bf16cache-vs-fp32 maxdiff={(c-ref).abs().max():.3f} | "
          f"bf16base-vs-bf16cache maxdiff={(b-c).abs().max():.3f} | cos(base,cache)={cos(b,c):.6f} | rel L2={((b-c).norm()/b.norm()):.4f}")
