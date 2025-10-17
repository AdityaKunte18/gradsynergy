import os, time, json, math, torch
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass

def nowstamp():
    return time.strftime("%Y%m%d-%H%M%S")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def jsonl_writer(path):
    f = open(path, "a", encoding="utf-8")
    def write(obj):
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
    return write

@dataclass
class GenOutput:
    text: str
    logprobs: torch.Tensor  # (seq,) log probs for generated tokens
    token_ids: torch.Tensor

def chat_format(prompt:str) -> str:
    # minimal compatible chat format; tweak for your model family
    return f"User: {prompt}\nAssistant:"

@torch.no_grad()
def generate_and_logprobs(model, tokenizer, prompt:str, max_new_tokens:int=64, top_p:float=0.95, temperature:float=0.7, device="cuda"):
    model.eval()
    text = chat_format(prompt)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attn = inputs["attention_mask"]

    # Greedy or top-p sampling; we also need per-step logprobs
    # We'll do simple greedy for stability; you can switch to sampling.
    generated = []
    logprobs = []
    cur_ids = input_ids
    for _ in range(max_new_tokens):
        out = model(input_ids=cur_ids, attention_mask=attn)
        logits = out.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        token = torch.argmax(probs, dim=-1, keepdim=True)  # greedy
        lp = torch.log(probs.gather(-1, token).squeeze(1)) # (1,)->(1,)
        generated.append(token)
        logprobs.append(lp.squeeze(0)) # scalar
        cur_ids = torch.cat([cur_ids, token], dim=1)
        attn = torch.ones_like(cur_ids)

        # Stop if EOA token is produced
        if token.item() == tokenizer.eos_token_id:
            break

    gen_ids = torch.cat(generated, dim=1) if generated else torch.empty((1,0), dtype=torch.long, device=device)
    text_out = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    return GenOutput(text=text_out, logprobs=torch.stack(logprobs) if logprobs else torch.empty(0, device=device), token_ids=gen_ids.squeeze(0))

def flat_params(named_params):
    vecs = []
    for _, p in named_params:
        if p.grad is not None:
            vecs.append(p.grad.detach().float().reshape(-1))
    if len(vecs)==0:
        return None
    return torch.cat(vecs, dim=0)

def moving_avg(prev, new, beta=0.9):
    if prev is None:
        return new
    return beta * prev + (1-beta)*new

def clip_and_renorm(weights:Dict[str,float], wmin=0.1, wmax=10.0):
    clamped = {k: max(wmin, min(wmax, v)) for k,v in weights.items()}
    s = sum(clamped.values())
    return {k: v/s for k,v in clamped.items()}
