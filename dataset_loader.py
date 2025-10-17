
from typing import List, Dict, Any, Tuple, Optional
from datasets import load_dataset

def load_gsm8k_small(n:int=200, split:str="train") -> List[Dict[str,Any]]:
    """
    Returns a list of {"prompt": str, "label": int}
    Uses the 'gsm8k' dataset and extracts the final numeric answer from 'answer' field.
    """
    ds = load_dataset("gsm8k", "main")[split]
    out = []
    import re
    for i, ex in enumerate(ds):
        if i>=n: break
        q = ex["question"]
        ans = ex["answer"]
        # Extract the last integer in the solution as the final answer
        m = re.findall(r"(-?\d+)", ans)
        if not m: 
            continue
        final = int(m[-1])
        out.append({"prompt": q + "\n\nPlease end with 'Answer: <n>'", "label": final})
    return out

def load_real_toxicity_prompts_small(n:int=200, split:str="train") -> List[Dict[str,Any]]:
    """
    Returns a list of {"prompt": str} that tend to be unsafe/toxic prompts (no label).
    We'll just use prompts as-is; reward function will detect refusal style.
    """
    try:
        ds = load_dataset("allenai/real-toxicity-prompts")[split]
    except Exception:
        # Some mirrors/updated versions exist; if main unavailable, fall back to a lite fork
        ds = load_dataset("oskarvanderwal/real-toxicity-prompts-lite")[split]
    out = []
    for i, ex in enumerate(ds):
        if i>=n: break
        p = ex.get("prompt", {}).get("text") if isinstance(ex.get("prompt"), dict) else ex.get("prompt")
        if not p:
            p = ex.get("text") or ""
        if not isinstance(p, str):
            continue
        out.append({"prompt": p})
    return out

def make_unified(gsm_n:int=200, rtp_n:int=200, split:str="train") -> List[Dict[str,Any]]:
    """
    Unified list compatible with the training loop (expects 'prompt' and optional 'label').
    """
    math = load_gsm8k_small(n=gsm_n, split=split)
    tox  = load_real_toxicity_prompts_small(n=rtp_n, split="train")  # RTP has only train
    return math + tox
