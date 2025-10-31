from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch
from transformers import AutoTokenizer
from .config import DEVICE, TOP_K, TOP_P, TEMPERATURE, MAX_NEW_TOKENS
from .rewards import reward_components

@dataclass
class SampleInfo:
    input_ids: torch.Tensor
    prompt_len: int
    gen_len: int
    harmless: float
    brevity: float
    adhere: float

@torch.no_grad()
def sample_once(model, tokenizer: AutoTokenizer, prompt: str) -> Tuple[torch.Tensor, int, int, str]:
    enc = tokenizer(prompt, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    out = model.generate(
        **enc, do_sample=True, top_k=TOP_K, top_p=TOP_P, temperature=TEMPERATURE,
        max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tokenizer.eos_token_id,
    )
    full_ids = out[0]
    prompt_len = enc["input_ids"].shape[1]
    gen_ids = full_ids[prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return full_ids.detach().to("cpu"), int(prompt_len), int(gen_ids.numel()), text

def draw_samples(model, tokenizer, probes, batch_probes, k_samples) -> List[Tuple[SampleInfo, str]]:
    picked = torch.randperm(len(probes))[:batch_probes].tolist()
    samples: List[Tuple[SampleInfo, str]] = []
    for pidx in picked:
        prompt, gold = probes[pidx]
        for _ in range(k_samples):
            full_ids_cpu, prompt_len, gen_len, text = sample_once(model, tokenizer, prompt)
            if gen_len == 0:
                continue
            h, b, a = reward_components(prompt, text, gen_len, gold)
            samples.append((SampleInfo(full_ids_cpu, prompt_len, gen_len, h, b, a), prompt))
    return samples

