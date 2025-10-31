from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch
from torch import nn
from . import config as C
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
def sample_once(model, tokenizer, prompt: str) -> Tuple[torch.Tensor, int, int, str]:
    enc = tokenizer(prompt, return_tensors="pt")
    enc = {k: v.to(C.DEVICE) for k, v in enc.items()}
    out = model.generate(
        **enc, do_sample=True, top_k=C.TOP_K, top_p=C.TOP_P, temperature=C.TEMPERATURE,
        max_new_tokens=C.MAX_NEW_TOKENS, pad_token_id=tokenizer.eos_token_id,
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

def reinforce_backward_from_samples(
    model, samples: List[SampleInfo], w: torch.Tensor, *,
    center_baseline: bool = True, add_entropy: float = 0.0,
    grad_accum_steps: int = 1
) -> float:
    """
    REINFORCE: token-sum objective; fp32 path.
    Returns baseline value used (mean or 0).
    """
    if len(samples) == 0:
        return 0.0

    scalars = [s.harmless * w[0].item() + s.brevity * w[1].item() + s.adhere * w[2].item()
               for s in samples]
    s_mean = float(sum(scalars) / len(scalars)) if center_baseline else 0.0

    model.train()
    from torch.cuda.amp import autocast
    for s, r_scalar in zip(samples, scalars):
        ids = s.input_ids.to(C.DEVICE, non_blocking=True)
        with autocast(enabled=False):
            outputs = model(input_ids=ids.unsqueeze(0), use_cache=False)
            logits = outputs.logits.float()
            log_probs = nn.functional.log_softmax(logits, dim=-1)
            probs = log_probs.exp()

            if s.gen_len > 0:
                lp_sum = []
                ent_sum = []
                for idx in range(s.prompt_len, s.prompt_len + s.gen_len):
                    tok_id = ids[idx]
                    lp_sum.append(log_probs[0, idx - 1, tok_id])
                    if add_entropy > 0.0:
                        p_step = probs[0, idx - 1]
                        ent = -(p_step * (p_step + 1e-32).log()).sum()
                        ent_sum.append(ent)
                lp_sum = torch.stack(lp_sum).sum()
                advantage = (r_scalar - s_mean) if center_baseline else r_scalar
                loss = - (advantage * lp_sum)
                if add_entropy > 0.0 and ent_sum:
                    loss += - add_entropy * torch.stack(ent_sum).mean()

        (loss / grad_accum_steps).backward()

        del outputs, logits, log_probs, probs, ids
        if C.DEVICE == "cuda":
            import torch
            torch.cuda.empty_cache()

    return s_mean
