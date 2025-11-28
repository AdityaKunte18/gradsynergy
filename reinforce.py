from typing import Dict, List
import numpy as np
import torch
from torch import nn
from .config import DEVICE, GRAD_ACCUM_STEPS, PROBE_ENTROPY_BONUS
from .lora_utils import flatten_grads_for_groups, robust_cosine

def reinforce_backward_from_samples(
    model, samples, w: torch.Tensor, *,
    center_baseline: bool = True, add_entropy: float = 0.0
) -> float:
    """
    REINFORCE: sum over generated tokens, fp32 math.
    - center_baseline: subtract batch-mean reward (variance reduction). Keep True for TRAINING.
      For PROBE passes, set False to avoid zero grads when rewards tie.
    - add_entropy: probe-only small entropy bonus to ensure non-zero grads (0 for training).
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
        ids = s.input_ids.to(DEVICE, non_blocking=True)
        with autocast(enabled=False):  # fp32 path
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

        (loss / GRAD_ACCUM_STEPS).backward()

        del outputs, logits, log_probs, probs, ids
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    return s_mean

def objective_grads_per_layer(model, samples, layer_groups) -> Dict[str, Dict[int, torch.Tensor]]:
    """
    Probe gradients per objective (Harmless, Brevity, Adherence). No baseline centering + small entropy bonus.
    """
    results: Dict[str, Dict[int, torch.Tensor]] = {}
    objectives = {
        "Harmless": torch.tensor([1.0, 0.0, 0.0]),
        "Brevity":  torch.tensor([0.0, 1.0, 0.0]),
        "Adherence":torch.tensor([0.0, 0.0, 1.0]),
    }
    from torch.cuda.amp import autocast
    for name, w in objectives.items():
        model.zero_grad(set_to_none=True)
        with autocast(enabled=False):
            _ = reinforce_backward_from_samples(
                model, samples, w,
                center_baseline=False,
                add_entropy=PROBE_ENTROPY_BONUS,
            )
        grads = flatten_grads_for_groups(layer_groups)
        results[name] = grads
        model.zero_grad(set_to_none=True)
    return results

def compute_pairwise_cosines_per_layer(obj_grads: Dict[str, Dict[int, torch.Tensor]]) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    layers = set()
    for d in obj_grads.values():
        layers.update(d.keys())
    for idx in sorted(layers):
        H = obj_grads["Harmless"].get(idx, torch.zeros(0))
        B = obj_grads["Brevity"].get(idx, torch.zeros(0))
        A = obj_grads["Adherence"].get(idx, torch.zeros(0))
        out[idx] = {"H-B": robust_cosine(H, B), "H-A": robust_cosine(H, A), "B-A": robust_cosine(B, A)}
    return out

def compute_global_objective_cosines(model, tokenizer, draw_fn) -> np.ndarray:
    samples_with_prompts = draw_fn(batch_probes=8, k_samples=2)
    samples = [s for (s, _) in samples_with_prompts]
    # Build global vectors by concatenating layer grads
    layer_groups = None
    from .lora_utils import get_trainable_layer_groups
    layer_groups = get_trainable_layer_groups(model)
    obj_layer_grads = objective_grads_per_layer(model, samples, layer_groups)
    glob = {}
    for name, layer_map in obj_layer_grads.items():
        parts = [vec for _, vec in sorted(layer_map.items(), key=lambda kv: kv[0]) if vec.numel() > 0]
        glob[name] = (torch.cat(parts) if parts else torch.zeros(0))
    names = ["Harmless", "Brevity", "Adherence"]
    M = np.zeros((3, 3), dtype=np.float32)
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            M[i, j] = robust_cosine(glob[ni], glob[nj])
    return M

