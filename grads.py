from typing import Dict

import numpy as np
import torch

from . import config as C
from .lora_utils import flatten_grads_for_groups, get_trainable_layer_groups
from .probes import load_probes_from_dataset
from .sampling import draw_samples, reinforce_backward_from_samples
from text_utils import robust_cosine

def objective_grads_per_layer(model, samples, layer_groups) -> Dict[str, Dict[int, torch.Tensor]]:
    results: Dict[str, Dict[int, torch.Tensor]] = {}
    objectives = {
        "Correctness": torch.tensor([1.0, 0.0, 0.0]),
        "Format": torch.tensor([0.0, 1.0, 0.0]),
        "Length": torch.tensor([0.0, 0.0, 1.0]),
    }
    from torch.cuda.amp import autocast

    for name, w in objectives.items():
        model.zero_grad(set_to_none=True)
        with autocast(enabled=False):
            _ = reinforce_backward_from_samples(
                model,
                samples,
                w,
                center_baseline=False,
                add_entropy=C.PROBE_ENTROPY_BONUS,
                grad_accum_steps=1,
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
        Cc = obj_grads["Correctness"].get(idx, torch.zeros(0))
        Fm = obj_grads["Format"].get(idx, torch.zeros(0))
        Ln = obj_grads["Length"].get(idx, torch.zeros(0))
        out[idx] = {
            "C-F": robust_cosine(Cc, Fm),
            "C-L": robust_cosine(Cc, Ln),
            "F-L": robust_cosine(Fm, Ln),
        }
    return out


def compute_global_objective_cosines(model, tokenizer, probe_file: str = "data/math500/train.parquet", probe_limit: int = 8) -> np.ndarray:
    probes = load_probes_from_dataset(probe_file, limit=probe_limit)
    samples_with_prompts = draw_samples(model, tokenizer, probes, batch_probes=min(len(probes), probe_limit), k_samples=2)
    samples = [s for (s, _) in samples_with_prompts]
    groups = get_trainable_layer_groups(model)
    obj_layer_grads = objective_grads_per_layer(model, samples, groups)
    glob: Dict[str, torch.Tensor] = {}
    for name, layer_map in obj_layer_grads.items():
        parts = [vec for _, vec in sorted(layer_map.items(), key=lambda kv: kv[0]) if vec.numel() > 0]
        glob[name] = torch.cat(parts) if parts else torch.zeros(0)
    names = ["Correctness", "Format", "Length"]
    M = np.zeros((3, 3), dtype=np.float32)
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            M[i, j] = robust_cosine(glob[ni], glob[nj])
    return M
