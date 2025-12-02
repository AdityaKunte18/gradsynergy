import re
from typing import Dict, List, Tuple, Optional, DefaultDict
from collections import defaultdict
import torch
from torch import nn
from peft import LoraConfig, get_peft_model

# Allow running as a script (no package) or as a module.
try:
    from .config import LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES
except ImportError:
    from config import LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES

def add_lora_adapters(model) -> nn.Module:
    cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES, bias="none", task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    print(f"[LoRA] Trainable params: {trainable:,} / {total:,} ({100.0*trainable/total:.2f}%)")
    return peft_model

def enable_checkpointing_and_freeze_base(model):
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False

_LAYER_RE = re.compile(r"\blayers\.(\d+)\b")

def layer_index_from_name(param_name: str) -> Optional[int]:
    m = _LAYER_RE.search(param_name)
    return int(m.group(1)) if m else None

def get_trainable_layer_groups(model) -> Dict[int, List[Tuple[str, nn.Parameter]]]:
    groups: DefaultDict[int, List[Tuple[str, nn.Parameter]]] = defaultdict(list)
    any_index = False
    for n, p in model.named_parameters():
        if p.requires_grad:
            idx = layer_index_from_name(n)
            if idx is not None:
                groups[idx].append((n, p)); any_index = True
    if not any_index:
        for n, p in model.named_parameters():
            if p.requires_grad:
                groups[-1].append((n, p))
    return dict(sorted(groups.items(), key=lambda kv: kv[0]))

def flatten_grads_for_groups(
    groups: Dict[int, List[Tuple[str, nn.Parameter]]], device: Optional[torch.device] = None
) -> Dict[int, torch.Tensor]:
    """
    Flatten gradients per layer/group. If device is provided, keep tensors there (e.g., GPU).
    """
    out: Dict[int, torch.Tensor] = {}
    for idx, items in groups.items():
        parts = [
            p.grad.detach().float().reshape(-1).to(device if device is not None else p.grad.device)
            for _, p in items
            if p.grad is not None
        ]
        out[idx] = torch.cat(parts, dim=0) if parts else torch.zeros(0, device=device)
    return out

def layer_grad_norms(groups) -> Dict[int, float]:
    norms = {}
    for idx, items in groups.items():
        parts = [p.grad.detach().float().reshape(-1).cpu()
                 for _, p in items if p.grad is not None]
        norms[idx] = float(torch.cat(parts).norm()) if parts else float('nan')
    return norms

def robust_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    a32 = a.float(); b32 = b.float()
    na = a32.norm(); nb = b32.norm()
    if na.item() == 0.0 or nb.item() == 0.0:
        return 0.0
    return float(torch.dot(a32, b32) / (na * nb))
