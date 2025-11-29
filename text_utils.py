import os
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterable, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
import re

_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


# ----------------------------
# Text helpers
# ----------------------------
def normalize(s: str) -> str:
    return " ".join(_WORD_RE.findall(s.lower()))


def sentence_count(text: str) -> int:
    return max(1, len(re.findall(r"[.!?]+", text)))


def word_count(text: str) -> int:
    return len(normalize(text).split())


def robust_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    a32 = a.float()
    b32 = b.float()
    na = a32.norm()
    nb = b32.norm()
    if na.item() == 0.0 or nb.item() == 0.0:
        return 0.0
    return float(torch.dot(a32, b32) / (na * nb))


# ----------------------------
# IO + logging helpers
# ----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def nowstamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def jsonl_writer(path: str) -> Callable[[dict], None]:
    f = open(path, "a", encoding="utf-8")

    def write(obj: dict) -> None:
        f.write(json.dumps(obj) + "\n")
        f.flush()

    return write


# ----------------------------
# Math utilities
# ----------------------------
def moving_avg(prev: float, new: float, beta: float = 0.9) -> float:
    if prev is None:
        return new
    return beta * prev + (1 - beta) * new


def clip_and_renorm(weights: dict, eps: float = 1e-6) -> dict:
    clipped = {k: max(eps, float(v)) for k, v in weights.items()}
    total = sum(clipped.values())
    return {k: v / total for k, v in clipped.items()} if total > 0 else clipped


# ----------------------------
# Generation helpers
# ----------------------------
@dataclass
class GenerationOutput:
    text: str
    logprobs: torch.Tensor  # shape [gen_len]


def generate_and_logprobs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    max_new_tokens: int,
    device: str = "cuda",
) -> GenerationOutput:
    """Generate text and return per-token logprobs for the generated portion."""
    enc = tokenizer(prompt, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    # Sample tokens without tracking grads, then compute logprobs with a differentiable forward pass.
    with torch.no_grad():
        gen = model.generate(
            **enc,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    sequences = gen.sequences  # [1, prompt + gen]
    prompt_len = enc["input_ids"].shape[1]
    gen_ids = sequences[0, prompt_len:]
    attn_mask = getattr(gen, "attention_mask", None)
    if attn_mask is None:
        attn_mask = torch.ones_like(sequences, device=device)

    outputs = model(
        input_ids=sequences.to(device),
        attention_mask=attn_mask.to(device),
        use_cache=False,
    )
    logits = outputs.logits[0]  # [seq, vocab]

    logprobs = []
    for t, tok_id in enumerate(gen_ids):
        # Logit at position predicts token t (offset by prompt).
        logit_idx = prompt_len + t - 1
        if logit_idx < 0 or logit_idx >= logits.shape[0]:
            continue
        step_lp = torch.log_softmax(logits[logit_idx], dim=-1)[tok_id]
        logprobs.append(step_lp)

    lp_tensor = torch.stack(logprobs) if logprobs else torch.zeros(0, device=device)
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return GenerationOutput(text=text, logprobs=lp_tensor)


# ----------------------------
# Misc helpers
# ----------------------------
def chat_format(messages: Iterable[Tuple[str, str]]) -> str:
    """Flatten a list of (role, content) pairs to a plain prompt."""
    parts = []
    for role, content in messages:
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def flat_params(model: torch.nn.Module) -> torch.Tensor:
    """Flatten trainable parameters into a single vector."""
    params = [p.detach().flatten() for p in model.parameters() if p.requires_grad]
    return torch.cat(params) if params else torch.zeros(0)


# ----------------------------
# Simple formatting heuristics
# ----------------------------
def looks_like_short_phrase(text: str, max_words: int = 10) -> bool:
    """Heuristic: short phrase means few words and no obvious sentence separators."""
    wc = word_count(text)
    if wc == 0 or wc > max_words:
        return False
    return sentence_count(text) <= 1


def looks_like_two_item_list(text: str) -> bool:
    """Detect two-item answers (bullets, numbered list, or 'X and Y')."""
    lines = [ln.strip(" -•\t") for ln in text.splitlines() if ln.strip()]
    if len(lines) == 2:
        return True
    numbered = [ln for ln in lines if re.match(r"^(\\d+\\.|\\d+\\))", ln)]
    if len(numbered) >= 2:
        return True
    parts = re.split(r",|;| and ", text)
    parts = [p.strip() for p in parts if p.strip()]
    return len(parts) == 2


def exact_match_loose(a: str, b: str) -> bool:
    """Case-insensitive match after stripping punctuation/spacing noise."""
    return normalize(a) == normalize(b)
