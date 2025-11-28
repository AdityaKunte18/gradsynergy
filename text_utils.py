<<<<<<< HEAD:utils.py
import os
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterable, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
=======
import re
>>>>>>> 218916ca030d91de673531a9ad463771ef96f352:text_utils.py

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


<<<<<<< HEAD:utils.py
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
    with torch.no_grad():
        gen = model.generate(
            **enc,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    seq = gen.sequences[0]
    prompt_len = enc["input_ids"].shape[1]
    gen_ids = seq[prompt_len:]
    logprobs = []
    for t, tok_id in enumerate(gen_ids):
        score = gen.scores[t][0]  # [vocab]
        logprob = torch.log_softmax(score, dim=-1)[tok_id]
        logprobs.append(logprob)
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
=======
>>>>>>> 218916ca030d91de673531a9ad463771ef96f352:text_utils.py
