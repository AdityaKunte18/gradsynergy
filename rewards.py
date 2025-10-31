from typing import Optional, Tuple
from .config import (
    UNSAFE_LIST, SAFETY_PEN_PER_HIT, CONCISION_LAMBDA,
    ADHERENCE_POS, ADHERENCE_NEG
)
from .text_utils import sentence_count, looks_like_short_phrase, looks_like_two_item_list, exact_match_loose

def contains_unsafe_per_occurrence(text: str) -> int:
    low = text.lower()
    return sum(low.count(w) for w in UNSAFE_LIST)

def adherence_score(prompt: str, output_text: str) -> float:
    p = prompt.lower()
    t = output_text.strip()
    if "answer in one short phrase" in p or "answer briefly" in p or "keep it concise" in p:
        return ADHERENCE_POS if looks_like_short_phrase(t) else ADHERENCE_NEG
    if "one or two sentences" in p:
        return ADHERENCE_POS if sentence_count(t) <= 2 else ADHERENCE_NEG
    if "list two" in p:
        return ADHERENCE_POS if looks_like_two_item_list(t) else ADHERENCE_NEG
    return 0.0

def reward_components(prompt: str, text: str, token_count: int, gold: Optional[str]) -> Tuple[float, float, float]:
    harmless = - SAFETY_PEN_PER_HIT * contains_unsafe_per_occurrence(text)
    brevity = - CONCISION_LAMBDA * max(1, token_count)  # always non-zero
    adhere = adherence_score(prompt, text)
    if gold is not None and exact_match_loose(text, gold):
        adhere += 0.25
    return float(harmless), float(brevity), float(adhere)

