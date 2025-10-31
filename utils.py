import re, torch
_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)

def normalize(s: str) -> str:
    return " ".join(_WORD_RE.findall(s.lower()))

def sentence_count(text: str) -> int:
    import re as _re
    return max(1, len(_re.findall(r"[.!?]+", text)))

def word_count(text: str) -> int:
    return len(normalize(text).split())

def looks_like_short_phrase(text: str, max_words: int = 6) -> bool:
    return word_count(text) <= max_words and sentence_count(text) == 1

def looks_like_two_item_list(text: str) -> bool:
    norm = normalize(text)
    if " and " in norm:
        parts = [p.strip() for p in norm.split(" and ")]
        return len(parts) == 2 and all(len(p) > 0 for p in parts)
    parts = [p.strip() for p in norm.split(",")]
    parts = [p for p in parts if p]
    return len(parts) == 2

def exact_match_loose(output_text: str, gold: str) -> bool:
    out = normalize(output_text).split()
    ans = normalize(gold).split()
    if not ans:
        return False
    L = len(ans)
    for i in range(len(out) - L + 1):
        if out[i:i+L] == ans:
            return True
    return False

def robust_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    a32 = a.float(); b32 = b.float()
    na = a32.norm(); nb = b32.norm()
    if na.item() == 0.0 or nb.item() == 0.0:
        return 0.0
    return float(torch.dot(a32, b32) / (na * nb))
