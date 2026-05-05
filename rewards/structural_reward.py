import re
from collections import Counter
from typing import List

# ── Domain template definitions ──────────────────────────────────────────────

DOMAIN_TEMPLATES = {
    "gsm8k":       ["decompose", "compute", "verify"],
    "mmlu":        ["recall", "evaluate"],
    "strategyqa":  ["decompose", "resolve", "synthesize"],
    "aqua_rat":    ["decompose", "compute", "verify"],
}

DOMAIN_MIN_THINK_CHARS = {
    "gsm8k":       80,
    "mmlu":        100,
    "strategyqa":  100,
    "aqua_rat":    120,
    "default":     80,
}

UNIVERSAL_MIN_THINK_CHARS = 50
MAX_NGRAM_REPEAT          = 4

# ── Compiled patterns ─────────────────────────────────────────────────────────

THINK_PATTERN  = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

GARBAGE_PATTERNS = [
    re.compile(r"(.)\1{20,}",                  re.DOTALL),  # 20+ repeated chars
    re.compile(r"(\b\w+\b)(\s+\1){5,}",        re.DOTALL),  # same word 5+ times
    re.compile(r"^[\s\W]*$",                   re.DOTALL),  # whitespace/punct only
]

# ── Helper functions ──────────────────────────────────────────────────────────

def compute_repetition_score(text: str, n: int = 5) -> int:
    words  = text.lower().split()
    if len(words) < n:
        return 0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    return max(Counter(ngrams).values()) if ngrams else 0


def structural_reward(completion: str, domain: str = "default", phase: str = "strict") -> float:
    """
    Returns a semi-dense continuous score in [0, 1.0].
    
    Components (0.2 each):
      1. Tag presence: Both <think> and <answer> exist
      2. Content: <answer> is non-empty
      3. Length: <think> meets minimum threshold
      4. Quality: No garbage patterns
      5. Repetition: Word-level repetition is within limits
    """
    score = 0.0
    
    think_m  = THINK_PATTERN.search(completion)
    answer_m = ANSWER_PATTERN.search(completion)

    # 1. Tag Presence & Order (0.2)
    if think_m and answer_m:
        score += 0.2
    
    think   = think_m.group(1).strip() if think_m else ""
    answer  = answer_m.group(1).strip() if answer_m else ""
    words   = think.lower().split()

    # 2. Answer Content (0.2)
    if answer:
        score += 0.2

    # 3. Length (0.2)
    min_chars = max(
        UNIVERSAL_MIN_THINK_CHARS,
        DOMAIN_MIN_THINK_CHARS.get(domain, DOMAIN_MIN_THINK_CHARS["default"])
    )
    if phase == "bootstrap":
        min_chars = int(min_chars * 0.6)
    
    if len(think) >= min_chars:
        score += 0.2

    # 4. Quality: No Garbage (0.2)
    garbage_found = False
    for pattern in GARBAGE_PATTERNS:
        if pattern.search(think):
            garbage_found = True
            break
    if not garbage_found and think:
        score += 0.2

    # 5. Repetition: Within Limits (0.2)
    if think and compute_repetition_score(think, n=5) <= MAX_NGRAM_REPEAT:
        score += 0.2

    # Domain templates gating (multiplicative)
    required_tags = DOMAIN_TEMPLATES.get(domain, [])
    if required_tags:
        found_tags = 0
        for tag in required_tags:
            if f"<{tag}>" in think and f"</{tag}>" in think:
                found_tags += 1
        
        if found_tags < len(required_tags):
            # Apply multiplicative penalty for missing reasoning steps
            score *= 0.5

    # Deterministic Diversity Bonus (max 0.01)
    if words:
        diversity = len(set(words)) / len(words)
        score += 0.01 * diversity

    return min(1.0, score)
