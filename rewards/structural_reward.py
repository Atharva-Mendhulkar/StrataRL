# rewards/structural_reward.py

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
MIN_TAG_CONTENT_CHARS     = 10

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


def _tag_content_lengths(think: str, required_tags: List[str]) -> dict:
    """
    PATCH I-11: extract stripped content length for each required tag.
    Returns {tag_name: content_char_count}. Missing tags return -1.
    """
    lengths = {}
    for tag in required_tags:
        m = re.search(rf"<{tag}>(.*?)</{tag}>", think, re.DOTALL)
        lengths[tag] = len(m.group(1).strip()) if m else -1
    return lengths

# ── Graded Reward Configuration ─────────────────────────────────────────────

STRUCTURAL_WEIGHTS = {
    "think_present":   0.25,   # <think> tag exists
    "answer_present":  0.25,   # <answer> tag exists
    "correct_order":   0.20,   # think precedes answer
    "min_length":      0.20,   # domain-appropriate minimum length satisfied
    "no_garbage":      0.10,   # passes repetition/garbage gates
}

def structural_reward(completion: str, domain: str = "default", phase: str = "strict") -> float:
    """
    Graded structural reward: partial credit per verified component.
    Returns float in [0.1].
    """
    score = 0.0
    
    think_match  = THINK_PATTERN.search(completion)
    answer_match = ANSWER_PATTERN.search(completion)
    
    # Component 1: think tag present
    if think_match:
        score += STRUCTURAL_WEIGHTS["think_present"]
    
    # Component 2: answer tag present
    if answer_match:
        score += STRUCTURAL_WEIGHTS["answer_present"]
    
    # Component 3: correct ordering (only testable if both present)
    if think_match and answer_match:
        if think_match.end() <= answer_match.start():
            score += STRUCTURAL_WEIGHTS["correct_order"]
    
    # Component 4: minimum think length (domain-aware)
    if think_match:
        think_content = think_match.group(1).strip()
        domain_min = DOMAIN_MIN_THINK_CHARS.get(
            domain, DOMAIN_MIN_THINK_CHARS["default"]
        )
        effective_min = max(UNIVERSAL_MIN_THINK_CHARS, domain_min)
        if phase == "bootstrap":
            effective_min = int(effective_min * 0.6)
            
        if len(think_content) >= effective_min:
            score += STRUCTURAL_WEIGHTS["min_length"]
    
    # Component 5: no garbage (only test if think block exists and has content)
    if think_match:
        think_content = think_match.group(1).strip()
        has_garbage = any(p.search(think_content) for p in GARBAGE_PATTERNS)
        rep_score   = compute_repetition_score(think_content, n=5)
        if not has_garbage and rep_score <= MAX_NGRAM_REPEAT:
            score += STRUCTURAL_WEIGHTS["no_garbage"]
    
    # Domain templates gating (multiplicative)
    required_tags = DOMAIN_TEMPLATES.get(domain, [])
    if required_tags and think_match:
        think_content = think_match.group(1)
        found_tags = 0
        tag_lengths = _tag_content_lengths(think_content, required_tags)
        for tag, length in tag_lengths.items():
            if length >= MIN_TAG_CONTENT_CHARS:
                found_tags += 1
        
        if found_tags < len(required_tags):
            score *= 0.5

    return round(score, 4)
