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

# ─────────────────────────────────────────────────────────────────────────────
# PATCH I-11 — closes the tag-presence-without-content loophole
#
# OBSERVED: at step 500, completions like
#   <decompose>5*17</decompose><compute>85</compute><verify>85</verify>
# (≈30-40 tokens total) passed structural_reward()=1.0 — every existing gate
# (tag presence, ordering, MIN_THINK_CHARS=80 via tag-syntax overhead alone,
# no garbage, no repetition) is satisfied with zero actual reasoning content.
#
# This made A_short ≈ A_long for "correct" completions, which then interacted
# with the I-10 length-norm bug to compound brevity over 500 steps.
#
# FIX: each REQUIRED domain tag must individually contain >= MIN_TAG_CONTENT_CHARS
# of stripped content. <decompose></decompose> (0 chars) now fails outright.
# 15 chars ≈ 3-4 tokens of genuine content per tag — low enough that real
# (even terse) reasoning clears it, high enough that empty/trivial tags don't.
#
# Side effect (intended): for gsm8k, 3 required tags x 15 chars minimum
# content + ~66 chars of tag syntax = ~111 chars, which already exceeds
# DOMAIN_MIN_THINK_CHARS["gsm8k"]=80. The old MIN_THINK_CHARS check becomes a
# fast pre-filter; this gate is the one that actually enforces substance.
# ─────────────────────────────────────────────────────────────────────────────

MIN_TAG_CONTENT_CHARS = 15

# ── Compiled patterns ─────────────────────────────────────────────────────────

THINK_PATTERN  = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

GARBAGE_PATTERNS = [
    re.compile(r"(.)\1{20,}",                  re.DOTALL),
    re.compile(r"(\b\w+\b)(\s+\1){5,}",        re.DOTALL),
    re.compile(r"^[\s\W]*$",                   re.DOTALL),
]

# ── Helper functions ──────────────────────────────────────────────────────────

def compute_repetition_score(text: str, n: int = 5) -> int:
    words = text.lower().split()
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


def structural_reward(completion: str, domain: str = "default") -> float:
    """
    Returns 1.0 iff all gates pass. 0.0 on any failure.

    Gates (in order):
      1. Both <think> and <answer> present, correctly ordered
      2. <think> content >= domain-appropriate minimum length
      3. <answer> content non-empty
      4. No garbage patterns in <think>
      5. Word-level n-gram repetition <= MAX_NGRAM_REPEAT
      6. All domain-specific template tags present in <think>
      7. PATCH I-11: each required tag has >= MIN_TAG_CONTENT_CHARS of
         stripped content (closes the empty-tag loophole)
    """
    think_m  = THINK_PATTERN.search(completion)
    answer_m = ANSWER_PATTERN.search(completion)

    # Gate 1
    if not think_m or not answer_m:
        return 0.0
    if think_m.end() > answer_m.start():
        return 0.0

    think  = think_m.group(1).strip()
    answer = answer_m.group(1).strip()

    # Gate 2
    min_chars = max(
        UNIVERSAL_MIN_THINK_CHARS,
        DOMAIN_MIN_THINK_CHARS.get(domain, DOMAIN_MIN_THINK_CHARS["default"])
    )
    if len(think) < min_chars:
        return 0.0

    # Gate 3
    if not answer:
        return 0.0

    # Gate 4
    for pattern in GARBAGE_PATTERNS:
        if pattern.search(think):
            return 0.0

    # Gate 5
    if compute_repetition_score(think, n=5) > MAX_NGRAM_REPEAT:
        return 0.0

    # Gate 6
    required_tags = DOMAIN_TEMPLATES.get(domain, [])
    for tag in required_tags:
        if f"<{tag}>" not in think or f"</{tag}>" not in think:
            return 0.0

    # Gate 7 — PATCH I-11: per-tag content length
    tag_lengths = _tag_content_lengths(think, required_tags)
    for tag, length in tag_lengths.items():
        if length < MIN_TAG_CONTENT_CHARS:
            return 0.0

    return 1.0
