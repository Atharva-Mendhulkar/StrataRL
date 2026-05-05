from collections import Counter
from typing import List

TOKEN_NGRAM_SIZE     = 5
TOKEN_NGRAM_MAX_REPS = 4

def token_repetition_penalty(
    completion_token_ids: List[int],
    n: int = TOKEN_NGRAM_SIZE,
    max_reps: int = TOKEN_NGRAM_MAX_REPS,
) -> float:
    if len(completion_token_ids) < n:
        return 1.0
    ngrams  = [tuple(completion_token_ids[i:i+n]) for i in range(len(completion_token_ids)-n+1)]
    if not ngrams:
        return 1.0
    return 0.0 if max(Counter(ngrams).values()) > max_reps else 1.0
