import re
import sympy

ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

def extract_answer(completion: str) -> str | None:
    m = ANSWER_PATTERN.search(completion)
    return m.group(1).strip() if m else None


def normalize_number_str(s: str) -> str:
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        return str(float(s))
    except:
        return s


def verify_numeric(completion: str, ground_truth: str) -> float:
    """GSM8K / AQuA-RAT: symbolic numeric equivalence via SymPy."""
    answer = extract_answer(completion)
    if answer is None:
        return 0.0
    try:
        result = sympy.simplify(sympy.sympify(answer) - sympy.sympify(ground_truth))
        return 1.0 if result == 0 else 0.0
    except:
        return float(normalize_number_str(answer) == normalize_number_str(ground_truth))


def verify_letter(completion: str, ground_truth: str) -> float:
    """MMLU: exact letter match A/B/C/D."""
    answer = extract_answer(completion)
    if answer is None:
        return 0.0
    m = re.search(r'\b([A-Da-d])\b', answer.strip())
    return float(bool(m) and m.group(1).upper() == ground_truth.strip().upper())


def verify_yesno(completion: str, ground_truth: str) -> float:
    """StrategyQA: yes/no match."""
    answer = extract_answer(completion)
    if answer is None:
        return 0.0
    return float(answer.strip().lower() == ground_truth.strip().lower())


DOMAIN_VERIFIERS = {
    "gsm8k":      verify_numeric,
    "aqua_rat":   verify_numeric,
    "mmlu":       verify_letter,
    "strategyqa": verify_yesno,
}
