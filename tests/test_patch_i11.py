from rewards.structural_reward import structural_reward

def test_empty_tags_fail_despite_passing_old_gates():
    """The exact step-500 collapse pattern: tags present, MIN_THINK_CHARS
    cleared via syntax overhead alone, but zero reasoning content."""
    completion = (
        "<think><decompose>5*17</decompose><compute>85</compute>"
        "<verify>85</verify></think><answer>85</answer>"
    )
    assert structural_reward(completion, domain="gsm8k") == 0.4

def test_genuine_short_reasoning_passes():
    """~15+ chars per tag of real content should still pass — I-11 isn't
    punishing brevity, only emptiness."""
    completion = (
        "<think><decompose>Janet earns 17 dollars per day for 5 days</decompose>"
        "<compute>5 times 17 equals 85</compute>"
        "<verify>85 dollars total checks out</verify></think>"
        "<answer>85</answer>"
    )
    assert structural_reward(completion, domain="gsm8k") == 1.0
