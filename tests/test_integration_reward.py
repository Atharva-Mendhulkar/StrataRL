"""
Tests the full reward pipeline: rollout → reward engine → SAN → advantages
using realistic (not synthetic) completions from the 0.5B model.
"""

import torch
from m4.m4_rollout_engine import build_m4_engine
from rewards.reward_engine  import score_batch
from training.advantage     import compute_san_advantages

GSMS_PROMPTS = [
    ("A store sold 15 apples at $2 each and 8 oranges at $3 each. How much total?",
     "54", "gsm8k"),
    ("Janet saves $17 per week. How much does she save in 6 weeks?",
     "102", "gsm8k"),
]

STRAT_PROMPTS = [
    ("Would a Venus flytrap survive in the Arctic?", "no", "strategyqa"),
]

def test_full_reward_pipeline():
    engine = build_m4_engine()

    all_prompts = [p for p, _, _ in GSMS_PROMPTS + STRAT_PROMPTS]
    all_gts     = [g for _, g, _ in GSMS_PROMPTS + STRAT_PROMPTS]
    all_domains = [d for _, _, d in GSMS_PROMPTS + STRAT_PROMPTS]

    # Format prompts with domain templates
    formatted = [_format_prompt(p, d) for p, d in zip(all_prompts, all_domains)]

    rollouts = engine.generate(formatted, G=4, max_new_tokens=200)

    gdpo_rewards, raw_rewards = score_batch(rollouts, all_gts, all_domains)

    # Shape checks
    B, G = len(rollouts), 4
    assert gdpo_rewards.shape == (B, G)
    assert raw_rewards.shape  == (3, B, G)

    # Value range checks
    assert not torch.isnan(gdpo_rewards).any(), "NaN in GDPO rewards"
    assert not torch.isinf(gdpo_rewards).any(), "Inf in GDPO rewards"

    # SAN advantages
    advantages = compute_san_advantages(gdpo_rewards, all_domains)
    assert advantages.shape == (B, G)
    assert advantages.max().item() <= 5.01
    assert advantages.min().item() >= -5.01
    assert not torch.isnan(advantages).any()

    print("\n✓ Full reward pipeline integration test passed")
    print(f"  Mean outcome reward: {raw_rewards[0].mean():.3f}")
    print(f"  Mean structural reward: {raw_rewards[1].mean():.3f}")
    print(f"  Mean GDPO reward: {gdpo_rewards.mean():.3f}")


def _format_prompt(question: str, domain: str) -> str:
    template_instructions = {
        "gsm8k": "Use <decompose>, <compute>, <verify> tags inside <think>.",
        "strategyqa": "Use <decompose>, <resolve>, <synthesize> tags inside <think>.",
        "mmlu": "Use <recall>, <evaluate> tags inside <think>.",
    }
    inst = template_instructions.get(domain, "Show your reasoning.")
    return (
        f"<|system|>\nYou are a careful reasoning assistant. {inst}\n"
        f"Format: <think>...</think><answer>...</answer>\n"
        f"<|user|>\n{question}\n<|assistant|>\n"
    )


if __name__ == "__main__":
    test_full_reward_pipeline()
