# tests/test_patch_i2.py
import inspect, pytest
import torch
from training.policy_update import grpo_loss

def test_no_ref_model_parameter():
    sig = inspect.signature(grpo_loss)
    assert "ref_model" not in sig.parameters

def test_old_logprobs_required():
    sig = inspect.signature(grpo_loss)
    assert "old_logprobs" in sig.parameters

def test_raw_kl_mean_in_output():
    src = inspect.getsource(grpo_loss)
    assert '"raw_kl_mean"' in src

def test_prompt_region_assertion_fires():
    import torch
    from unittest.mock import MagicMock
    B, G, seq, vocab = 1, 2, 20, 100
    input_ids       = torch.randint(0, vocab, (B*G, seq))
    attn_mask       = torch.ones(B*G, seq)
    comp_mask       = torch.zeros(B*G, seq)
    comp_mask[:, 10:] = 1
    advantages      = torch.zeros(B*G, seq)
    bad_logprobs    = torch.zeros(B*G, seq)
    bad_logprobs[:, :10] = -0.5   # non-zero in prompt region

    mock_model = MagicMock()
    mock_model.return_value.logits = torch.randn(B*G, seq, vocab)

    with pytest.raises(AssertionError, match="PROMPT-REGION CONTAMINATION"):
        grpo_loss(mock_model, input_ids, attn_mask, comp_mask, advantages, bad_logprobs)
