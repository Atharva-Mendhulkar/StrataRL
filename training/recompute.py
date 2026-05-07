# training/recompute.py

import torch
import torch.nn.functional as F
from typing import Dict

RECOMPUTE_INTERVAL     = 25
DRIFT_WARN_THRESHOLD   = 0.5
DRIFT_ABORT_THRESHOLD  = 2.0


def should_recompute(step: int) -> bool:
    return step % RECOMPUTE_INTERVAL == 0


@torch.no_grad()
def teacher_forced_recompute(
    policy_model,
    input_ids:       torch.Tensor,
    attention_mask:  torch.Tensor,
    completion_mask: torch.Tensor,
    old_logprobs:    torch.Tensor,
    step:            int,
) -> Dict:
    policy_model.eval()

    outputs     = policy_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits      = outputs.logits[:, :-1, :]
    labels      = input_ids[:, 1:]
    log_probs   = F.log_softmax(logits, dim=-1)
    policy_logp = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

    old_aligned = old_logprobs[:, 1:]
    comp_mask   = completion_mask[:, 1:]

    diff        = (policy_logp - old_aligned).abs()
    drift_mean  = (diff * comp_mask).sum() / (comp_mask.sum() + 1e-8)
    drift_max   = diff.max()

    drift_mean_val = drift_mean.item()
    status = "ABORT" if drift_mean_val > DRIFT_ABORT_THRESHOLD else \
             "WARN"  if drift_mean_val > DRIFT_WARN_THRESHOLD  else "OK"

    policy_model.train()

    return {
        "recompute_drift_mean": drift_mean_val,
        "recompute_drift_max":  drift_max.item(),
        "recompute_status":     status,
        "recompute_step":       step,
    }
