"""
M4-specific rollout engine.
Replaces vLLM with HuggingFace generate() + MPS backend.
Captures per-token logprobs for use in GRPO loss (replaces reference model).
"""

import torch
import torch.nn.functional as F
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer


class M4RolloutEngine:

    def __init__(self, model, tokenizer, device="mps"):
        self.model     = model.to(device)
        self.tokenizer = tokenizer
        self.device    = device

    @torch.no_grad()
    def generate(
        self,
        prompts:        List[str],
        G:              int   = 4,
        temperature:    float = 0.9,
        top_p:          float = 0.95,
        max_new_tokens: int   = 512,
        min_new_tokens: int   = 20,
    ) -> List[Dict]:
        """
        Generate G completions per prompt and capture per-token logprobs.
        Ensures G valid samples per prompt with retry logic for EOS edge cases.
        """
        self.model.eval()
        
        rollouts = []
        eos_id   = self.tokenizer.eos_token_id

        for prompt in prompts:
            prompt_ids = self.tokenizer(
                prompt, return_tensors="pt", add_special_tokens=True
            ).input_ids.to(self.device)
            prompt_len = prompt_ids.shape[1]

            completions      = []
            all_token_ids    = []
            all_logprobs     = []
            finish_reasons   = []

            # Hard cap on retries to prevent silent training stalls
            valid_count  = 0
            attempts     = 0
            max_attempts = G + 4

            while valid_count < G and attempts < max_attempts:
                attempts += 1
                outputs = self.model.generate(
                    prompt_ids,
                    max_new_tokens    = max_new_tokens,
                    min_new_tokens    = min_new_tokens,
                    do_sample         = True,
                    temperature       = temperature,
                    top_p             = top_p,
                    repetition_penalty= 1.1,
                    return_dict_in_generate = True,
                    output_scores     = True,
                    pad_token_id      = eos_id,
                    eos_token_id      = eos_id,
                    use_cache         = True,
                )

                full_ids     = outputs.sequences[0]
                comp_ids     = full_ids[prompt_len:].tolist()

                # Extract per-token logprobs
                token_logps  = []
                for step_idx, step_scores in enumerate(outputs.scores):
                    if step_idx >= len(comp_ids):
                        break
                    log_probs_step = F.log_softmax(step_scores[0], dim=-1)
                    chosen_token   = comp_ids[step_idx]
                    token_logps.append(log_probs_step[chosen_token].item())

                # ── EOS Truncation Safety ─────────────────────────────────────
                if eos_id in comp_ids:
                    eos_idx = comp_ids.index(eos_id)
                    if eos_idx == 0:
                        # Skip empty reasoning: prevents downstream zero-length errors
                        continue
                    comp_ids    = comp_ids[:eos_idx + 1]
                    token_logps = token_logps[:eos_idx + 1]

                completion_text = self.tokenizer.decode(comp_ids, skip_special_tokens=True)
                finish_reason   = "stop" if comp_ids[-1] == eos_id else "length"

                completions.append(completion_text)
                all_token_ids.append(comp_ids)
                all_logprobs.append(token_logps)
                finish_reasons.append(finish_reason)
                valid_count += 1

            # Fallback if insufficient valid samples: pad with last valid or dummy
            while len(completions) < G:
                if len(completions) > 0:
                    completions.append(completions[-1])
                    all_token_ids.append(all_token_ids[-1])
                    all_logprobs.append(all_logprobs[-1])
                    finish_reasons.append(finish_reasons[-1])
                else:
                    # Emergency fallback if ALL attempts failed
                    dummy_ids = [self.tokenizer.pad_token_id or 0] * min_new_tokens
                    completions.append("empty_fallback")
                    all_token_ids.append(dummy_ids)
                    all_logprobs.append([0.0] * min_new_tokens)
                    finish_reasons.append("fallback")

            rollouts.append({
                "prompt":           prompt,
                "completions":      completions,
                "token_ids":        all_token_ids,
                "rollout_logprobs": all_logprobs,
                "finish_reasons":   finish_reasons,
            })

        self.model.train()
        return rollouts


def build_m4_engine(model_id: str = "Qwen/Qwen2.5-0.5B-Instruct") -> M4RolloutEngine:
    device    = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype = torch.bfloat16,
    )
    return M4RolloutEngine(model, tokenizer, device=device)
