"""
M4-specific rollout engine.
Replaces vLLM with HuggingFace generate() backend.
Captures per-token logprobs for use in GRPO loss.
Optimized for Kaggle P100 / CUDA environments.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer


class M4RolloutEngine:

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        G: int = 4,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        max_new_tokens: int = 256,
        min_new_tokens: int = 20,
    ) -> List[Dict]:
        """
        Generate G completions per prompt and capture per-token logprobs.
        Ensures G valid samples per prompt with retry logic.
        """

        self.model.eval()

        rollouts = []
        eos_id = self.tokenizer.eos_token_id

        for prompt in prompts:

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=True,
                padding=True,
                truncation=True,
            )

            prompt_ids = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)

            prompt_len = prompt_ids.shape[1]

            completions = []
            all_token_ids = []
            all_logprobs = []
            finish_reasons = []

            valid_count = 0
            attempts = 0
            max_attempts = G + 4

            while valid_count < G and attempts < max_attempts:

                attempts += 1

                outputs = self.model.generate(
                    input_ids=prompt_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=1.1,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=eos_id,
                    eos_token_id=eos_id,
                    use_cache=True,
                )

                full_ids = outputs.sequences[0]
                comp_ids = full_ids[prompt_len:].tolist()

                # Extract per-token logprobs
                token_logps = []

                for step_idx, step_scores in enumerate(outputs.scores):

                    if step_idx >= len(comp_ids):
                        break

                    log_probs_step = F.log_softmax(step_scores[0], dim=-1)

                    chosen_token = comp_ids[step_idx]

                    token_logps.append(
                        log_probs_step[chosen_token].item()
                    )

                # EOS truncation safety
                if eos_id in comp_ids:

                    eos_idx = comp_ids.index(eos_id)

                    if eos_idx == 0:
                        continue

                    comp_ids = comp_ids[:eos_idx + 1]
                    token_logps = token_logps[:eos_idx + 1]

                completion_text = self.tokenizer.decode(
                    comp_ids,
                    skip_special_tokens=True
                )

                finish_reason = (
                    "stop"
                    if comp_ids[-1] == eos_id
                    else "length"
                )

                completions.append(completion_text)
                all_token_ids.append(comp_ids)
                all_logprobs.append(token_logps)
                finish_reasons.append(finish_reason)

                valid_count += 1

            # Fallback protection
            while len(completions) < G:

                if len(completions) > 0:

                    completions.append(completions[-1])

                    all_token_ids.append(all_token_ids[-1])

                    all_logprobs.append(all_logprobs[-1])

                    finish_reasons.append(finish_reasons[-1])

                else:

                    dummy_ids = [
                        self.tokenizer.pad_token_id or 0
                    ] * min_new_tokens

                    completions.append("empty_fallback")

                    all_token_ids.append(dummy_ids)

                    all_logprobs.append(
                        [0.0] * min_new_tokens
                    )

                    finish_reasons.append("fallback")

            # Prompt logprobs for invariant checks
            prompt_out = self.model(
                input_ids=prompt_ids,
                attention_mask=attention_mask
            )

            prompt_logits = prompt_out.logits[:, :-1, :]

            prompt_lp_all = F.log_softmax(
                prompt_logits,
                dim=-1
            )

            prompt_labels = prompt_ids[:, 1:]

            prompt_token_lp = prompt_lp_all.gather(
                2,
                prompt_labels.unsqueeze(-1)
            ).squeeze(-1)

            prompt_logprobs = (
                [0.0] + prompt_token_lp[0].tolist()
            )

            rollouts.append({
                "prompt": prompt,
                "completions": completions,
                "token_ids": all_token_ids,
                "rollout_logprobs": all_logprobs,
                "prompt_logprobs": prompt_logprobs,
                "finish_reasons": finish_reasons,
                "completion_start_idx": prompt_len,
                "completion_end_idxs": [
                    prompt_len + len(ids)
                    for ids in all_token_ids
                ]
            })

        self.model.train()

        if self.device == "cuda":
            torch.cuda.empty_cache()

        return rollouts

    @torch.no_grad()
    def generate_for_eval(
        self,
        prompts: List[str],
        temperature: float = 0.0,
        max_tokens: int = 256,
        n: int = 1,
    ) -> List[str]:
        """
        Greedy or sampled generation for evaluation.
        Returns flat list of completions.
        """

        self.model.eval()

        completions = []

        for prompt in prompts:

            inputs = self.tokenizer(
                prompt, return_tensors="pt", add_special_tokens=True
            ).to(self.device)
            prompt_ids = inputs.input_ids
            attention_mask = inputs.attention_mask

            do_sample = temperature > 0
            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "attention_mask": attention_mask,
            }

            if do_sample:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["num_return_sequences"] = n
            else:
                gen_kwargs["top_p"] = None
                gen_kwargs["top_k"] = None
                gen_kwargs["temperature"] = None
            
            outputs = self.model.generate(prompt_ids, **gen_kwargs)

            for output in outputs:

                comp_ids = output[prompt_ids.shape[1]:]

                text = self.tokenizer.decode(
                    comp_ids,
                    skip_special_tokens=True
                )

                completions.append(text)

        self.model.train()

        if self.device == "cuda":
            torch.cuda.empty_cache()

        return completions


def build_m4_engine(
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
) -> M4RolloutEngine:

    device = (
        "cuda"
        if torch.cuda.is_available()
        else (
            "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    )

    model.gradient_checkpointing_enable()

    return M4RolloutEngine(
        model=model,
        tokenizer=tokenizer,
        device=device,
    )