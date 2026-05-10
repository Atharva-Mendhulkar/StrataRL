"""
Kaggle/CUDA rollout engine.
Replaces m4/m4_rollout_engine.py for production P100/T4 GPU runs.
Uses HuggingFace generate() with BitsAndBytes 4-bit quantization.
vLLM integration can be dropped in here by swapping generate() call.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


class KaggleRolloutEngine:

    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model     = model
        self.tokenizer = tokenizer
        self.device    = device

    @torch.no_grad()
    def generate(
        self,
        prompts:        List[str],
        G:              int   = 8,
        temperature:    float = 0.85,
        top_p:          float = 0.95,
        max_new_tokens: int   = 2048,
        min_new_tokens: int   = 100,
    ) -> List[Dict]:
        self.model.eval()
        rollouts = []
        eos_id   = self.tokenizer.eos_token_id

        for prompt in prompts:
            inputs = self.tokenizer(
                prompt, return_tensors="pt", add_special_tokens=True
            ).to(self.device)
            prompt_ids = inputs.input_ids
            attention_mask = inputs.attention_mask
            prompt_len = prompt_ids.shape[1]

            completions    = []
            all_token_ids  = []
            all_logprobs   = []
            finish_reasons = []

            valid_count  = 0
            attempts     = 0
            max_attempts = G + 4

            while valid_count < G and attempts < max_attempts:
                attempts += 1
                num_to_generate = G - valid_count
                outputs = self.model.generate(
                    prompt_ids,
                    attention_mask          = attention_mask,
                    max_new_tokens          = max_new_tokens,
                    min_new_tokens          = min_new_tokens,
                    do_sample               = True,
                    temperature             = temperature,
                    top_p                   = top_p,
                    num_return_sequences    = num_to_generate,
                    repetition_penalty      = 1.1,
                    return_dict_in_generate = True,
                    output_scores           = True,
                    pad_token_id            = eos_id,
                    eos_token_id            = eos_id,
                    use_cache               = True,
                    cache_implementation    = "static",
                )

                for seq_idx in range(num_to_generate):
                    full_ids = outputs.sequences[seq_idx]
                    comp_ids = full_ids[prompt_len:].tolist()

                    token_logps = []
                    for step_idx, step_scores in enumerate(outputs.scores):
                        if step_idx >= len(comp_ids):
                            break
                        log_probs_step = F.log_softmax(step_scores[seq_idx], dim=-1)
                        chosen_token   = comp_ids[step_idx]
                        token_logps.append(log_probs_step[chosen_token].item())

                    # EOS truncation safety
                    if eos_id in comp_ids:
                        eos_idx = comp_ids.index(eos_id)
                        if eos_idx == 0:
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

            while len(completions) < G:
                if len(completions) > 0:
                    completions.append(completions[-1])
                    all_token_ids.append(all_token_ids[-1])
                    all_logprobs.append(all_logprobs[-1])
                    finish_reasons.append(finish_reasons[-1])
                else:
                    dummy_ids = [self.tokenizer.pad_token_id or 0] * min_new_tokens
                    completions.append("empty_fallback")
                    all_token_ids.append(dummy_ids)
                    all_logprobs.append([0.0] * min_new_tokens)
                    finish_reasons.append("fallback")

            rollouts.append({
                "prompt":               prompt,
                "completions":          completions,
                "token_ids":            all_token_ids,
                "rollout_logprobs":     all_logprobs,
                "finish_reasons":       finish_reasons,
                "completion_start_idx": prompt_len,
                "completion_end_idxs":  [prompt_len + len(ids) for ids in all_token_ids],
            })

        self.model.train()
        return rollouts

    @torch.no_grad()
    def generate_for_eval(
        self,
        prompts:     List[str],
        temperature: float = 0.0,
        max_tokens:  int   = 512,
        n:           int   = 1,
    ) -> List[str]:
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
                text = self.tokenizer.decode(comp_ids, skip_special_tokens=True)
                completions.append(text)

        self.model.train()
        return completions


def build_kaggle_engine(
    model_id:     str  = "Qwen/Qwen2.5-3B-Instruct",
    load_in_4bit: bool = True,
) -> KaggleRolloutEngine:
    """
    Builds the CUDA rollout engine with optional 4-bit QLoRA quantization.
    load_in_4bit=True is required for G=8 on a P100 (16GB VRAM).
    """
    device    = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit              = True,
            bnb_4bit_compute_dtype    = torch.float16,
            bnb_4bit_use_double_quant = True,
            bnb_4bit_quant_type       = "nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config = bnb_config,
            torch_dtype         = torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype = torch.float16,
        ).to(device)

    return KaggleRolloutEngine(model, tokenizer, device=device)
