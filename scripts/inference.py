import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_reasoning(prompt: str, base_model_id: str = "Qwen/Qwen2.5-3B-Instruct", adapter_path: str = "outputs/final"):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    print("Loading base model (bf16/fp16)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("Loading adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    # Apply template (modify as needed for specific benchmarks)
    formatted_prompt = f"<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n<think>\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    print("Generating response (this may take a moment)...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode and print
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print("\n--- MODEL OUTPUT ---")
    print(f"<think>\n{response}")
    print("--------------------\n")

if __name__ == "__main__":
    example_question = "If I have 15 apples and eat 2, then buy 5 more, but drop half of my total apples, how many do I have left?"
    generate_reasoning(example_question)
