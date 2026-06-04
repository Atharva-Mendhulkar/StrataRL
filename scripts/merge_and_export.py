import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_adapter(base_model_id: str, adapter_path: str, output_path: str):
    print(f"Loading base model: {base_model_id}")
    # Load base model (in fp16/bf16 to save RAM during merge)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("Merging adapter weights into base model...")
    # This physically adds the LoRA matrix (A * B) to the base weights (W)
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to {output_path}...")
    Path(output_path).mkdir(exist_ok=True, parents=True)
    merged_model.save_pretrained(output_path)
    
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.save_pretrained(output_path)
    
    print("✓ Model successfully merged and saved!")
    print("\nNext step: You can convert this folder to GGUF using llama.cpp:")
    print(f"python llama.cpp/convert_hf_to_gguf.py {output_path} --outfile {output_path}/model.gguf --outtype q8_0")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter_path", default="outputs/final")
    parser.add_argument("--output_path", default="outputs/merged")
    args = parser.parse_args()
    merge_adapter(args.base_model, args.adapter_path, args.output_path)
