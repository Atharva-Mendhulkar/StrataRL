import json
import random
import requests

GSM8K_MANUAL = [
    {"prompt": "Janet earns $17 per day. How much does she save in 6 weeks?", "ground_truth": "102"},
    {"prompt": "A store sold 15 apples at $2 each and 8 oranges at $3 each. How much total?", "ground_truth": "54"},
    {"prompt": "If I have 10 apples and give 3 to John, how many do I have left?", "ground_truth": "7"},
    {"prompt": "A train travels 60 miles per hour. How far does it go in 3 hours?", "ground_truth": "180"},
    {"prompt": "What is 15% of 200?", "ground_truth": "30"},
]

def load_domain_samples(domain: str, n: int = 100):
    samples = []
    
    if domain == "gsm8k":
        url = "https://raw.githubusercontent.com/openai/gsm8k/main/gsm8k/data/train.jsonl"
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                for line in r.text.strip().split("\n"):
                    item = json.loads(line)
                    samples.append({
                        "prompt": [
                            {"role": "system", "content": "You are a precise math reasoning assistant. Solve the following problem step by step. Show your internal reasoning process inside <think> tags. Place only your final numeric answer inside <answer> tags."},
                            {"role": "user", "content": f"Question: {item['question']}"}
                        ],
                        "ground_truth": item["answer"].split("####")[-1].strip()
                    })
            else:
                for item in GSM8K_MANUAL:
                    samples.append({
                        "prompt": [
                            {"role": "system", "content": "You are a precise math reasoning assistant. Solve the following problem step by step. Show your internal reasoning process inside <think> tags. Place only your final numeric answer inside <answer> tags."},
                            {"role": "user", "content": f"Question: {item['prompt']}"}
                        ],
                        "ground_truth": item["ground_truth"]
                    })
        except Exception:
            for item in GSM8K_MANUAL:
                samples.append({
                    "prompt": [
                        {"role": "system", "content": "You are a precise math reasoning assistant. Solve the following problem step by step. Show your internal reasoning process inside <think> tags. Place only your final numeric answer inside <answer> tags."},
                        {"role": "user", "content": f"Question: {item['prompt']}"}
                    ],
                    "ground_truth": item["ground_truth"]
                })
            
    elif domain == "mmlu":
        raw_samples = [
            {"prompt": "Which of the following is a prime number? (A) 4 (B) 9 (C) 11 (D) 15", "ground_truth": "C"},
            {"prompt": "What is the capital of France? (A) London (B) Paris (C) Berlin (D) Rome", "ground_truth": "B"},
            {"prompt": "Who wrote 'Romeo and Juliet'? (A) Shakespeare (B) Dickens (C) Austen (D) Twain", "ground_truth": "A"},
            {"prompt": "Which element has atomic number 1? (A) Helium (B) Hydrogen (C) Oxygen (D) Carbon", "ground_truth": "B"},
        ]
        for item in raw_samples:
            samples.append({
                "prompt": [
                    {"role": "system", "content": "You are a knowledgeable assistant. Answer the following multiple choice question. Think step by step inside <think> tags. Then, write only the single letter (A, B, C, or D) corresponding to the correct answer inside <answer> tags."},
                    {"role": "user", "content": f"Question: {item['prompt']}"}
                ],
                "ground_truth": item["ground_truth"]
            })
            
    elif domain == "strategyqa":
        raw_samples = [
            {"prompt": "Would a Venus flytrap survive in the Arctic?", "ground_truth": "no"},
            {"prompt": "Can a shark climb a tree?", "ground_truth": "no"},
            {"prompt": "Is a tomato a fruit?", "ground_truth": "yes"},
            {"prompt": "Does the moon orbit the Earth?", "ground_truth": "yes"},
            {"prompt": "Can humans breathe underwater without equipment?", "ground_truth": "no"},
            {"prompt": "Is the sun a planet?", "ground_truth": "no"},
            {"prompt": "Are dogs mammals?", "ground_truth": "yes"},
            {"prompt": "Can a computer think like a human?", "ground_truth": "no"},
        ]
        for item in raw_samples:
            samples.append({
                "prompt": [
                    {"role": "system", "content": "You are a logical reasoning assistant. Answer the following yes/no question. Explain your reasoning inside <think> tags. Finally, write only 'yes' or 'no' inside <answer> tags."},
                    {"role": "user", "content": f"Question: {item['prompt']}"}
                ],
                "ground_truth": item["ground_truth"]
            })
    
    # Fill up to n
    if len(samples) < n and len(samples) > 0:
        samples = (samples * (n // len(samples) + 1))[:n]
    
    random.seed(42)
    return random.sample(samples, min(n, len(samples)))

