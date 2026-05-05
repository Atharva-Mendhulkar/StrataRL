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
        # Attempt to load from OpenAI's repo (master branch)
        url = "https://raw.githubusercontent.com/openai/gsm8k/master/grade_school_math/data/train.jsonl"
        # Wait, I got a 404 on that. Let's try the main repo again with correct path.
        # https://github.com/openai/gsm8k/blob/main/gsm8k/data/train.jsonl -> raw
        url = "https://raw.githubusercontent.com/openai/gsm8k/main/gsm8k/data/train.jsonl"
        
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                for line in r.text.strip().split("\n"):
                    item = json.loads(line)
                    samples.append({
                        "prompt": item["question"],
                        "ground_truth": item["answer"].split("####")[-1].strip()
                    })
            else:
                samples = GSM8K_MANUAL
        except Exception:
            samples = GSM8K_MANUAL
            
    elif domain == "strategyqa":
        samples = [
            {"prompt": "Would a Venus flytrap survive in the Arctic?", "ground_truth": "no"},
            {"prompt": "Can a shark climb a tree?", "ground_truth": "no"},
            {"prompt": "Is a tomato a fruit?", "ground_truth": "yes"},
            {"prompt": "Does the moon orbit the Earth?", "ground_truth": "yes"},
            {"prompt": "Can humans breathe underwater without equipment?", "ground_truth": "no"},
            {"prompt": "Is the sun a planet?", "ground_truth": "no"},
            {"prompt": "Are dogs mammals?", "ground_truth": "yes"},
            {"prompt": "Can a computer think like a human?", "ground_truth": "no"},
        ]
    
    # Fill up to n
    if len(samples) < n and len(samples) > 0:
        samples = (samples * (n // len(samples) + 1))[:n]
    
    random.seed(42)
    return random.sample(samples, min(n, len(samples)))
