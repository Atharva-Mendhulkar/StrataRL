import sys

with open("m4/m4_rollout_engine.py", "r") as f:
    lines = f.readlines()

new_lines = []
skip = False

for i, line in enumerate(lines):
    if "token_logps = []" in line:
        # We need to insert the torch.no_grad block right BEFORE the seq_idx loop
        # Wait, it's easier to find the outputs.sequences block
        pass

