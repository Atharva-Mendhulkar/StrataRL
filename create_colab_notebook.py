import json
import re

with open('kaggle_condition_b.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        # Replace Kaggle Dataset copy with Colab Drive mount & copy
        if 'SRC = "/kaggle/input' in ''.join(cell['source']):
            cell['source'] = [
                "from google.colab import drive\n",
                "import os, shutil\n",
                "drive.mount('/content/drive')\n",
                "\n",
                "# Assuming you uploaded the code as StrataRL-main.zip to your Colab root or Drive\n",
                "if not os.path.exists('/content/StrataRL-main'):\n",
                "    !unzip -q /content/drive/MyDrive/StrataRL-main.zip -d /content/\n",
                "print(\"✓ Repository ready\")\n"
            ]
        # Replace PROJECT_ROOT
        source_str = ''.join(cell['source'])
        source_str = source_str.replace('/kaggle/working', '/content')
        
        # Replace Kaggle Secrets with Colab Userdata
        if 'from kaggle_secrets import UserSecretsClient' in source_str:
            cell['source'] = [
                "import os\n",
                "import wandb\n",
                "from google.colab import userdata\n",
                "\n",
                "try:\n",
                "    wandb_api_key = userdata.get('WANDB_API_KEY')\n",
                "    wandb.login(key=wandb_api_key)\n",
                "    print(\"✓ wandb logged in\")\n",
                "except Exception as e:\n",
                "    print(\"⚠️ Could not authenticate with Weights & Biases.\")\n",
                "    os.environ[\"WANDB_DISABLED\"] = \"true\"\n"
            ]
        else:
            # Reconstruct list of strings
            lines = source_str.split('\n')
            cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []

with open('colab_condition_b.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Created colab_condition_b.ipynb")
