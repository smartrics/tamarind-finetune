import json
import random
from pathlib import Path
import pandas as pd

# === CONFIGURATION ===
tune_input_dir = Path(".data/test_data_1").absolute()
spec_input_dir = Path(".data/spec_data").absolute()


def add_prompt(prompt, record):
    return {
        "input": prompt + "\n\n" + record["input"],
        "output": record["output"]
    }

# === Load and process wf_*.json files ===
def load_tune_data(input_dir):
    jsonl_pairs = []
    prompt = ""
    with open(input_dir / "prompt.md", "r") as f:
        prompt = f.read()
        
    for file in input_dir.glob("*.json"):
        if file.name == "prompt.md":
            continue
        with open(file, "r") as f:
            data = json.load(f)

        for k in data.keys():
            content = data[k]

            instructions = content.get("instructions", [])
            workflow = content.get("workflow", [])
            metadata = content.get("metadata", {})

            user_input = json.dumps({
                "metadata": metadata,
                "instructions": instructions
            }, separators=(',', ':'))

            model_output = json.dumps({
                "workflow": workflow
            }, separators=(',', ':'))

            jsonl_pairs.append(add_prompt(prompt, {
                "input": user_input,
                "output": model_output
            }))
    return jsonl_pairs

# === Load and process spec*.json + validity_dataset.py ===
def load_spec_data(spec_dir):
    spec_pairs = []

    prompt = ""
    with open(spec_dir / "prompt.md", "r") as f:
        prompt = f.read()


    # Load spec*.json files
    for file in spec_dir.glob("spec*.json"):
        with open(file, "r") as f:
            records = json.load(f)
            for r in records:
                spec_pairs.append(add_prompt(prompt, r))

    # Load validation_data from validity_dataset.py
    spec_val = []
    val_path = spec_dir / "validity_dataset.json"
    if val_path.exists():
        with open(val_path, "r") as f:
            loaded = json.load(f)
            for r in loaded:
                spec_val.append(add_prompt(prompt, r))

    return spec_pairs, spec_val

# === Load datasets ===
tune_data = load_tune_data(tune_input_dir)
spec_data, spec_val_data = load_spec_data(spec_input_dir)

# === Reserve 10% of spec_data for test, then merge the rest ===
random.shuffle(spec_data)
n_spec = len(spec_data)
spec_test_end = int(n_spec * 0.1)
spec_test_data = spec_data[:spec_test_end]
spec_train_data = spec_data[spec_test_end:]

n_tune = len(tune_data)
tune_train_end = int(n_tune * 0.8)
tune_test_end = int(n_tune * 0.9)

tune_train_data = tune_data[:tune_train_end]
tune_test_data = tune_data[tune_train_end:tune_test_end]
tune_val_data = tune_data[tune_test_end:]


# Combine tune + spec train data
combined_train = tune_train_data + spec_train_data
combined_test = tune_test_data + spec_test_data
combined_val = tune_val_data + spec_val_data

# Shuffle everything
random.shuffle(combined_train)
random.shuffle(combined_test)
random.shuffle(combined_val)

# Compute max lengths
max_input_len = max([len(d["input"]) for d in combined_train + combined_test + combined_val], default=0)
max_output_len = max([len(d["output"]) for d in combined_train + combined_test + combined_val], default=0)

min_input_len = min([len(d["input"]) for d in combined_train + combined_test + combined_val], default=0)
min_output_len = min([len(d["output"]) for d in combined_train + combined_test + combined_val], default=0)

for d in combined_train + combined_test + combined_val:
    if len(d["input"]) == min_input_len:
        print(d)

for d in combined_train + combined_test + combined_val:
    if len(d["output"]) == min_output_len:
        print(d)

output_train_path = Path("data_codet5/training_data.jsonl")
output_test_path = Path("data_codet5/test_data.jsonl")
output_val_path = Path("data_codet5/validation_data.jsonl")

def write_jsonl_file(path, data):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, separators=(',', ':')) + "\n")

write_jsonl_file(output_train_path, combined_train)
write_jsonl_file(output_test_path, combined_test)
write_jsonl_file(output_val_path, combined_val)

# Display summary
df = pd.DataFrame({
    "Set": [
        "WF_Data","SP_Data",
        "WF_Training", "WF_Test", "WF_Validation",
        "SP_Training", "SP_Test", "SP_Validation",
        "Training", "Test", "Validation", 
        "Max Input Length", "Max Output Length",
        "Min Input Length", "Min Output Length"
        ],
    "Count": [
        len(tune_data), len(spec_data), 
        len(tune_train_data), len(tune_test_data), len(tune_val_data), 
        len(spec_train_data), len(spec_test_data), len(spec_val_data), 
        len(combined_train), len(combined_test), len(combined_val), 
        max_input_len, max_output_len,
        min_input_len, min_output_len
        ]
})

print(df)
