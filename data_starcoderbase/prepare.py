import json
import random
from pathlib import Path
import pandas as pd

# === CONFIGURATION ===
tune_input_dir = Path(".data/test_data_1").absolute()
spec_input_dir = Path(".data/spec_data").absolute()
output_dir = Path("data_starcoderbase")
output_dir.mkdir(parents=True, exist_ok=True)

ADD_PROMPT = False

def add_prompt(prompt, record):
    if ADD_PROMPT:
        return {"input": prompt + "\n\n" + record["input"], "output": record["output"]}
    return record

def load_tune_data(input_dir):
    prompt = (input_dir / "prompt.md").read_text()
    pairs = []
    for file in input_dir.glob("*.json"):
        if file.name == "prompt.md":
            continue
        data = json.loads(file.read_text())
        for content in data.values():
            user_input = json.dumps(
                {
                    "metadata": content.get("metadata", {}),
                    "instructions": content.get("instructions", []),
                },
                separators=(",", ":"),
            )
            model_output = json.dumps(
                {"workflow": content.get("workflow", [])}, separators=(",", ":")
            )
            pairs.append(
                add_prompt(prompt, {"input": user_input, "output": model_output})
            )
    return pairs


def load_spec_data(spec_dir):
    prompt = (spec_dir / "prompt.md").read_text()
    spec_pairs = []
    for file in spec_dir.glob("*.json"):
        records = json.loads(file.read_text())
        for r in records:
            spec_pairs.append(add_prompt(prompt, r))
    return spec_pairs


def prepare_csv_data(dataset):
    rows = []
    for idx, item in enumerate(dataset):
        content = f"<fim_prefix>{item['input']}<fim_suffix><fim_middle>{item['output']}"
        rows.append({"id": idx, "content": content})
    return pd.DataFrame(rows)


# === Load datasets ===
tune_data = load_tune_data(tune_input_dir)
spec_data = load_spec_data(spec_input_dir)

random.shuffle(tune_data)
random.shuffle(spec_data)

# === Split datasets ===
def split_data(data):
    n = len(data)
    train_end = int(n * 0.8)
    test_end = int(n * 0.9)
    return data[:train_end], data[train_end:test_end], data[test_end:]

tune_train_data, tune_test_data, tune_val_data = split_data(tune_data)
spec_train_data, spec_test_data, spec_val_data = split_data(spec_data)

# === Combine datasets ===
combined_train = tune_train_data + spec_train_data
combined_test = tune_test_data + spec_test_data
combined_val = tune_val_data

random.shuffle(combined_train)
random.shuffle(combined_test)
random.shuffle(combined_val)

# === Write CSVs ===
prepare_csv_data(combined_train).to_csv(output_dir / "training_data.csv", index=False)
prepare_csv_data(combined_test).to_csv(output_dir / "test_data.csv", index=False)
prepare_csv_data(combined_val).to_csv(output_dir / "validation_data.csv", index=False)
