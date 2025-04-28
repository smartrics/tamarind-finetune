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
        rows.append({"id": idx, "question": item['input'], "response": item["output"]})
    return pd.DataFrame(rows)


# === Load datasets ===
tamarind_data = load_tune_data(tune_input_dir) + load_spec_data(spec_input_dir)

random.shuffle(tamarind_data)

# === Write CSVs ===
prepare_csv_data(tamarind_data).to_csv(output_dir / "tamarind_data.csv", index=False)
