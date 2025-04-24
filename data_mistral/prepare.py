import json
import os
import random
import hashlib

TAMARIND_PATH = "C:/Users/fab_c/work/github/smartrics/tamarind"
TRAINING_PATH = f"{TAMARIND_PATH}/apps/training"

WF_DATA_PATH = f"{TRAINING_PATH}/data/test_data_1"
SPEC_DATA_PATH = f"{TRAINING_PATH}/data/spec_data"

WF_PROMPT_FILES = [f"{WF_DATA_PATH}/prompt.md", f"{TAMARIND_PATH}/WORKFLOW_SPEC.md"]
SPEC_PROMPT_FILES = [f"{SPEC_DATA_PATH}/prompt.md" ]

LOCAL_DATA_PATH = "./data"


def make_system_prompt(p_list):
    # Load and concatenate system prompt from files
    system_prompt_content = ""
    for file in p_list:
        if os.path.exists(file):
            with open(file, "r", encoding="utf-8") as f:
                system_prompt_content += f.read() + "\n\n"
    return system_prompt_content

def system_message(p_list) :
    return {
    "role": "system",
    "content": make_system_prompt(p_list).strip(),  # Ensures the prompt is cleanly formatted
}

def spec_load_data():
    data_array = []
    for filename in os.listdir(SPEC_DATA_PATH):
        if filename.startswith("spec_") and filename.endswith(".json"):
            file_path = os.path.join(SPEC_DATA_PATH, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                content = json.load(file)
                for elem in content:
                    _in = elem.get("input")
                    _out = elem.get("output")
                    tot = _in + _out
                    val = {
                        "id": str(hashlib.sha256(tot.encode()).hexdigest()),
                        "input": _in,
                        "output": _out,
                    }
                    data_array.append(val)
                print(f"Loaded {filename} - list size: {len(content)} - total size: {len(data_array)}")

    random.shuffle(data_array)
    return data_array


def wf_load_data():
    # Load JSON files into data_dict
    data_array = []

    for filename in os.listdir(WF_DATA_PATH):
        if filename.startswith("data_") and filename.endswith(".json"):
            file_path = os.path.join(WF_DATA_PATH, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                content: dict = json.load(file)
                for key in content:
                    val = {
                        "instructions": content[key]["instructions"],
                        "metadata": content[key]["metadata"],
                        "workflow": content[key]["workflow"],
                        "id": key
                    }
                    data_array.append(val)
                print(f"Loaded {filename} - dict size: {len(content)} - total size: {len(data_array)}")

    random.shuffle(data_array)
    return data_array

def spec_load_validity_data():
    data_array = []
    file_path = os.path.join(SPEC_DATA_PATH, "validity_dataset.json")
    with open(file_path, "r", encoding="utf-8") as file:
        content: list = json.load(file)
        for elem in content:
            _in = elem.get("input")
            _out = elem.get("output")
            tot = _in + _out
            val = {
                "id": str(hashlib.sha256(tot.encode()).hexdigest()),
                "input": _in,
                "output": _out,
            }
            data_array.append(val)
        print(f"Loaded {file_path} - list size: {len(content)} - total size: {len(data_array)}")

    return data_array

def spec_process(arr: list):
    df_formatted = []
    for pt in arr:
        obj = {}
        obj["id"] = pt["id"]
        obj["messages"] = []
        obj["messages"].append(system_message(SPEC_PROMPT_FILES))
        obj["messages"].append(
            {
                "role": "user",
                "content": pt["input"],
            }
        )
        obj["messages"].append(
            {
                "role": "assistant",
                "content": pt["output"],
            }
        )
        df_formatted.append(obj)
    return df_formatted

# Convert data_dict to the required format
def wf_process(arr: list):
    df_formatted = []

    for pt in arr:
        obj = {}
        obj["id"] = pt["id"]
        obj["messages"] = []
        obj["messages"].append(system_message(WF_PROMPT_FILES))
        obj["messages"].append(
            {
                "role": "user",
                "content": f"""
                ### Input:
                {json.dumps(pt["instructions"])}

                ### Context:
                {json.dumps(pt["metadata"])}

                ### Response:
                """,
            }
        )
        obj["messages"].append(
            {
                "role": "assistant",
                "content": json.dumps(pt["workflow"]),
            }
        )
        df_formatted.append(obj)
    return df_formatted


data_array = wf_load_data()
total_count = len(data_array)
train_count = int(0.8 * total_count)
val_count = int(0.1 * total_count)

data_array = wf_process(data_array)

training_data = data_array[:train_count]
validation_data = data_array[train_count:train_count + val_count]
test_data = data_array[train_count + val_count:]

def to_jsonl(json_array, filename):
    with open(filename, "w") as f:
        for item in json_array:
            f.write(json.dumps(item, separators=(",", ":")) + "\n")

to_jsonl(training_data, f"{LOCAL_DATA_PATH}/wf_training_data.jsonl")
to_jsonl(validation_data, f"{LOCAL_DATA_PATH}/wf_validation_data.jsonl")
to_jsonl(test_data, f"{LOCAL_DATA_PATH}/wf_test_data.jsonl")

print(f"wf data. training_len={len(training_data)}, test_len={len(test_data)}, valdation_len={len(validation_data)} ")

data_array = spec_load_data()
total_count = len(data_array)
train_count = int(0.9 * total_count)
data_array = spec_process(data_array)
training_data = data_array[:train_count]
test_data = data_array[train_count:]
validation_data = spec_process(spec_load_validity_data())

to_jsonl(training_data, f"{LOCAL_DATA_PATH}/spec_training_data.jsonl")
to_jsonl(validation_data, f"{LOCAL_DATA_PATH}/spec_validation_data.jsonl")
to_jsonl(test_data, f"{LOCAL_DATA_PATH}/spec_test_data.jsonl")

print(f"spec data. training_len={len(training_data)}, test_len={len(test_data)}, valdation_len={len(validation_data)} ")

for f in ["test_data.jsonl", "training_data.jsonl", "validation_data.jsonl"]:
    with open(f"{LOCAL_DATA_PATH}/{f}", "w") as dest:
        wf = open(f"{LOCAL_DATA_PATH}/wf_{f}", "r").read()
        spec = open(f"{LOCAL_DATA_PATH}/spec_{f}", "r").read()
        dest.write(wf)
        dest.write(spec)
