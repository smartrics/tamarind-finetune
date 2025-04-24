import os
import json

def find_content_lens(j):
    i = 0
    o = 0
    for m in j["messages"]:
        if m["role"] in ("system", "user") :
            i = i + len(m["content"])
        if m["role"] in ("assistant") :
            o = o + len(m["content"])
    return i, o

def find_max_line_length(directory):
    file_max_lengths = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if "jsonl" not in file:
                continue
            file_path = os.path.join(root, file)
            max_i_length = 0
            max_o_length = 0
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        j_line = json.loads(line)
                        i, o = find_content_lens(j_line)
                        max_i_length = max(max_i_length, i)
                        max_o_length = max(max_o_length, o)
                file_max_lengths[file_path] = (max_i_length, max_o_length)
            except (UnicodeDecodeError, IOError):
                print(f"Could not read file: {file_path}")
    return file_max_lengths

if __name__ == "__main__":
    max_length = find_max_line_length("./data_mistral")
    
    print(f"The maximum line length in all files is: {max_length}")