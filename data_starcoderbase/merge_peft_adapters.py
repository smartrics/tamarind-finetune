from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str, default="bigcode/large-model")
    parser.add_argument("--peft_model_path", type=str, default="/")
    parser.add_argument("--merged_model_name_or_path", type=str, default="bigcode/large-model-merged")
    parser.add_argument("--push_to_hub", action="store_true", default=True)

    return parser.parse_args()

def main():
    args = get_args()

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        return_dict=True,
        torch_dtype=torch.float16 
    )

    model = PeftModel.from_pretrained(base_model, args.peft_model_path)
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)

    if args.push_to_hub:
        print(f"Saving to hub '{args.merged_model_name_or_path}' ...")
        model.push_to_hub(f"{args.merged_model_name_or_path}")
        tokenizer.push_to_hub(f"{args.merged_model_name_or_path}")
    else:
        model.save_pretrained(f"{args.merged_model_name_or_path}")
        tokenizer.save_pretrained(f"{args.merged_model_name_or_path}")
        print(f"Model saved to '{args.merged_model_name_or_path}'")

if __name__ == "__main__" :
    main()