import argparse
import os

import torch
from accelerate import Accelerator
from datasets import load_dataset
import time
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict
from torch.utils.data import IterableDataset
from tqdm import tqdm
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, logging, set_seed
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


def patched_load_rng_state(self, checkpoint_folder):
    import torch
    import os
    rng_file = os.path.join(checkpoint_folder, "rng_state.pth")
    if os.path.isfile(rng_file):
        try:
            checkpoint_rng_state = torch.load(rng_file, weights_only=False)  # <-- PATCHED HERE
            self._rng_state = checkpoint_rng_state
        except Exception as e:
            print(f"⚠️ Failed to load RNG state from checkpoint. Continuing without it. Error: {e}")

Trainer._load_rng_state = patched_load_rng_state

"""
Fine-Tune StarCoder on Code Alpaca/SE
"""


class ETACallback(TrainerCallback):
    def __init__(self):
        self.start_time = None

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.start_time is None:
            self.start_time = time.time()

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if self.start_time is None or state.global_step == 0:
            return

        elapsed = time.time() - self.start_time
        steps_completed = state.global_step
        total_steps = args.max_steps

        percent_done = steps_completed / total_steps
        estimated_total_time = elapsed / percent_done
        eta = estimated_total_time - elapsed

        def hms(seconds):
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            return f"{h}h {m}m {s}s"

        print(f"[ETA] {percent_done:.1%} complete — Elapsed: {hms(elapsed)}, Remaining: {hms(eta)}")


class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        kwargs["model"].save_pretrained(checkpoint_folder)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        torch.save({}, pytorch_model_path)
        return control

try:
    from safetensors.torch import load_file as safe_load
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

class LoadBestPeftModelCallback(TrainerCallback):
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        print(f"Loading best peft model from {state.best_model_checkpoint} (score: {state.best_metric}).")

        checkpoint_dir = state.best_model_checkpoint
        model = kwargs["model"]

        safetensors_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
        bin_path = os.path.join(checkpoint_dir, "adapter_model.bin")

        if os.path.exists(safetensors_path) and SAFETENSORS_AVAILABLE:
            print(f"Loading adapter weights from {safetensors_path}")
            adapters_weights = safe_load(safetensors_path)
        elif os.path.exists(bin_path):
            print(f"Loading adapter weights from {bin_path}")
            adapters_weights = torch.load(bin_path)
        else:
            raise FileNotFoundError(
                f"No adapter model file found in {checkpoint_dir}. Expected one of: adapter_model.safetensors or adapter_model.bin"
            )

        set_peft_model_state_dict(model, adapters_weights)
        return control
    
class LoadBestPeftModelCallbackOld(TrainerCallback):
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        print(f"Loading best peft model from {state.best_model_checkpoint} (score: {state.best_metric}).")
        best_model_path = os.path.join(state.best_model_checkpoint, "adapter_model.bin")
        adapters_weights = torch.load(best_model_path)
        model = kwargs["model"]
        set_peft_model_state_dict(model, adapters_weights)
        return control
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="bigcode/large-model")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceH4/CodeAlpaca_20K")
    parser.add_argument("--dataset_path", type=str, default="./dataset.csv")
    parser.add_argument("--dataset_type", type=str, default="csv")
    parser.add_argument("--subset", type=str)
    parser.add_argument("--split", type=str)
    parser.add_argument("--size_valid_set", type=int, default=10000)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--shuffle_buffer", type=int, default=5000)

    parser.add_argument("--input_column_name", type=str, default="prompt")
    parser.add_argument("--output_column_name", type=str, default="completion")

    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--eos_token_id", type=int, default=49152)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--no_fp16", action="store_false")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", action="store_false", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_freq", default=100, type=int)
    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument("--save_freq", default=1000, type=int)

    return parser.parse_args()


def chars_token_ratio(dataset, tokenizer, input_column_name="prompt", output_column_name="completion", nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example, input_column_name, output_column_name)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_sample_text(example, input_column_name="prompt", output_column_name="completion"):
    """Prepare the text from a sample of the dataset."""
    text = f"Question: {example[input_column_name]}\n\nAnswer: {example[output_column_name]}"
    return text


class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        input_column_name="prompt",
        output_column_name="completion"
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else args.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.input_column_name = input_column_name
        self.output_column_name = output_column_name

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(prepare_sample_text(next(iterator), self.input_column_name, self.output_column_name))
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    self.current_size += 1
                    yield {
                        "input_ids": torch.LongTensor(input_ids),
                        "labels": torch.LongTensor(input_ids),
                    }

def create_datasets(tokenizer, args):
    if args.dataset_path:
        ext = args.dataset_type
        if not ext:
            ext = os.path.splitext(args.dataset_path)[1][1:]
        dataset = load_dataset(ext, data_files=args.dataset_path, split=args.split)
        dataset = dataset.train_test_split(test_size=0.1, seed=args.seed)
    else:
        dataset = load_dataset(
            args.dataset_name,
            data_dir=args.subset,
            split=args.split,
            use_auth_token=True,
            num_proc=args.num_workers if not args.streaming else None,
            streaming=args.streaming,
        )


    if args.streaming:
        print("Loading the dataset in streaming mode")
        valid_data = dataset.take(args.size_valid_set)
        train_data = dataset.skip(args.size_valid_set)
        train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
    else:
        train_data = dataset["train"]
        valid_data = dataset["test"]
        print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    chars_per_token = chars_token_ratio(train_data, tokenizer, args.input_column_name, args.output_column_name)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
        input_column_name=args.input_column_name,
        output_column_name=args.output_column_name
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
        input_column_name=args.input_column_name,
        output_column_name=args.output_column_name
    )
    return train_dataset, valid_dataset


def run_training(args, train_data, val_data):
    print("Loading the model")
    # disable caching mechanism when using gradient checkpointing
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        use_auth_token=True,
        use_cache=not args.no_gradient_checkpointing,
        load_in_8bit=True,
        device_map={"": Accelerator().process_index},
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules = ["c_proj", "c_attn", "q_attn"]
    )

    model = get_peft_model(model, lora_config)

    print_trainable_parameters(model)

    train_data.start_iteration = 0

    print("Starting main loop")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        max_steps=args.max_steps,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=args.log_freq,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        fp16=not args.no_fp16,
        bf16=args.bf16,
        weight_decay=args.weight_decay,
        run_name="StarCoder-finetuned",
        report_to="wandb",
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(model=model, 
                    args=training_args, 
                    train_dataset=train_data, 
                    eval_dataset=val_data, 
                    callbacks=[
                        SavePeftModelCallback, 
                        LoadBestPeftModelCallback])

    print("Training...")
    import glob

    # Check for existing checkpoints
    checkpoints = sorted(
        glob.glob(os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-*")),
        key=lambda x: int(x.split("-")[-1]),
    )

    if checkpoints:
        last_checkpoint = checkpoints[-1]
        print(f"Found existing checkpoint at {last_checkpoint}. Resuming training...")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("No checkpoint found. Starting training from scratch.")
        trainer.train()

    print("Saving last checkpoint of the model")
    model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_auth_token=True)
    train_dataset, eval_dataset = create_datasets(tokenizer, args)
    run_training(args, train_dataset, eval_dataset)


if __name__ == "__main__":
    args = get_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    main(args)