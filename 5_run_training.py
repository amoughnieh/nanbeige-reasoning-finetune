"""
5_training_unsloth.py

Fine-tunes a base LLM using Unsloth and QLoRA.
Implements token-length filtering, sequence packing, and behavioral alignment
on synthetic Chain-of-Thought (CoT) reasoning traces.
"""

import argparse
import random
import os
import json
from datasets import Dataset
import sys

from unsloth import FastLanguageModel
from unsloth_zoo import tokenizer_utils
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer, SFTConfig

# Apply Unsloth patch
tokenizer_utils.fix_untrained_tokens = lambda *args, **kwargs: None


def get_args():
    parser = argparse.ArgumentParser(description="Unsloth QLoRA Fine-Tuning for Behavioral Alignment")

    # Paths
    parser.add_argument("--config", type=str, default="config/config.json", help="Path to JSON config file to override defaults")
    parser.add_argument("--base_model_path", type=str, default=None, help="Path to base model or HF hub ID")
    parser.add_argument("--train_data", type=str, default=None, help="Path to training data JSONL")
    parser.add_argument("--eval_data", type=str, default=None, help="Path to evaluation data JSONL")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the adapter")

    # Hyperparameters
    parser.add_argument("--system_prompt", type=str, default=None, help="System prompt for the model")
    parser.add_argument("--drop_thinking_prob", type=float, default=None, help="Probability of dropping the thinking trace during training.")
    parser.add_argument("--max_train_tokens", type=int, default=None, help="Filter out examples longer than this token count.")
    parser.add_argument("--max_seq_length", type=int, default=None, help="Maximum sequence length for the model.")

    # LoRA Config
    parser.add_argument("--lora_r", type=int, default=None, help="LoRA Rank (lower for behavioral, higher for knowledge)")
    parser.add_argument("--lora_alpha", type=int, default=None, help="LoRA Alpha")
    parser.add_argument("--lora_dropout", type=float, default=None, help="LoRA Dropout (Keep 0 for Unsloth)")

    # Training Config
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size per device")
    parser.add_argument("--grad_accum", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=None, help="Warmup ratio")
    parser.add_argument("--lr_scheduler_type", type=str, default=None, help="The learning rate scheduler (e.g., linear, cosine, constant)")

    return parser.parse_args()

def main(args):
    # --- Setup Real-Time Console Mirroring ---
    os.makedirs(args.output_dir, exist_ok=True)
    log_file_path = os.path.join(args.output_dir, "training_run.log")

    class Tee(object):
        def __init__(self, *files):
            self.files = files

        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()  # Force write to disk immediately

        def flush(self):
            for f in self.files:
                f.flush()

    # Open the log file and redirect stdout/stderr to both console and file
    log_file = open(log_file_path, "a", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    print(f"\n--- Logging started: {log_file_path} ---")
    print(f"--- Training Session Started: {args.output_dir} ---")

    # --- Load Model and Tokenizer ---
    print(f"Loading model and tokenizer from {args.base_model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        trust_remote_code=True,
    )
    print("Model loaded.")

    # --- Attach LoRA Adapters ---
    print("Attaching LoRA adapters...")
    lora_target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
    )
    model.print_trainable_parameters()

    # --- Data Filtering and Formatting Helpers ---
    def format_full_sequence(record):
        title = record.get("title", "")
        body = record.get("question_body", "")
        question = f"{title}\n\n{body}".strip()
        answer = record.get("enriched_answer", "").replace("**", "")
        thinking = record.get("thinking_trace_re", "") or ""

        assistant_content = f"<think>{thinking}</think>\n\n{answer}" if thinking else answer

        messages = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_content},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    def is_within_length(record):
        formatted = format_full_sequence(record)
        tokens = tokenizer(formatted, add_special_tokens=False)["input_ids"]
        return len(tokens) <= args.max_train_tokens

    # --- Load Data ---
    print("\nLoading datasets...")

    train_records_all = []
    eval_records = []

    # 1. Load Training Data
    if os.path.exists(args.train_data):
        with open(args.train_data, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if not r.get("trace_truncated", False):
                        train_records_all.append(r)

    # 2. Load Evaluation Data
    if os.path.exists(args.eval_data):
        with open(args.eval_data, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if not r.get("trace_truncated", False):
                        eval_records.append(r)

    print(f"Raw training records loaded : {len(train_records_all)}")
    print(f"Eval records loaded         : {len(eval_records)}")

    # Apply token length filter
    print(f"Filtering to max_train_tokens = {args.max_train_tokens}...")
    train_records = [r for r in train_records_all if is_within_length(r)]

    with_thinking = sum(1 for r in train_records if r.get("thinking_trace_re", ""))
    without_thinking = len(train_records) - with_thinking

    print(f"Training examples after filter : {len(train_records)}")
    print(f"  With thinking traces         : {with_thinking}")
    print(f"  Without thinking traces      : {without_thinking}")
    print(f"  Excluded (too long)          : {len(train_records_all) - len(train_records)}")

    train_dataset = Dataset.from_list(train_records)
    eval_dataset = Dataset.from_list(eval_records)

    # --- Formatting Function for Trainer ---
    def formatting_func(example):
        outputs = []
        for title, body, answer, thinking in zip(
                example["title"],
                example["question_body"],
                example["enriched_answer"],
                example["thinking_trace_re"],
        ):
            question = f"{title}\n\n{body}".strip()
            answer = (answer or "").replace("**", "")
            thinking = thinking or ""

            use_thinking_tags = bool(thinking)
            if use_thinking_tags and random.random() < args.drop_thinking_prob:
                thinking = ""

            assistant_content = f"<think>{thinking}</think>\n\n{answer}" if use_thinking_tags else answer

            messages = [
                {"role": "system", "content": args.system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": assistant_content},
            ]
            outputs.append(tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            ))
        return outputs

    # --- Training Configuration ---
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        max_seq_length=args.max_seq_length,
        packing=True,
        eval_strategy="steps",
        eval_steps=15,
        save_strategy="steps",
        save_steps=15,
        save_total_limit=2,
        logging_steps=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_pin_memory=False,
        optim="adamw_8bit",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_func,
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    # --- Execute Training ---
    print("\nStarting training...")
    print(f"  Epochs             : {args.epochs}")
    print(f"  Effective batch    : {args.batch_size * args.grad_accum}")
    print(f"  LoRA rank          : {args.lora_r}")
    print(f"  Packing            : True")

    trainer.train()

    # --- Save LoRA Adapter ---
    print(f"\nSaving LoRA adapter to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done. LoRA adapter saved.")


if __name__ == "__main__":
    args = get_args()

    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        config_args = {}
        if "global" in config_data:
            config_args.update(config_data["global"])
        if "training" in config_data:
            config_args.update(config_data["training"])

        for key, value in config_args.items():
            if hasattr(args, key) and getattr(args, key) is None:
                setattr(args, key, value)

    main(args)