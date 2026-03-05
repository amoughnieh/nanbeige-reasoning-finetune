"""
4_create_splits.py

Splits the final reasoning dataset into training and evaluation sets.
Ensures a randomized, reproducible split and saves them as separate JSONL files.
"""

import argparse
import json
import random
import os

def get_args():
    parser = argparse.ArgumentParser(description="Split JSONL dataset into train and eval sets.")
    parser.add_argument("--config", type=str, default="config/config.json", help="Path to config file")
    parser.add_argument("--input_jsonl", type=str, default=None, help="The final dataset from script 3")
    parser.add_argument("--train_file", type=str, default=None, help="Path to save the training set")
    parser.add_argument("--eval_file", type=str, default=None, help="Path to save the evaluation set")
    parser.add_argument("--eval_size", type=int, default=None, help="Number of examples to put in the eval set")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible splitting")
    return parser.parse_args()

def main(args):
    random.seed(args.seed)

    if not os.path.exists(args.input_jsonl):
        print(f"Error: Input file {args.input_jsonl} not found.")
        return

    # Load all records
    print(f"Loading data from {args.input_jsonl}...")
    with open(args.input_jsonl, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]

    total_count = len(records)
    if total_count <= args.eval_size:
        print(f"Error: Dataset size ({total_count}) is smaller than or equal to requested eval size ({args.eval_size}).")
        return

    # Shuffle and Split
    random.shuffle(records)
    eval_set = records[:args.eval_size]
    train_set = records[args.eval_size:]

    # Save Train set
    with open(args.train_file, 'w', encoding='utf-8') as f:
        for item in train_set:
            f.write(json.dumps(item) + '\n')

    # Save Eval set
    with open(args.eval_file, 'w', encoding='utf-8') as f:
        for item in eval_set:
            f.write(json.dumps(item) + '\n')

    print("\n" + "="*40)
    print("       DATASET SPLIT COMPLETE")
    print("="*40)
    print(f"Total Records:   {total_count}")
    print(f"Training Set:    {len(train_set)} -> {args.train_file}")
    print(f"Evaluation Set:  {len(eval_set)}  -> {args.eval_file}")
    print(f"Random Seed:     {args.seed}")
    print("="*40)


if __name__ == "__main__":
    args = get_args()

    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        config_args = {}
        if "global" in config_data:
            config_args.update(config_data["global"])
        if "splits" in config_data:
            config_args.update(config_data["splits"])

        for key, value in config_args.items():
            if hasattr(args, key) and getattr(args, key) is None:
                setattr(args, key, value)

    main(args)