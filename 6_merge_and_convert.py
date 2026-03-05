"""
6_merge_and_convert.py
Finalized version: Internal execution via runpy to bypass Miniconda ghosts
and absolute pathing for Google Drive stability.
"""

import argparse
import os
import sys
import subprocess
import time
import runpy
import json
from unsloth import FastLanguageModel
from transformers import AutoTokenizer


def get_args():
    parser = argparse.ArgumentParser(description="Merge LoRA and orchestrate GGUF conversion")
    parser.add_argument("--config", type=str, default="config/config.json", help="Path to config file")
    parser.add_argument("--base_model_path", type=str, default=None, help="Path to the base model")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to the LoRA adapter to merge")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the merged model")
    parser.add_argument("--quantization", type=str, default=None, help="GGUF quantization format (e.g., Q6_K)")
    parser.add_argument("--max_seq_length", type=int, default=None, help="Maximum sequence length")
    return parser.parse_args()


def run_command(cmd, description):
    print(f"\n--- {description} ---")
    print(f"Executing: {' '.join(cmd)}")
    # shell=False is safer for absolute paths once we are in the correct venv
    result = subprocess.run(cmd, capture_output=False, text=True, shell=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")
    print("--- Done ---")


def main(args):
    # 1. Setup Absolute Paths & Drive Sync
    abs_output_dir = os.path.abspath(args.output_dir)
    args.adapter_path = os.path.abspath(args.adapter_path)

    if not os.path.exists(abs_output_dir):
        os.makedirs(abs_output_dir, exist_ok=True)
        print(f">>> Created directory on G: Drive: {abs_output_dir}")
        time.sleep(3)  # Wait for Virtual Drive metadata to catch up

    quant_format = args.quantization.upper()
    abs_bf16_path = os.path.abspath(os.path.join(abs_output_dir, "model-bf16.gguf"))
    abs_final_path = os.path.abspath(os.path.join(abs_output_dir, f"model-{quant_format}.gguf"))

    # 2. Merge Weights via Unsloth
    print(f"Loading model with adapter from: {args.adapter_path}...")
    model, _ = FastLanguageModel.from_pretrained(
        model_name=args.adapter_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        trust_remote_code=True,
    )

    print(f"Loading tokenizer from: {args.base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=False)

    print("\nMerging LoRA into base (saving as 16-bit Hugging Face format)...")
    model.save_pretrained_merged(abs_output_dir, tokenizer, save_method="merged_16bit")
    print("Base merge complete.")

    # 3. Internal GGUF Conversion (Bypasses Miniconda hijacking)
    # This requires 'convert_hf_to_gguf.py' to be in the same folder.
    convert_script = os.path.abspath("convert_hf_to_gguf.py")

    if not os.path.exists(convert_script):
        print(f"ERROR: {convert_script} not found in current folder.")
        print("Please run the Invoke-WebRequest command to download it first.")
        sys.exit(1)

    print(f"\n--- Converting HF to BF16 GGUF (Internal Process) ---")

    # Temporarily override sys.argv to pass parameters to the internal script
    old_argv = sys.argv
    sys.argv = [
        "convert_hf_to_gguf.py",
        abs_output_dir,
        "--outtype", "bf16",
        "--outfile", abs_bf16_path
    ]

    try:
        # runpy executes the script inside the current venv-llm process
        runpy.run_path(convert_script, run_name="__main__")
    finally:
        sys.argv = old_argv  # Restore original arguments

    print("--- Conversion Done ---")

    # 4. Quantization via llama-quantize
    cmd_quantize = [
        "llama-quantize",
        abs_bf16_path,
        abs_final_path,
        quant_format
    ]
    run_command(cmd_quantize, f"Quantizing to {quant_format}")

    # 5. Cleanup intermediate 7GB file
    if os.path.exists(abs_bf16_path):
        os.remove(abs_bf16_path)
        print(f"\nCleanup: Deleted intermediate file {abs_bf16_path}")

    print(f"\nPipeline Complete! Final GGUF saved to: {abs_final_path}")


if __name__ == "__main__":
    args = get_args()

    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        config_args = {}
        if "global" in config_data:
            config_args.update(config_data["global"])
        if "merge" in config_data:
            config_args.update(config_data["merge"])

        for key, value in config_args.items():
            if hasattr(args, key) and getattr(args, key) is None:
                setattr(args, key, value)

    if not args.adapter_path:
        raise ValueError("adapter_path must be provided via command line.")

    main(args)