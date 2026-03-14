"""
3_trace_generation.py

Generates synthetic, non-linear Chain-of-Thought reasoning traces
for enriched Q&A pairs using a frontier teacher model.
"""

import argparse
import json
import os
import time
import numpy as np
from openai import OpenAI
from transformers import AutoTokenizer
from tabulate import tabulate
from dotenv import load_dotenv

load_dotenv()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.json", help="Path to config file")
    parser.add_argument("--input_jsonl", type=str, default=None, help="Path to the input JSONL file containing enriched pairs")
    parser.add_argument("--output_jsonl", type=str, default=None, help="Path to save the pairs with reasoning traces")
    parser.add_argument("--base_model_path", type=str, default=None, help="Path to the base model for the tokenizer")
    parser.add_argument("--api_base_url", type=str, default=None, help="Base URL for the LLM API")
    parser.add_argument("--model", type=str, default=None, help="The LLM model name to use for trace generation")
    parser.add_argument("--max_tokens", type=int, default=None, help="Maximum tokens to generate for the trace")
    parser.add_argument("--max_items", type=int, default=None, help="Maximum number of items to process in this run (leave blank to process all)")
    parser.add_argument("--delay", type=float, default=None, help="Delay in seconds between API requests")
    return parser.parse_args()


def load_local_tokenizer(tokenizer_path):
    print(f"Loading tokenizer from: {tokenizer_path}...")
    try:
        return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load tokenizer: {e}. Individual token counts will be unavailable.")
        return None


def get_processed_ids(path):
    processed = set()
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed.add(data['question_id'])
                except:
                    continue
    return processed


def build_re_prompt(item):
    return f"""System Role:
You are a Senior Electrical Engineering Specialist. Your task is to perform a rigorous, internal "Whiteboard Stress-Test" of the following circuit problem before verifying the Target Answer.

Task:
Generate a high-density internal monologue (scratchpad) that reconstructs the engineering logic. 

Strict Constraints:
1. Invisible Integration: Do NOT announce your steps. Never use phrases like "Let's perform a sensitivity analysis", "Hypothesis 1:", "Slam:", or "Numerical validation:". Weave these elements naturally into a seamless stream of consciousness.
2. Raw Text Only: No tags, no headers, no bullet points.
3. No Meta-Review: Do not review or critique the Target Answer at the end. Your output is the *preceding* thought process that leads to the answer, not a grading rubric of it.
4. Internal Friction & Doubt: The trace must NOT be linear. Question the user's assumptions and explore potential pitfalls.
5. Sensitivity Analysis: Perform a "what-if" on at least two critical variables.
6. Numerical Validation: Manually derive the primary numbers found in the target answer using first-principles physics.
7. Negative Hypothesis Testing: Explicitly consider and dismiss at least two incorrect theories using quantitative arguments, but do so naturally as part of the troubleshooting process.
8. Conclusion: End the monologue naturally as soon as the logical bridge to the target answer is fully established. Do not add filler text.

QUESTION:
{item['question_body']}

TARGET ANSWER:
{item['enriched_answer']}"""


def run_re_pipeline(args, client, tokenizer):
    processed_ids = get_processed_ids(args.output_jsonl)
    print(f"Skipping {len(processed_ids)} already processed items.")

    success_count = 0

    with open(args.input_jsonl, 'r', encoding='utf-8') as infile, \
            open(args.output_jsonl, 'a', encoding='utf-8') as outfile:

        for line in infile:
            if args.max_items is not None and success_count >= args.max_items:
                break

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            qid = item['question_id']
            if qid in processed_ids:
                continue

            print(f"Processing QID: {qid}...")
            prompt = build_re_prompt(item)

            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=args.max_tokens
                )

                trace = response.choices[0].message.content.strip()
                item['thinking_trace_re'] = trace
                item['trace_truncated'] = (response.choices[0].finish_reason == "length")

                token_count_str = "N/A"
                if tokenizer:
                    token_count = len(tokenizer.encode(trace))
                    token_count_str = f"{token_count} tokens"

                outfile.write(json.dumps(item) + '\n')
                outfile.flush()

                success_count += 1
                total_to_do = args.max_items if args.max_items else "End of File"
                truncated_status = "!! TRUNCATED !!" if item['trace_truncated'] else "OK"

                print(f"[{success_count}/{total_to_do}] {truncated_status} — qid: {qid} | Trace: {token_count_str}")
                time.sleep(args.delay)

            except Exception as e:
                err = str(e).lower()
                if any(x in err for x in ['quota', '429', 'insufficient']):
                    print(f"Quota exhausted. Stopping. Error: {e}")
                    break
                else:
                    print(f"Failed QID {qid}: {e}")
                    time.sleep(2)

    print(f"\nBatch finished. Total items added: {success_count}")


def run_token_audit(output_path, tokenizer):
    if not os.path.exists(output_path) or not tokenizer:
        return

    token_counts = []
    truncated_ids = []

    print("\nAuditing full file for final stats...")
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                trace = item.get('thinking_trace_re', "")
                if trace:
                    token_counts.append(len(tokenizer.encode(trace)))
                    if item.get('trace_truncated', False):
                        truncated_ids.append(item['question_id'])
            except:
                continue

    if not token_counts:
        return

    stats = [
        ["Metric", "Value"],
        ["Total Traces", len(token_counts)],
        ["Truncated Count", len(truncated_ids)],
        ["Avg Tokens", f"{np.mean(token_counts):.2f}"],
        ["Median Tokens", int(np.median(token_counts))],
        ["Min Tokens", np.min(token_counts)],
        ["Max Tokens", np.max(token_counts)],
        ["Total Tokens", f"{np.sum(token_counts):,}"]
    ]

    print("\n" + "=" * 45)
    print("       NANBEIGE RE-TRACE TOKEN AUDIT")
    print("=" * 45)
    print(tabulate(stats, headers="firstrow", tablefmt="github"))
    print("=" * 45)

    if truncated_ids:
        print(f"\nWarning: The following QIDs were truncated due to token limits: {truncated_ids}")


def main(args):
    api_string = 'DASHSCOPE_API_KEY'
    api_key = os.getenv(api_string)

    if not api_key:
        raise ValueError(f"API Key not found. Ensure {api_string} is set in your .env file.")

    client = OpenAI(
        api_key=api_key,
        base_url=args.api_base_url,
    )
    tokenizer = load_local_tokenizer(args.base_model_path)

    run_re_pipeline(args, client, tokenizer)
    run_token_audit(args.output_jsonl, tokenizer)


if __name__ == "__main__":
    args = get_args()

    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        config_args = {}
        if "global" in config_data:
            config_args.update(config_data["global"])
        if "trace_generation" in config_data:
            config_args.update(config_data["trace_generation"])

        for key, value in config_args.items():
            if hasattr(args, key) and getattr(args, key) is None:
                setattr(args, key, value)

    main(args)