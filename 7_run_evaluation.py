"""
7_run_evaluation.py

Performs an evaluation of a local LLM (via llama-cpp-python) using an
extracted subset of held-out Q&A examples. Outputs inference results
and a token count summary table to a formatted Markdown file.
"""

import argparse
import json
import os
from llama_cpp import Llama


def get_args():
    parser = argparse.ArgumentParser(description="Run evaluation on held-out dataset using llama.cpp")
    parser.add_argument("--config", type=str, default="config/config.json", help="Path to config file")
    parser.add_argument("--eval_jsonl", type=str, default=None, help="Path to the JSONL file containing evaluation examples")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the quantized GGUF model")
    parser.add_argument("--output_md", type=str, default=None, help="Path to save the Markdown results")
    parser.add_argument("--system_prompt", type=str, default=None, help="System prompt to initialize the model context")
    return parser.parse_args()


def main(args):
    table_data = []

    print(f"Loading model from {args.model_path}...")
    llm = Llama(
        model_path=args.model_path,
        n_gpu_layers=-1,
        n_ctx=30000,
        verbose=False
    )

    def count_tokens(text):
        if not text: return 0
        return len(llm.tokenize(text.encode("utf-8")))

    # --- Initialize Markdown File ---
    with open(args.output_md, "w", encoding="utf-8") as out_f:
        out_f.write("# Evaluation Results\n\n")

    # --- Process Evaluation File ---
    print(f"Starting evaluation on {args.eval_jsonl}...\n")
    with open(args.eval_jsonl, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue

            record = json.loads(line)
            q_id = record.get("question_id", "UNKNOWN_ID")
            title = record.get("title", "No Title")
            body = record.get("question_body", "")
            orig_thinking = record.get("thinking_trace_re", "")
            anchored_thinking = record.get("final_anchored_thinking", "")
            orig_answer = record.get("original_answer", "")
            enriched_answer = record.get("enriched_answer", "")

            prompt = (
                f"system\n{args.system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{title}\n\n{body}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

            response = llm(
                prompt,
                max_tokens=30000,
                stop=["<|im_end|>"],
                temperature=0.6,
            )

            generated_text = response["choices"][0]["text"]

            if "</think>" in generated_text:
                parts = generated_text.split("</think>")
                pred_thinking = parts[0].replace("<think>", "").strip()
                pred_answer = parts[1].strip()
            else:
                pred_thinking = "FAILED TO CLOSE THINKING BLOCK"
                pred_answer = generated_text.strip()

            # Thinking comparison (Original/Generated vs Predicted)
            orig_th_count = count_tokens(orig_thinking)
            pred_th_count = count_tokens(pred_thinking)

            # Answer comparison (Enriched vs Predicted)
            enr_ans_count = count_tokens(enriched_answer)
            pred_ans_count = count_tokens(pred_answer)

            table_data.append(
                f"| {idx + 1} | {q_id} | {orig_th_count} | {pred_th_count} | {enr_ans_count} | {pred_ans_count} |"
            )

            # --- Console Printout ---
            print(f"\n{'=' * 80}")
            print(f"EXAMPLE {idx + 1} | ID: {q_id}")
            print(f"QUESTION: {title}")
            print(f"{'-' * 80}")
            print(f"TOKEN COUNTS:")
            print(f"  Thinking: Orig={orig_th_count} | Pred={pred_th_count}")
            print(f"  Answer:   Enriched={enr_ans_count} | Pred={pred_ans_count}")
            print(f"{'=' * 80}")

            print(f"\n[ ENRICHED ANSWER ]\n{'-' * 20}\n{enriched_answer}\n")
            print(f"\n[ PREDICTED THINKING ]\n{'-' * 20}\n{pred_thinking}\n")
            print(f"\n[ PREDICTED ANSWER ]\n{'-' * 20}\n{pred_answer}\n")
            print(f"{'=' * 80}\n")

            # --- Append to Markdown ---
            with open(args.output_md, "a", encoding="utf-8") as out_f:
                out_f.write(f"## [{idx + 1}] ID: {q_id} | {title}\n\n")
                out_f.write("### Enriched Answer\n")
                out_f.write(f"{enriched_answer}\n\n")
                out_f.write("### Predicted Output\n")
                out_f.write(f"{generated_text}\n\n")
                out_f.write("---\n\n")

    # --- Append Token Table to Markdown ---
    with open(args.output_md, "a", encoding="utf-8") as out_f:
        out_f.write("# Token Count Comparisons\n\n")
        out_f.write("| Example | Orig Think | Pred Think | Enr Ans | Pred Ans |\n")
        out_f.write("|---|---|---|---|---|\n")
        for row in table_data:
            out_f.write(row + "\n")

    # --- Print Unified Table to Console ---
    print("\n\n" + "=" * 65)
    print("FINAL EVALUATION TOKEN COUNT SUMMARY")
    print("=" * 65)
    print(f"| {'Ex':<3} | {'Orig Th':<8} | {'Pred Th':<8} | {'Enr Ans':<8} | {'Pred Ans':<8} |")
    print("-" * 65)

    for idx, row in enumerate(table_data):
        cols = [c.strip() for c in row.split('|') if c.strip()]
        if len(cols) == 6:
            print(f"| {cols[0]:<3} | {cols[2]:<8} | {cols[3]:<8} | {cols[4]:<8} | {cols[5]:<8} |")

    print("=" * 65)
    print(f"\nEvaluation complete. Markdown saved to {args.output_md}")


if __name__ == "__main__":
    args = get_args()

    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        config_args = {}
        if "global" in config_data:
            config_args.update(config_data["global"])
        if "evaluation" in config_data:
            config_args.update(config_data["evaluation"])

        for key, value in config_args.items():
            if hasattr(args, key) and getattr(args, key) is None:
                setattr(args, key, value)

    if not args.eval_jsonl or not args.model_path:
        raise ValueError("Both eval_jsonl and model_path must be provided via command line or config file.")

    main(args)