"""
8_run_benchmarks.py

Runs benchmarking on the GSM8K, mmlu_ee (MMLU Electrical Engineering),
and ee_qa (Stack Exchange EE QA) datasets.
Compares the base model and fine-tuned model based on exact-match accuracy
and automated LLM-as-a-Judge evaluations. Tracks reasoning trace token counts
to measure CoT distillation and latency. Uses .gguf models exclusively
for an apples-to-apples comparison.
"""

import argparse
import json
import os
import re
import time
import random
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from llama_cpp import Llama
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def get_args():
    parser = argparse.ArgumentParser(description="Run benchmarks to compare base and fine-tuned models")
    parser.add_argument("--config", type=str, default="config/config.json", help="Path to config file")
    parser.add_argument("--base_model_path", type=str, default=None, help="Path to base model (.gguf file or directory)")
    parser.add_argument("--ft_model_path", type=str, default=None, help="Path to merged FT model (.gguf file or directory)")
    parser.add_argument("--system_prompt", type=str, default="You are a helpful AI assistant.", help="System prompt to initialize the model context")
    parser.add_argument("--max_items", type=int, default=100, help="Maximum number of evaluation items")
    parser.add_argument("--max_new_tokens", type=int, default=70000, help="Max tokens to generate per answer")
    parser.add_argument("--run_prefix", type=str, default=None, help="Prefix for output files (e.g., 260312-1803). Generated automatically if empty.")
    parser.add_argument("--benchmark", type=str, default="gsm8k", choices=["gsm8k", "mmlu_ee", "ee_qa"], help="Benchmark to run: gsm8k, mmlu_ee, or ee_qa")
    parser.add_argument("--eval_jsonl", type=str, default=None, help="Path to evaluation JSONL file for ee_qa")
    parser.add_argument("--model", type=str, default=None, help="Judge model name for ee_qa")
    parser.add_argument("--api_base_url", type=str, default=None, help="API Base URL for the Judge model")
    return parser.parse_args()


def find_gguf_path(path):
    if not path:
        return None
    if os.path.isfile(path) and path.lower().endswith('.gguf'):
        return path
    if os.path.isdir(path):
        for file in os.listdir(path):
            if file.lower().endswith('.gguf'):
                return os.path.join(path, file)
        raise FileNotFoundError(f"No .gguf file found in directory: {path}")
    raise FileNotFoundError(f"Path does not exist: {path}")


def clean_number_string(text):
    if not text:
        return None
    text = text.replace(",", "")
    text = text.rstrip(".")
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if match:
        return match.group(0)
    return text.strip()


def extract_predicted_answer(text, benchmark="gsm8k"):
    if not text:
        return None

    if benchmark == "mmlu_ee":
        box_match = re.findall(r"\\boxed\{([A-D])\}", text, re.IGNORECASE)
        if box_match:
            return box_match[-1].upper()

        fallback_match = re.findall(r"\b([A-D])\b", text, re.IGNORECASE)
        if fallback_match:
            return fallback_match[-1].upper()
        return None

    box_match = re.findall(r"\\boxed\{([^}]*)\}", text)
    if box_match:
        return clean_number_string(box_match[-1])

    anchor_match = re.search(r"(?:Final [Aa]nswer|The answer is|yields)[:\s]*([0-9.,-]+)", text)
    if anchor_match:
        return clean_number_string(anchor_match.group(1))

    numbers = re.findall(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?", text)
    if numbers:
        return clean_number_string(numbers[-1])

    return None


def extract_reference_answer(text):
    parts = text.split("#### ")
    if len(parts) > 1:
        ans = parts[1].strip()
        cleaned = clean_number_string(ans)
        return cleaned if cleaned else ans
    return None


def extract_thinking_trace(text):
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def count_tokens(llm, text):
    if not text: return 0
    return len(llm.tokenize(text.encode("utf-8")))


def evaluate_model(llm, dataset, max_items, max_new_tokens, log_file, state_dict, state_key, state_file, system_prompt, benchmark):
    start_index = len(state_dict[state_key])
    items_to_process = min(len(dataset), max_items)

    if start_index >= items_to_process:
        print(f"  -> All {items_to_process} items already evaluated for {state_key}. Skipping inference.")
        return state_dict[state_key]

    print(f"  -> Resuming {state_key} from item {start_index + 1}. Processing {items_to_process - start_index} remaining items sequentially...")

    for i in tqdm(range(start_index, items_to_process), desc="Inference Progress", unit="item", initial=start_index, total=items_to_process):
        item = dataset[i]

        if benchmark == "mmlu_ee":
            q_text = item["question"]
            choices = item["choices"]
            q = f"{q_text}\n\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}"
            ref = ["A", "B", "C", "D"][item["answer"]]
            if system_prompt == "You are a helpful AI assistant.":
                sys_prompt_to_use = "You are an expert electrical engineer. Think step-by-step and place your final chosen letter (A, B, C, or D) inside a \\boxed{} tag."
            else:
                sys_prompt_to_use = system_prompt
        elif benchmark == "ee_qa":
            q = item["question_body"]
            ref = item["enriched_answer"]
            sys_prompt_to_use = system_prompt
        elif benchmark == "gsm8k":
            q = item["question"]
            ref = str(item["answer"])
            sys_prompt_to_use = system_prompt

        prompt = (
            f"system\n{sys_prompt_to_use}<|im_end|>\n"
            f"<|im_start|>user\n{q}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        start_time = time.time()
        response = llm(
            prompt,
            max_tokens=max_new_tokens,
            stop=["<|im_end|>"],
            temperature=0.01,
        )
        end_time = time.time()
        generation_time = end_time - start_time

        out = response["choices"][0]["text"]

        if benchmark == "mmlu_ee":
            ref_ans = ref
        elif benchmark == "ee_qa":
            ref_ans = ref
        elif benchmark == "gsm8k":
            ref_ans = extract_reference_answer(ref)

        thinking_trace = extract_thinking_trace(out)

        if "<think>" in out and "</think>" not in out:
            answer_trace = ""
            pred_ans = None
        else:
            if "</think>" in out:
                answer_trace = out.split("</think>")[-1].strip()
            else:
                answer_trace = out.strip()

            if benchmark == "ee_qa":
                pred_ans = answer_trace
            else:
                pred_ans = extract_predicted_answer(answer_trace, benchmark)

        thinking_tokens = count_tokens(llm, thinking_trace)
        answer_tokens = count_tokens(llm, answer_trace)

        if benchmark == "ee_qa":
            is_correct = None
        else:
            is_correct = (ref_ans == pred_ans)

        result_item = {
            "question": q,
            "reference_answer": ref_ans,
            "predicted_answer": pred_ans,
            "is_correct": is_correct,
            "thinking_tokens": thinking_tokens,
            "answer_tokens": answer_tokens,
            "time_seconds": generation_time,
            "raw_output": out
        }

        state_dict[state_key].append(result_item)
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, indent=4)

        if log_file:
            q_num = i + 1
            header_text = f" Question {q_num:03d} "
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{'#' * 40}\n")
                f.write(f"#####{header_text.center(30)}#####\n")
                f.write(f"{'#' * 40}\n\n")
                f.write(f"--- Question ---\n{q}\n\n")
                f.write(f"--- Reference Answer ---\n{ref_ans}\n\n")
                f.write(f"--- Model Answer (Thinking Omitted) ---\n{answer_trace}\n\n")
                f.write(f"Correct: {is_correct} | Thinking Tokens: {thinking_tokens} | Answer Tokens: {answer_tokens} | Time: {generation_time:.2f}s\n")
                f.write("=" * 80 + "\n\n")

    return state_dict[state_key]


def run_llm_judge(args, state_dict, output_file):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("WARNING: GEMINI_API_KEY not found. Skipping LLM Judge evaluation.")
        return

    client = OpenAI(api_key=api_key, base_url=args.api_base_url)
    base_results = state_dict["base_results"]
    ft_results = state_dict["ft_results"]

    judge_json = output_file.replace(".csv", "_VERDICTS.json")
    judge_results = []
    ft_wins = 0
    base_wins = 0
    ties = 0

    if os.path.exists(judge_json):
        with open(judge_json, "r", encoding="utf-8") as f:
            judge_results = json.load(f)
        for res in judge_results:
            if res.get("ft_won"):
                ft_wins += 1
            elif res.get("base_won"):
                base_wins += 1
            elif res.get("tie"):
                ties += 1
        print(f"Found existing judge state. Resuming from item {len(judge_results) + 1}...")

    print(f"\n{'=' * 80}")
    print(">>> RUNNING LLM-AS-A-JUDGE EVALUATION <<<")
    print(f"{'=' * 80}\n")

    for i, (base_item, ft_item) in enumerate(zip(base_results, ft_results)):
        if i < len(judge_results):
            continue

        q = base_item["question"]
        ref = base_item["reference_answer"]
        base_ans = base_item["predicted_answer"]
        ft_ans = ft_item["predicted_answer"]

        if not base_ans or not ft_ans:
            continue

        is_ft_a = random.choice([True, False])
        ans_a = ft_ans if is_ft_a else base_ans
        ans_b = base_ans if is_ft_a else ft_ans

        prompt = (
            "You are an expert Electrical Engineering professor grading two student answers.\n\n"
            f"Question:\n{q}\n\n"
            f"Ground Truth Reference:\n{ref}\n\n"
            f"Student A:\n{ans_a}\n\n"
            f"Student B:\n{ans_b}\n\n"
            "Evaluate both students strictly on their factual accuracy regarding the core technical question, comparing them directly to the Ground Truth.\n"
            "CRITICAL GRADING RULES:\n"
            "1. First, identify if the core physical or technical conclusion of each student matches the Ground Truth.\n"
            "2. DO NOT award partial credit or a win for tangential advice, safety tips, or long formatting if the core factual answer is incorrect.\n"
            "3. DO NOT penalize concise answers. Verbosity does not equal correctness.\n"
            "4. If BOTH students fail the core technical factual question, you MUST declare a Tie, regardless of which answer is more detailed, polite, or helpful.\n\n"
            "Provide a detailed explanation of your reasoning analyzing their core factual accuracy. Then, on a new line at the very end, output your final verdict strictly as: 'Verdict: A', 'Verdict: B', or 'Verdict: Tie'."
        )

        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=16000,
                temperature=0.0
            )
            full_response = response.choices[0].message.content.strip()

            verdict_match = re.search(r"Verdict:\s*(A|B|Tie)", full_response, re.IGNORECASE)

            if verdict_match:
                verdict_raw = verdict_match.group(1).upper()
                if "TIE" in verdict_raw:
                    winner = "Tie"
                    ties += 1
                elif "A" in verdict_raw:
                    winner = "Fine-Tuned" if is_ft_a else "Base"
                elif "B" in verdict_raw:
                    winner = "Base" if is_ft_a else "Fine-Tuned"
            else:
                winner = "Error"

            if winner == "Fine-Tuned":
                ft_wins += 1
            elif winner == "Base":
                base_wins += 1

            judge_results.append({
                "question": q,
                "model_a": "Fine-Tuned" if is_ft_a else "Base",
                "model_b": "Base" if is_ft_a else "Fine-Tuned",
                "answer_a": ans_a,
                "answer_b": ans_b,
                "ft_won": winner == "Fine-Tuned",
                "base_won": winner == "Base",
                "tie": winner == "Tie",
                "judge_explanation": full_response
            })

            with open(judge_json, "w", encoding="utf-8") as f:
                json.dump(judge_results, f, indent=4)

            print(f"\n--- [Question {i+1}] ---")
            print(f"Winner: {winner}")
            print(f"Judge Reasoning:\n{full_response}")
            print("-" * 40)

            time.sleep(1.0)

        except Exception as e:
            print(f"Error judging item {i}: {e}")

    total_judged = len(judge_results)
    win_rate = (ft_wins / total_judged) * 100 if total_judged else 0
    base_win_rate = (base_wins / total_judged) * 100 if total_judged else 0
    tie_rate = (ties / total_judged) * 100 if total_judged else 0

    print(f"\n--- Final Judge Results ---")
    print(f"FT Win Rate:   {win_rate:.1f}% ({ft_wins})")
    print(f"Base Win Rate: {base_win_rate:.1f}% ({base_wins})")
    print(f"Tie Rate:      {tie_rate:.1f}% ({ties})")

    with open(judge_json, "w", encoding="utf-8") as f:
        json.dump(judge_results, f, indent=4)
    print(f"Saved detailed judge verdicts and explanations to {judge_json}")


def main(args):
    run_prefix = args.run_prefix if args.run_prefix else datetime.now().strftime("%y%m%d-%H%M")
    benchmarks_dir = "benchmarks"
    os.makedirs(benchmarks_dir, exist_ok=True)

    b_name = args.benchmark.upper()
    state_file = os.path.join(benchmarks_dir, f"{run_prefix}_{b_name}_benchmark_state.json")
    log_file = os.path.join(benchmarks_dir, f"{run_prefix}_{b_name}_benchmark_generation_log.txt")
    output_file = os.path.join(benchmarks_dir, f"{run_prefix}_{b_name}_evaluation_results.csv")

    state_dict = {"base_results": [], "ft_results": []}

    print(f"\n{'=' * 80}")
    print(f"--- Benchmarking Session Started (Prefix: {run_prefix} | Benchmark: {b_name}) ---")
    print(f"{'=' * 80}\n")

    # Standard JSON State Loading
    if os.path.exists(state_file):
        with open(state_file, "r", encoding="utf-8") as f:
            state_dict = json.load(f)
        print(f"Found existing state file. Loaded {len(state_dict['base_results'])} Base items and {len(state_dict['ft_results'])} FT items.\n")

    # Initialize log file if it doesn't exist
    if not os.path.exists(log_file):
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"=== BENCHMARK GENERATION LOG ({b_name}) ===\n\n")

    if args.benchmark == "mmlu_ee":
        dataset = load_dataset("cais/mmlu", "electrical_engineering", split="test")
    elif args.benchmark == "ee_qa":
        if not args.eval_jsonl or not os.path.exists(args.eval_jsonl):
            raise ValueError("ERROR: eval_jsonl path is invalid or missing. Required for ee_qa benchmark.")
        dataset = []
        with open(args.eval_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                dataset.append(json.loads(line))
    else:
        dataset = load_dataset("gsm8k", "main", split="test")

    # ==========================================
    # BASE MODEL EVALUATION
    # ==========================================
    if len(state_dict["base_results"]) < args.max_items:
        if not args.base_model_path:
            raise ValueError("ERROR: 'base_model_path' is required to run the Base Model.")

        actual_base_path = find_gguf_path(args.base_model_path)
        print(f"Loading Base Model via llama.cpp from {actual_base_path}...")
        base_llm = Llama(
            model_path=actual_base_path,
            n_gpu_layers=-1,
            n_ctx=70000,
            verbose=False
        )

        print("Evaluating Base Model...")
        if len(state_dict["base_results"]) == 0:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f">>> EVALUATING BASE MODEL <<<\n{'=' * 80}\n\n")

        evaluate_model(base_llm, dataset, args.max_items, args.max_new_tokens, log_file, state_dict, "base_results", state_file, args.system_prompt, args.benchmark)

        del base_llm
    else:
        print("Base Model already fully evaluated. Skipping model load.")

    # ==========================================
    # FINE-TUNED MODEL EVALUATION
    # ==========================================
    if len(state_dict["ft_results"]) < args.max_items:
        ft_target_path = args.ft_model_path or getattr(args, "model_path", None)
        if not ft_target_path:
            raise ValueError("ERROR: 'ft_model_path' or 'model_path' (in config) is required to run the Fine-Tuned Model.")

        actual_ft_path = find_gguf_path(ft_target_path)
        print(f"\nLoading MERGED Fine-Tuned Model via llama.cpp directly from {actual_ft_path}...")
        ft_llm = Llama(
            model_path=actual_ft_path,
            n_gpu_layers=-1,
            n_ctx=70000,
            verbose=False
        )

        print("Evaluating Fine-Tuned Model...")
        if len(state_dict["ft_results"]) == 0:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f">>> EVALUATING FINE-TUNED MODEL <<<\n{'=' * 80}\n\n")

        evaluate_model(ft_llm, dataset, args.max_items, args.max_new_tokens, log_file, state_dict, "ft_results", state_file, args.system_prompt, args.benchmark)

        del ft_llm
    else:
        print("Fine-Tuned Model already fully evaluated. Skipping model load.")

    # ==========================================
    # LLM JUDGE EVALUATION (For ee_qa benchmark)
    # ==========================================
    if args.benchmark == "ee_qa" and len(state_dict["base_results"]) >= min(len(dataset), args.max_items) and len(state_dict["ft_results"]) >= min(len(dataset), args.max_items):
        run_llm_judge(args, state_dict, output_file)

    # ==========================================
    # FINAL REPORTING (Reads strictly from JSON state)
    # ==========================================
    base_df = pd.DataFrame(state_dict["base_results"])
    ft_df = pd.DataFrame(state_dict["ft_results"])

    base_accuracy = base_df["is_correct"].dropna().mean() * 100 if not base_df.empty and not base_df["is_correct"].isna().all() else 0
    base_avg_tokens = base_df["thinking_tokens"].mean() if not base_df.empty else 0
    base_avg_ans_tokens = base_df["answer_tokens"].mean() if not base_df.empty else 0
    base_avg_time = base_df["time_seconds"].mean() if not base_df.empty else 0

    ft_accuracy = ft_df["is_correct"].dropna().mean() * 100 if not ft_df.empty and not ft_df["is_correct"].isna().all() else 0
    ft_avg_tokens = ft_df["thinking_tokens"].mean() if not ft_df.empty else 0
    ft_avg_ans_tokens = ft_df["answer_tokens"].mean() if not ft_df.empty else 0
    ft_avg_time = ft_df["time_seconds"].mean() if not ft_df.empty else 0

    print("\n\n" + "=" * 98)
    print(f"FINAL BENCHMARK SUMMARY ({args.benchmark.upper()})")
    print("=" * 98)
    print(f"| {'Model Type':<15} | {'Accuracy (%)':<15} | {'Avg Think Tokens':<18} | {'Avg Ans Tokens':<15} | {'Avg Time (s)':<15} |")
    print("-" * 98)
    print(f"| {'Base Model':<15} | {base_accuracy:<15.2f} | {base_avg_tokens:<18.1f} | {base_avg_ans_tokens:<15.1f} | {base_avg_time:<15.2f} |")
    print(f"| {'Fine-Tuned':<15} | {ft_accuracy:<15.2f} | {ft_avg_tokens:<18.1f} | {ft_avg_ans_tokens:<15.1f} | {ft_avg_time:<15.2f} |")
    print("=" * 98)

    if not base_df.empty and not ft_df.empty:
        combined_df = pd.DataFrame({
            "question": base_df["question"],
            "base_correct": base_df["is_correct"],
            "base_think_tokens": base_df["thinking_tokens"],
            "base_ans_tokens": base_df["answer_tokens"],
            "base_time": base_df["time_seconds"],
            "ft_correct": ft_df["is_correct"],
            "ft_think_tokens": ft_df["thinking_tokens"],
            "ft_ans_tokens": ft_df["answer_tokens"],
            "ft_time": ft_df["time_seconds"]
        })
        combined_df.to_csv(output_file, index=False)
        print(f"Detailed results saved to {output_file}")


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
            elif not hasattr(args, key):
                setattr(args, key, value)

    main(args)