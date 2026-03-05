"""
2_data_enrichment.py

Enriches curated Q&A pairs using a frontier LLM API. It selectively loads
target posts from XML dumps to minimize memory overhead and utilizes a
tracking mechanism for safe, restartable API execution.
"""

import argparse
import json
import os
import time
import re
import html
import pandas as pd
import xml.etree.ElementTree as ET
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.json", help="Path to config file")
    parser.add_argument("--input_csv", type=str, default=None, help="Path to the input CSV file containing curated pairs")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to the folder containing Posts.xml and Comments.xml")
    parser.add_argument("--output_jsonl", type=str, default=None, help="Path to save the enriched pairs JSONL")
    parser.add_argument("--processed_ids_file", type=str, default=None, help="Path to track processed question IDs")
    parser.add_argument("--model", type=str, default=None, help="The LLM model name to use for enrichment")
    parser.add_argument("--api_base_url", type=str, default=None, help="Base URL for the LLM API")
    parser.add_argument("--delay", type=float, default=None, help="Delay in seconds between API requests")
    parser.add_argument("--max_items", type=int, default=None, help="Maximum number of items to process in this run")
    return parser.parse_args()


def parse_int(s):
    if s is None:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def load_text_data(posts_path, comments_path, target_qids, target_aids):
    questions = {}
    answers = {}
    comments = []

    for event, elem in ET.iterparse(posts_path, events=('end',)):
        if elem.tag != 'row':
            continue

        post_id = parse_int(elem.attrib.get('Id'))
        post_type = elem.attrib.get('PostTypeId')

        if post_type == '1' and post_id in target_qids:
            questions[post_id] = {
                'Title': elem.attrib.get('Title'),
                'Body': elem.attrib.get('Body'),
                'Tags': elem.attrib.get('Tags')
            }
        elif post_type == '2' and post_id in target_aids:
            answers[post_id] = {
                'Body': elem.attrib.get('Body')
            }
        elem.clear()

    for event, elem in ET.iterparse(comments_path, events=('end',)):
        if elem.tag != 'row':
            continue

        post_id = parse_int(elem.attrib.get('PostId'))
        if post_id in target_qids:
            comments.append({
                'PostId': post_id,
                'Score': parse_int(elem.attrib.get('Score')),
                'Text': elem.attrib.get('Text'),
                'CreationDate': elem.attrib.get('CreationDate')
            })
        elem.clear()

    return questions, answers, pd.DataFrame(comments)


def clean_html(body):
    if body is None:
        return ''
    text = html.unescape(body)
    text = re.sub(r'<pre><code>(.*?)</code></pre>',
                  lambda m: '\n[CODE]\n' + m.group(1).strip() + '\n[/CODE]\n',
                  text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text


def build_prompt(title, body, comments, answer):
    comment_block = ""
    if comments:
        for i, (score, text) in enumerate(comments, 1):
            comment_block += f"[comment_{i}] (score={score}): {text}\n"
    else:
        comment_block = "None"

    return f"""You are an expert electrical engineer and digital hardware specialist with deep knowledge across circuit design, FPGAs, microcontrollers, Verilog, VHDL, signal processing, and power electronics.

You will be given a technical question from an electrical engineering Q&A forum, a verified correct answer to that question, and optionally some community comments. Your task is to produce an enriched version of the correct answer.

The enriched answer must:
1. Preserve the technical correctness of the provided answer — do not contradict or deviate from it
2. Expand on the underlying principles and theory that explain why the answer is correct
3. Add relevant technical context, important caveats, or practical considerations that deepen understanding
4. State important distinctions factually and concisely where relevant (for example: "note that this applies to FPGAs but not ASICs")
5. Be written as a direct, confident technical answer — not as a reasoning process, tutorial, or exploration of alternatives
6. Use clear prose without excessive formatting, bullet overload, tables, or emoji
7. If the question involves code, include a corrected or improved code example that reflects the best practices described in the answer

---

QUESTION TITLE: {title}

QUESTION BODY:
{body}

COMMUNITY COMMENTS:
{comment_block}

CORRECT ANSWER:
{answer}

---

Produce the enriched answer now."""


def get_unprocessed(df, processed_path):
    if not os.path.exists(processed_path):
        return df
    with open(processed_path, 'r') as f:
        processed = set(int(x.strip()) for x in f.read().splitlines() if x.strip())
    unprocessed = df[~df['ParentId'].isin(processed)]
    print(f"Already processed: {len(processed)}")
    print(f"Remaining: {len(unprocessed)}")
    return unprocessed


def save_processed_id(qid, processed_path):
    with open(processed_path, 'a') as f:
        f.write(f"{qid}\n")


def enrich_pair(row, questions_dict, answers_dict, comments_df, client, model):
    qid = int(row['ParentId'])
    aid = int(row['Id'])

    question = questions_dict.get(qid, {})
    answer = answers_dict.get(aid, {})

    q_comments = comments_df[comments_df['PostId'] == qid].sort_values(
        'CreationDate') if not comments_df.empty else pd.DataFrame()
    comments_list = [(c.Score, c.Text) for c in q_comments.itertuples()]

    title = clean_html(question.get('Title', ''))
    body = clean_html(question.get('Body', ''))
    answer_text = clean_html(answer.get('Body', ''))

    prompt = build_prompt(title, body, comments_list, answer_text)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        extra_body={"enable_thinking": False}
    )

    choice = response.choices[0]
    if choice.finish_reason == "length":
        print(f"Truncated: qid={qid}. Data discarded.")
        return None

    enriched_text = choice.message.content

    return {
        'question_id': qid,
        'answer_id': aid,
        'title': title,
        'question_body': body,
        'original_answer': answer_text,
        'enriched_answer': enriched_text,
        'tags': question.get('Tags', ''),
    }


def main(args):
    api_string = 'DASHSCOPE_API_KEY'
    api_key = os.getenv(api_string)

    if not api_key:
        raise ValueError(f"API Key not found. Ensure {api_string} is set in your .env file.")

    posts_path = os.path.join(args.data_dir, "Posts.xml")
    comments_path = os.path.join(args.data_dir, "Comments.xml")

    if not os.path.exists(posts_path) or not os.path.exists(comments_path):
        print(f"Error: Could not find Posts.xml or Comments.xml in {args.data_dir}")
        return

    client = OpenAI(
        api_key=api_key,
        base_url=args.api_base_url,
    )

    pairs_df = pd.read_csv(args.input_csv)
    unprocessed_df = get_unprocessed(pairs_df, args.processed_ids_file)

    if unprocessed_df.empty:
        print("All pairs in this CSV have been processed.")
        return

    target_qids = set(unprocessed_df['ParentId'])
    target_aids = set(unprocessed_df['Id'])

    print("Loading text data from XML...")
    questions_dict, answers_dict, comments_df = load_text_data(
        posts_path,
        comments_path,
        target_qids,
        target_aids
    )

    success = 0
    failed = 0
    quota_exceeded = False

    print("Starting API enrichment loop...")
    with open(args.output_jsonl, 'a') as outfile:
        for i, (_, row) in enumerate(unprocessed_df.iterrows()):
            if args.max_items and success >= args.max_items:
                break
            qid = int(row['ParentId'])
            try:
                result = enrich_pair(row, questions_dict, answers_dict, comments_df, client, args.model)
                if result:
                    outfile.write(json.dumps(result) + '\n')
                    outfile.flush()

                save_processed_id(qid, args.processed_ids_file)
                success += 1
                total_to_do = args.max_items if args.max_items else len(unprocessed_df)
                print(f"[{success}/{total_to_do}] OK — question_id: {qid}")
                time.sleep(args.delay)

            except Exception as e:
                err_str = str(e).lower()
                if 'quota' in err_str or 'insufficient' in err_str or '429' in err_str:
                    print(f"\nQuota exhausted after {success} successful requests. Stopping.")
                    quota_exceeded = True
                    break
                else:
                    failed += 1
                    print(f"[{i + 1}/{len(unprocessed_df)}] FAILED — question_id: {qid} | error: {e}")
                    time.sleep(2)

    print(f"\nDone. Success: {success} | Failed: {failed} | Quota exceeded: {quota_exceeded}")


if __name__ == "__main__":
    args = get_args()

    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        config_args = {}
        if "global" in config_data:
            config_args.update(config_data["global"])
        if "enrichment" in config_data:
            config_args.update(config_data["enrichment"])

        for key, value in config_args.items():
            if hasattr(args, key) and getattr(args, key) is None:
                setattr(args, key, value)

    if not args.data_dir:
        raise ValueError("data_dir must be provided either via command line or config file.")

    main(args)