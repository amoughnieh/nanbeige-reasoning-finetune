"""
1_data_curation.py

Parses Stack Exchange XML data dumps, applies quality and length filters,
and generates a clean dataset of eligible Q&A pairs for LLM fine-tuning.
"""

import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime
import re
import os
import argparse
import json

# --- Configuration ---
def get_args():
    parser = argparse.ArgumentParser(description="Parse and filter Stack Exchange XML dumps.")
    parser.add_argument("--config", type=str, default="config/config.json", help="Path to config file")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to the extracted Stack Exchange XML files.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the CSV of curated pairs.")
    parser.add_argument("--min_upvotes", type=int, default=None, help="Minimum upvotes for a question/answer pair.")
    parser.add_argument("--max_downvote_ratio", type=float, default=None, help="Maximum allowable ratio of downvotes to upvotes.")
    parser.add_argument("--min_length", type=int, default=None, help="Minimum character length for bodies.")
    return parser.parse_args()


# --- Parsing Helpers ---
def parse_datetime(s):
    return datetime.fromisoformat(s) if s else None


def parse_int(s):
    if s is None: return None
    try:
        return int(s)
    except ValueError:
        return None


def parse_posts(filepath):
    questions, answers = [], []
    for event, elem in ET.iterparse(filepath, events=('end',)):
        if elem.tag != 'row': continue
        post_type = elem.attrib.get('PostTypeId')

        if post_type == '1':
            questions.append({
                'Id': parse_int(elem.attrib.get('Id')),
                'Title': elem.attrib.get('Title'),
                'Body': elem.attrib.get('Body'),
                'Tags': elem.attrib.get('Tags'),
                'Score': parse_int(elem.attrib.get('Score')),
            })
        elif post_type == '2':
            answers.append({
                'Id': parse_int(elem.attrib.get('Id')),
                'ParentId': parse_int(elem.attrib.get('ParentId')),
                'Body': elem.attrib.get('Body'),
                'Score': parse_int(elem.attrib.get('Score')),
            })
        elem.clear()
    return pd.DataFrame(questions), pd.DataFrame(answers)


def parse_votes(filepath):
    votes = []
    for event, elem in ET.iterparse(filepath, events=('end',)):
        if elem.tag != 'row': continue
        vote_type = elem.attrib.get('VoteTypeId')
        if vote_type in ('2', '3'):
            votes.append({
                'PostId': parse_int(elem.attrib.get('PostId')),
                'VoteTypeId': parse_int(vote_type),
            })
        elem.clear()
    return pd.DataFrame(votes)


def clean_html_length(body):
    """Strips HTML tags and measures actual text length."""
    clean = re.sub(r'<[^>]+>', '', body or '')
    return len(clean.strip())


# --- Main Pipeline ---
def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    print("1. Parsing XML dumps...")
    questions_df, answers_df = parse_posts(os.path.join(args.data_dir, 'Posts.xml'))
    votes_df = parse_votes(os.path.join(args.data_dir, 'Votes.xml'))

    print("2. Calculating vote thresholds for questions...")
    upvotes_q = votes_df[votes_df['VoteTypeId'] == 2].groupby('PostId').size().reset_index(name='Upvotes')
    downvotes_q = votes_df[votes_df['VoteTypeId'] == 3].groupby('PostId').size().reset_index(name='Downvotes')

    q = questions_df[['Id']].copy()
    q = q.merge(upvotes_q, left_on='Id', right_on='PostId', how='left').drop(columns='PostId')
    q = q.merge(downvotes_q, left_on='Id', right_on='PostId', how='left').drop(columns='PostId')
    q[['Upvotes', 'Downvotes']] = q[['Upvotes', 'Downvotes']].fillna(0).astype(int)

    # Filter 1: Questions must have >= 5 upvotes and downvotes <= 30% of upvotes
    filtered_questions = q[
        (q['Upvotes'] >= args.min_upvotes) & (q['Downvotes'] <= args.max_downvote_ratio * q['Upvotes'])]
    filtered_question_ids = set(filtered_questions['Id'])

    print("3. Filtering answers and matching to questions...")
    filtered_answers = answers_df[answers_df['ParentId'].isin(filtered_question_ids)].copy()

    upvotes_a = votes_df[votes_df['VoteTypeId'] == 2].groupby('PostId').size().reset_index(name='Upvotes')
    downvotes_a = votes_df[votes_df['VoteTypeId'] == 3].groupby('PostId').size().reset_index(name='Downvotes')

    filtered_answers = filtered_answers.merge(upvotes_a, left_on='Id', right_on='PostId', how='left').drop(
        columns=['PostId'])
    filtered_answers = filtered_answers.merge(downvotes_a, left_on='Id', right_on='PostId', how='left').drop(
        columns=['PostId'])
    filtered_answers[['Upvotes', 'Downvotes']] = filtered_answers[['Upvotes', 'Downvotes']].fillna(0).astype(int)

    # Filter 2: Get the top answer per question, apply the same vote ratio filter
    top_answers = filtered_answers.sort_values('Upvotes', ascending=False).groupby('ParentId').first().reset_index()
    clean_pairs = top_answers[
        (top_answers['Upvotes'] >= args.min_upvotes) & (top_answers['Downvotes'] <= args.max_downvote_ratio * top_answers['Upvotes'])].copy()

    print("4. Applying Image and Length Filters...")
    questions_with_images = set(questions_df[questions_df['Body'].str.contains('<img', na=False)]['Id'])
    answers_with_images = set(answers_df[answers_df['Body'].str.contains('<img', na=False)]['Id'])

    final_pairs = clean_pairs[~clean_pairs['ParentId'].isin(questions_with_images)]
    final_pairs = final_pairs[~final_pairs['Id'].isin(answers_with_images)].copy()

    final_pairs['QuestionLength'] = final_pairs['ParentId'].map(
        questions_df.set_index('Id')['Body'].apply(clean_html_length))
    final_pairs['AnswerLength'] = final_pairs['Id'].map(answers_df.set_index('Id')['Body'].apply(clean_html_length))

    # Filter 3: Minimum text length of 300 characters
    final_pairs_filtered = final_pairs[
        (final_pairs['QuestionLength'] >= args.min_length) & (final_pairs['AnswerLength'] >= args.min_length)].copy()

    print("5. Saving to disk...")
    output_file = os.path.join(args.output_dir, 'curated_pairs.csv')
    final_pairs_filtered.to_csv(output_file, index=False)

    print(f"Pipeline Complete. \nTotal Eligible Pairs Saved: {len(final_pairs_filtered)}")


if __name__ == "__main__":
    args = get_args()

    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        config_args = {}
        if "global" in config_data:
            config_args.update(config_data["global"])
        if "curation" in config_data:
            config_args.update(config_data["curation"])

        for key, value in config_args.items():
            if hasattr(args, key) and getattr(args, key) is None:
                setattr(args, key, value)

    if not args.data_dir:
        raise ValueError("data_dir must be provided either via command line or config file.")

    main(args)