import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

import anthropic
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel
from tqdm import tqdm

from collect import load_done_ids, now

MODEL = "claude-haiku-5-5"
MAX_TOKENS = 16000
WORKERS = 4
MAX_RETRIES = 3
ANSWERS_FILE = "answers.jsonl"
OUTPUT_FILE = "sentiments.jsonl"
ERRORS_FILE = "sentiment_errors.log"

PROMPT = """Below is an AI assistant's answer to the question: "{query}" (product category: {category}).

<answer>
{answer}
</answer>

List every brand the answer mentions, and give the answer's sentiment toward that brand only:
- positive: the answer recommends or praises it, even if it notes minor drawbacks
- negative: the answer criticizes it or advises against it
- neutral: the answer only names it, or its praise and criticism are balanced

How to name brands:
- If a brand is one of these tracked brands, or a sub-brand or product line of one, use exactly the tracked name (before the brackets): {tracked}
- Otherwise use the brand name as the answer writes it, in Latin script.
- List each brand once.
- Only include brands that make or sell products in this category. Skip stores and websites mentioned only as places to buy or read reviews (such as Amazon, Flipkart, GSMArena) and certification bodies."""


class BrandSentiment(BaseModel):
    brand: str
    sentiment: Literal["positive", "neutral", "negative"]


class Extraction(BaseModel):
    brands: list[BrandSentiment]


def tracked_list(category_brands):
    # "Xiaomi (Redmi, Mi, POCO)" jaisa - sirf Latin aliases jo brand naam se alag hain
    items = []
    for b in category_brands.itertuples():
        extra = [a for a in b.aliases.split("|") if a != b.brand and a.isascii()]
        items.append(f"{b.brand} ({', '.join(extra)})" if extra else b.brand)
    return "; ".join(items)


def extract(client, query_id, query, category, tracked, answer):
    response = client.messages.parse(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": PROMPT.format(
            query=query, category=category, tracked=tracked, answer=answer)}],
        output_format=Extraction,
    )
    if response.parsed_output is None:
        raise ValueError(f"JSON nahi mila, stop_reason={response.stop_reason}")
    return {
        "query_id": query_id,
        "model": MODEL,
        "brands": [b.model_dump() for b in response.parsed_output.brands],
        "stop_reason": response.stop_reason,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "timestamp": now(),
    }


def main():
    parser = argparse.ArgumentParser(description="answers.jsonl ke har answer se brands aur unka sentiment nikalo")
    parser.add_argument("--limit", type=int, help="sirf pehle N answers chalao")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit 1 ya usse bada hona chahiye")

    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY .env mein set nahi hai.")

    queries = pd.read_csv("queries.csv")
    brands = pd.read_csv("brands.csv")
    with open(ANSWERS_FILE, encoding="utf-8") as f:
        answers = pd.DataFrame([json.loads(line) for line in f if line.strip()])
    df = answers.drop_duplicates("query_id", keep="last").merge(queries, on="query_id").sort_values("query_id")
    if args.limit is not None:
        df = df.head(args.limit)

    done = load_done_ids(OUTPUT_FILE)
    pending = df[~df["query_id"].isin(done)]
    print(f"{len(df)} answers: {len(df) - len(pending)} pehle se done, {len(pending)} baaki")
    if pending.empty:
        return

    tracked = {cat: tracked_list(g) for cat, g in brands.groupby("category_id")}
    client = anthropic.Anthropic(max_retries=MAX_RETRIES)
    saved = failed = 0

    # Output aur errors sirf main thread likhta hai, isliye file pe lock ki zaroorat nahi
    with ThreadPoolExecutor(max_workers=WORKERS) as pool, \
            open(OUTPUT_FILE, "a", encoding="utf-8") as out_f, \
            open(ERRORS_FILE, "a", encoding="utf-8") as errors_f, \
            tqdm(total=len(pending), desc="Sentiment") as bar:
        futures = {
            pool.submit(extract, client, r.query_id, r.query_text, r.category,
                        tracked[r.category_id], r.answer): r.query_id
            for r in pending.itertuples()
        }
        for future in as_completed(futures):
            query_id = futures[future]
            try:
                record = future.result()
            except anthropic.AuthenticationError:
                pool.shutdown(cancel_futures=True)
                sys.exit("API key invalid hai - .env check karo.")
            except Exception as e:  # retries ke baad bhi fail - log karke aage badho
                errors_f.write(f"{now()}\t{query_id}\t{type(e).__name__}: {e}\n")
                errors_f.flush()
                failed += 1
            else:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                saved += 1
            bar.update(1)

    print(f"{saved} saved, {failed} failed" + (f" (details: {ERRORS_FILE})" if failed else ""))


if __name__ == "__main__":
    main()
