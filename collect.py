import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import anthropic
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

MODEL = "claude-haiku-5-5"
MAX_TOKENS = 2000
WORKERS = 4
# SDK khud rate limit (429), 5xx aur connection errors pe badhte wait (exponential backoff) ke saath retry karta hai
MAX_RETRIES = 3
ANSWERS_FILE = "answers.jsonl"
ERRORS_FILE = "errors.log"


def now():
    return datetime.now(timezone.utc).isoformat()


def load_done_ids(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["query_id"])
            except (json.JSONDecodeError, KeyError):
                continue  # adhuri/kharab line - woh query dobara chalegi
    return done


def ask(client, query_id, query_text):
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": query_text}],
    )
    answer = "".join(block.text for block in response.content if block.type == "text")
    return {
        "query_id": query_id,
        "model": MODEL,
        "answer": answer,
        "stop_reason": response.stop_reason,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "timestamp": now(),
    }


def main():
    parser = argparse.ArgumentParser(description="queries.csv ke answers Claude API se collect karo")
    parser.add_argument("--limit", type=int, help="sirf pehli N queries chalao")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit 1 ya usse bada hona chahiye")

    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY .env mein set nahi hai.")

    queries = pd.read_csv("queries.csv")
    if args.limit is not None:
        queries = queries.head(args.limit)

    done = load_done_ids(ANSWERS_FILE)
    pending = queries[~queries["query_id"].isin(done)]
    print(f"{len(queries)} queries: {len(queries) - len(pending)} pehle se done, {len(pending)} baaki")
    if pending.empty:
        return

    client = anthropic.Anthropic(max_retries=MAX_RETRIES)
    saved = failed = 0

    # Answers aur errors sirf main thread likhta hai, isliye file pe lock ki zaroorat nahi
    with ThreadPoolExecutor(max_workers=WORKERS) as pool, \
            open(ANSWERS_FILE, "a", encoding="utf-8") as answers_f, \
            open(ERRORS_FILE, "a", encoding="utf-8") as errors_f, \
            tqdm(total=len(pending), desc="Collecting") as bar:
        futures = {
            pool.submit(ask, client, row.query_id, row.query_text): row.query_id
            for row in pending.itertuples()
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
                answers_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                answers_f.flush()
                saved += 1
            bar.update(1)

    print(f"{saved} saved, {failed} failed" + (f" (details: {ERRORS_FILE})" if failed else ""))


if __name__ == "__main__":
    main()
