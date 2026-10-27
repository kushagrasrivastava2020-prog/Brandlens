import os
import sys

import anthropic
import pandas as pd
from dotenv import load_dotenv

MODEL = "claude-haiku-5-5"

# Windows console ko Hindi/Unicode output ke liye UTF-8 pe set karo
sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
if not os.getenv("ANTHROPIC_API_KEY"):
    sys.exit("ANTHROPIC_API_KEY .env mein set nahi hai.")

queries = pd.read_csv("queries.csv")
first = queries.iloc[0]
print(f"Query [{first['query_id']} | {first['language']}]: {first['query_text']}\n")

client = anthropic.Anthropic()

try:
    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        messages=[{"role": "user", "content": first["query_text"]}],
    )
except anthropic.AuthenticationError:
    sys.exit("API key invalid hai - .env check karo.")
except anthropic.RateLimitError:
    sys.exit("Rate limit hit hua - thodi der baad try karo.")
except anthropic.APIStatusError as e:
    sys.exit(f"API error {e.status_code}: {e.message}")
except anthropic.APIConnectionError:
    sys.exit("API se connect nahi ho paya - internet check karo.")

if response.stop_reason == "refusal":
    category = response.stop_details.category if response.stop_details else None
    sys.exit(f"Model ne refuse kiya (category: {category}).")

for block in response.content:
    if block.type == "text":
        print(block.text)

print(f"\n[stop_reason={response.stop_reason}, "
      f"input_tokens={response.usage.input_tokens}, "
      f"output_tokens={response.usage.output_tokens}]")
