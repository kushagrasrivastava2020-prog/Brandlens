"""Check my brand: ek brand ke liye study ke 24 sawaal (8 query types x 3 languages) abhi Haiku se poochho aur analyse karo."""
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic
import pandas as pd

from collect import ask
from extract import clean_name, compile_alias, first_match, normalize
from sentiment import extract as extract_brands

QUERIES_FILE = "queries.csv"
BRANDS_FILE = "brands.csv"
WORKERS = 4
MAX_RETRIES = 3
LANGUAGES = ["English", "Hindi", "Hinglish"]
# claude-haiku-5-5, USD per million tokens (100K tokens tak ke prompts)
PRICE_INPUT, PRICE_OUTPUT = 0.10, 0.50
# study ke average se: har check = 24 answers + 24 extractions, ~38K input aur ~36K output tokens
EXPECTED_COST = (38_000 * PRICE_INPUT + 36_000 * PRICE_OUTPUT) / 1e6


def category_queries(category):
    queries = pd.read_csv(QUERIES_FILE)
    return queries[queries["category"] == category].sort_values("query_id")


def known_aliases(brand, category):
    # brand pehle se brands.csv mein hai to uske aliases (Devanagari wale bhi) saath mein match karo
    brands = pd.read_csv(BRANDS_FILE)
    hit = brands[(brands["category"] == category) & (brands["brand"].str.casefold() == brand.casefold())]
    return hit["aliases"].iloc[0].split("|") if len(hit) else []


def brand_matcher(brand, aliases):
    names = list(dict.fromkeys(n.strip() for n in [brand, *aliases] if n.strip()))
    return names, [compile_alias(n, brand) for n in names]


def is_same_brand(name, names, patterns):
    name = normalize(clean_name(name))
    return name.casefold() in {n.casefold() for n in names} or any(p.search(name) for p in patterns)


def first_index(name, text):
    # study wale matching rules hi: case-sensitive ambiguous naam, "Noise cancellation" jaise exclusions
    name = clean_name(name)
    m = compile_alias(name, name).search(text)
    return m.start() if m else None


def brand_order(text, brand, start, others):
    # answer mein pehli baar aane ka order, saare brands ke beech (list ke andar-bahar dono)
    placed = [(s, n) for n in others if (s := first_index(n, text)) is not None]
    if start is not None:
        placed.append((start, brand))
    return [n for _, n in sorted(placed)]


def check_one(client, row, brand, names, patterns, tracked):
    answer = ask(client, row.query_id, row.query_text)
    found = extract_brands(client, row.query_id, row.query_text, row.category, tracked, answer["answer"])
    text = normalize(answer["answer"])

    start = first_match(patterns, text)
    sentiment = next((b["sentiment"] for b in found["brands"] if is_same_brand(b["brand"], names, patterns)), None)
    others = list(dict.fromkeys(clean_name(b["brand"]) for b in found["brands"]
                                if not is_same_brand(b["brand"], names, patterns)))
    order = brand_order(text, brand, start, others)
    return {
        "query_id": row.query_id,
        "language": row.language,
        "query_type": row.query_type,
        "query_text": row.query_text,
        "mentioned": start is not None,
        "position": order.index(brand) + 1 if start is not None else None,
        "sentiment": sentiment,
        "brands_in_order": order,
        "other_brands": others,
        "answer": answer["answer"],
        "input_tokens": answer["input_tokens"] + found["input_tokens"],
        "output_tokens": answer["output_tokens"] + found["output_tokens"],
    }


def run_check(brand, aliases, category, api_key, on_progress=None):
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY nahi mila")
    queries = category_queries(category)
    names, patterns = brand_matcher(brand, [*aliases, *known_aliases(brand, category)])
    extra = [n for n in names if n != brand]
    tracked = f"{brand} ({', '.join(extra)})" if extra else brand
    client = anthropic.Anthropic(api_key=api_key, max_retries=MAX_RETRIES)

    rows, errors = [], Counter()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(check_one, client, row, brand, names, patterns, tracked)
                   for row in queries.itertuples()]
        for done, future in enumerate(as_completed(futures), start=1):
            try:
                rows.append(future.result())
            except anthropic.AuthenticationError:
                pool.shutdown(cancel_futures=True)
                raise
            except Exception as e:  # retries ke baad bhi fail - sirf error ki type gino, aage badho
                errors[type(e).__name__] += 1
            if on_progress:
                on_progress(done, len(futures))

    results = pd.DataFrame(rows)
    if not results.empty:  # language, phir study wala query order
        results = results.sort_values(["language", "query_id"],
                                      key=lambda s: s.map(LANGUAGES.index) if s.name == "language" else s)
    tokens_in = int(results["input_tokens"].sum()) if not results.empty else 0
    tokens_out = int(results["output_tokens"].sum()) if not results.empty else 0
    return results, {
        "brand": brand, "category": category, "names": names,
        "failed": sum(errors.values()), "errors": dict(errors),
        "input_tokens": tokens_in, "output_tokens": tokens_out,
        "cost": (tokens_in * PRICE_INPUT + tokens_out * PRICE_OUTPUT) / 1e6,
    }


def language_summary(results):
    rows = []
    for lang in LANGUAGES:
        d = results[results["language"] == lang]
        m = d[d["mentioned"]]
        sentiments = Counter(m["sentiment"].fillna("not rated"))
        rows.append({
            "Language": lang,
            "Mentioned in": f"{len(m)}/{len(d)}",
            "Avg position": round(m["position"].mean(), 1) if len(m) else None,
            "Positive": sentiments["positive"],
            "Neutral": sentiments["neutral"],
            "Negative": sentiments["negative"],
        })
    return pd.DataFrame(rows)


def top_competitors(results, n=10):
    # "realme" aur "Realme" ek hi gine jaate hain; har answer mein ek brand ek hi baar
    seen, names, positions = {}, {}, {}
    for r in results.itertuples():
        for name in r.other_brands:
            key = name.casefold()
            names.setdefault(key, Counter())[name] += 1
            seen.setdefault(key, Counter())[r.language] += 1
            if name in r.brands_in_order:
                positions.setdefault(key, []).append(r.brands_in_order.index(name) + 1)
    rows = [{
        "Brand": names[k].most_common(1)[0][0],
        f"Answers (of {len(results)})": sum(c.values()),
        **{lang: c[lang] for lang in LANGUAGES},
        "Avg position": round(sum(positions[k]) / len(positions[k]), 1) if positions.get(k) else None,
    } for k, c in seen.items()]
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values([f"Answers (of {len(results)})", "Brand"], ascending=[False, True]).head(n)


def to_csv(results):
    out = results.drop(columns=["input_tokens", "output_tokens"]).copy()
    for col in ["brands_in_order", "other_brands"]:
        out[col] = out[col].map("|".join)
    return out.to_csv(index=False).encode("utf-8-sig")  # BOM taaki Excel Hindi sahi dikhaye
