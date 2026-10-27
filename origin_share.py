"""List par depend na karne wala metric: answers ke SAARE brands (list ke andar + bahar) mein Indian-origin ka share."""
import json
import os
import sys
from typing import Literal

import anthropic
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel

from extract import clean_name, compile_alias, match_brand

MODEL = "claude-haiku-5-5"
MAX_TOKENS = 16000
MAX_RETRIES = 3
SENTIMENTS_FILE = "sentiments.jsonl"
ORIGINS_FILE = "unlisted_origins.csv"  # Haiku ke origin tags ka cache
OUTPUT_FILE = "origin_share.csv"
N_BOOT = 1000
SEED = 42
LANGUAGES = ["English", "Hindi", "Hinglish"]

PROMPT = """Below are brand names that an AI assistant mentioned when answering shoppers' questions about {category} in India.

For each name, say where the brand originated:
- Indian: the brand was founded or launched in India, even if a foreign company owns it now (for example Lakmé)
- Global: the brand was founded or launched outside India, even if it is made or sold in India (for example Pond's)
- unknown: you are not sure, or the name is not a brand

Return every name exactly as written below.

{names}"""


class BrandOrigin(BaseModel):
    brand: str
    origin: Literal["Indian", "Global", "unknown"]


class Origins(BaseModel):
    brands: list[BrandOrigin]


def extracted_brands(queries, brands):
    # har answer ke har brand ki ek row; tracked brand ka origin brands.csv se, baaki ka baad mein Haiku se
    patterns = {b.brand_id: [compile_alias(a, b.brand) for a in b.aliases.split("|")] for b in brands.itertuples()}
    by_category = dict(list(brands.groupby("category_id")))
    origin_of = dict(zip(brands["brand_id"], brands["origin"]))
    rows = []
    with open(SENTIMENTS_FILE, encoding="utf-8") as f:
        for record in map(json.loads, f):
            q = queries.loc[record["query_id"]]
            for name in {clean_name(b["brand"]) for b in record["brands"]}:
                brand_id = match_brand(name, by_category[q["category_id"]], patterns)
                rows.append({"query_id": record["query_id"], "group_id": q["group_id"], "language": q["language"],
                             "category_id": q["category_id"], "category": q["category"], "brand": name,
                             "tracked": brand_id is not None, "origin": origin_of.get(brand_id),
                             # "Redmi" aur "Xiaomi" ek hi tracked brand hain - answer mein ek baar gino
                             "key": brand_id if brand_id is not None else name.casefold()})
    return pd.DataFrame(rows).drop_duplicates(["query_id", "key"])


def tag_unlisted(df):
    tagged = pd.read_csv(ORIGINS_FILE) if os.path.exists(ORIGINS_FILE) else \
        pd.DataFrame(columns=["category_id", "category", "brand", "origin"])
    known = set(zip(tagged["category_id"], tagged["brand"].str.casefold()))
    todo = df[~df["tracked"]].drop_duplicates(["category_id", "key"])
    todo = todo[[(c, k) not in known for c, k in zip(todo["category_id"], todo["key"])]]
    print(f"Unlisted names: {len(tagged)} pehle se tagged, {len(todo)} baaki")
    if todo.empty:
        return tagged

    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY .env mein set nahi hai.")
    client = anthropic.Anthropic(max_retries=MAX_RETRIES)
    new, missing = [], 0
    for (category_id, category), g in todo.groupby(["category_id", "category"]):
        names = sorted(g["brand"])
        response = client.messages.parse(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": PROMPT.format(
                category=category, names="\n".join(f"- {n}" for n in names))}],
            output_format=Origins,
        )
        if response.parsed_output is None:
            sys.exit(f"{category}: JSON nahi mila, stop_reason={response.stop_reason}")
        got = {b.brand.casefold(): b.origin for b in response.parsed_output.brands}
        missing += sum(n.casefold() not in got for n in names)
        new += [{"category_id": category_id, "category": category, "brand": n,
                 "origin": got.get(n.casefold(), "unknown")} for n in names]
        print(f"  {category}: {len(names)} names tagged")
    if missing:
        print(f"  {missing} names Haiku ne wapas nahi diye - unknown maane")
    tagged = pd.concat([tagged, pd.DataFrame(new)], ignore_index=True)
    tagged.to_csv(ORIGINS_FILE, index=False, encoding="utf-8")
    return tagged


def bootstrap_share(df):
    # group_id par resample (ek sawaal ke teeno language versions saath); share = Indian / (Indian + Global)
    counts = (df.assign(indian=df["origin"] == "Indian").groupby(["group_id", "language"])["indian"]
              .agg(["sum", "count"]).unstack("language").fillna(0))
    indian = np.stack([counts[("sum", lang)].to_numpy() for lang in LANGUAGES], -1)
    total = np.stack([counts[("count", lang)].to_numpy() for lang in LANGUAGES], -1)

    def share(idx):
        return indian[idx].sum(axis=-2) / total[idx].sum(axis=-2)

    point = share(np.arange(len(counts)))
    boot = share(np.random.default_rng(SEED).integers(0, len(counts), size=(N_BOOT, len(counts))))
    rows = {lang: (point[i], *np.percentile(boot[:, i], [2.5, 97.5])) for i, lang in enumerate(LANGUAGES)}
    for i, j in [(1, 0), (2, 0), (2, 1)]:
        rows[f"{LANGUAGES[i]} − {LANGUAGES[j]}"] = (point[i] - point[j],
                                                   *np.percentile(boot[:, i] - boot[:, j], [2.5, 97.5]))
    return pd.DataFrame(rows, index=["share", "ci_low", "ci_high"]).T


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    queries = pd.read_csv("queries.csv").set_index("query_id")
    brands = pd.read_csv("brands.csv")

    df = extracted_brands(queries, brands)
    tagged = tag_unlisted(df)
    tag_of = {(c, b.casefold()): o for c, b, o in zip(tagged["category_id"], tagged["brand"], tagged["origin"])}
    untracked = ~df["tracked"]
    df.loc[untracked, "origin"] = [tag_of.get((c, k), "unknown")
                                   for c, k in zip(df.loc[untracked, "category_id"], df.loc[untracked, "key"])]

    print(f"\n{len(df)} brand mentions ({df['tracked'].mean():.0%} tracked): "
          f"{df['origin'].value_counts().to_dict()}. Share mein 'unknown' shaamil nahi.")
    known = df[df["origin"].isin(["Indian", "Global"])]

    results = []
    overall = bootstrap_share(known)
    print(f"\nIndian-origin share of ALL brands in answers ({N_BOOT} bootstrap resamples on group_id, 95% CI):")
    for label, r in overall.iterrows():
        signed = "−" in label
        fmt = (lambda x: f"{x:+.1%}") if signed else (lambda x: f"{x:.1%}")
        n = known.loc[known["language"] == label].shape[0] if label in LANGUAGES else ""
        print(f"  {label:<20}{fmt(r.share):>8}   [{fmt(r.ci_low)}, {fmt(r.ci_high)}]   {n}")
        results.append({"level": "overall", "category": "all", "language": label, **r.to_dict(), "brands": n})

    print("\nBy category (share, 95% CI):")
    print(f"  {'category':<18}" + "".join(f"{lang:>24}" for lang in LANGUAGES))
    for category, g in known.groupby("category"):
        res = bootstrap_share(g)
        cells = []
        for lang in LANGUAGES:
            r = res.loc[lang]
            cells.append(f"{r.share:.0%} [{r.ci_low:.0%}, {r.ci_high:.0%}]")
            results.append({"level": "category", "category": category, "language": lang, **r.to_dict(),
                            "brands": int((g["language"] == lang).sum())})
        print(f"  {category:<18}" + "".join(f"{c:>24}" for c in cells))

    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\n{OUTPUT_FILE} likha. Haiku tags: {ORIGINS_FILE}")


if __name__ == "__main__":
    main()
