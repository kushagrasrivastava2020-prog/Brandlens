import json
import os
import re
import sys
import unicodedata
from collections import Counter

import pandas as pd

ANSWERS_FILE = "answers.jsonl"
SENTIMENTS_FILE = "sentiments.jsonl"
OUTPUT_FILE = "mentions.csv"
UNLISTED_FILE = "unlisted_brands.csv"

# Ye naam aam English words bhi hain, isliye inke aliases sirf exact case mein match honge
AMBIGUOUS = {w.lower() for w in [
    "Simple", "Nothing", "Noise", "Plum", "Safari", "Campus", "Hero", "Dove",
    "Lotus", "Carrier", "Mi", "ON", "HP", "LG", "Lava", "Titan",
    "Minimalist",  # review mein "a clean, minimalist vibe" galat pakda gaya tha
    "Society",  # Society Tea ke answers mein akela "Society" bhi aata hai
    # naye brands (draft_flags.csv): aam English/Hinglish shabd
    "Google", "Pixel", "Surface", "Framework", "Bata", "Brooks", "Suave", "Purple", "Nectar", "Helix", "Triumph",
]}
GOOGLE_SERVICES = r"\s+(?:Play|Pay|Assistant|Maps|Photos|Search|Lens|Gemini|Fit|Home|Wallet|Drive|Account|Messages|Store|One|Translate|TV)\b"
# Brand ke baad ye shabd aaye to woh brand nahi hai ("Active Noise Cancellation", "नॉइज़ कैंसलेशन", "Google Play")
NOT_FOLLOWED_BY = {
    "Noise": r"[\s-]*(?i:cancel|isolat|reduc|कैंसल|कैंसिल|रिडक्शन)",
    "Google Pixel": GOOGLE_SERVICES,
    "Google Pixel Buds": GOOGLE_SERVICES,
}
# Brand se pehle ye shabd ho to woh doosre brand ki product line hai ("Sleepwell Helix")
NOT_PRECEDED_BY = {
    "Helix": "Sleepwell ",
}
DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def normalize(text):
    # ज़ / é jaise letters ke do Unicode roop hote hain; curly apostrophe (Pond’s) ko bhi seedha karo
    return unicodedata.normalize("NFC", text).replace("’", "'").replace("‘", "'")


def compile_alias(alias, brand):
    alias = normalize(alias)
    before = f"(?<!{re.escape(NOT_PRECEDED_BY[brand])})" if brand in NOT_PRECEDED_BY else ""
    after = f"(?!{NOT_FOLLOWED_BY[brand]})" if brand in NOT_FOLLOWED_BY else ""
    if DEVANAGARI.search(alias):
        return re.compile(before + re.escape(alias) + after)
    # Case-sensitive: 3 ya kam letters, ya alias/brand ka koi word AMBIGUOUS mein ho
    # (jaise "ON Gold Standard", aur Hero ka "Splendor")
    words = {w.lower() for w in alias.split()} | {brand.lower()}
    flags = 0 if len(alias) <= 3 or words & AMBIGUOUS else re.IGNORECASE
    return re.compile(rf"{before}\b{re.escape(alias)}\b{after}", flags)


def first_match(patterns, text):
    starts = [m.start() for p in patterns if (m := p.search(text))]
    return min(starts) if starts else None


def clean_name(name):
    return re.sub(r"\s*\(.*?\)", "", normalize(name)).strip()  # "Apple (iPhone)" -> "Apple"


def match_brand(name, category_brands, patterns):
    # LLM ka diya naam -> brand_id: pehle exact brand naam, phir aliases (jaise "Redmi" -> Xiaomi)
    name = clean_name(name)
    for b in category_brands.itertuples():
        if name.casefold() == normalize(b.brand).casefold():
            return b.brand_id
    for b in category_brands.itertuples():
        if any(p.search(name) for p in patterns[b.brand_id]):
            return b.brand_id
    # LLM ne sirf parent naam likha ("Microsoft" -> "Microsoft Surface"), aur category mein aisa ek hi brand hai
    prefixed = [b.brand_id for b in category_brands.itertuples()
                if normalize(b.brand).casefold().startswith(name.casefold() + " ")]
    return prefixed[0] if len(prefixed) == 1 else None


def load_sentiments(path, category_of, brands_by_category, patterns):
    # query_id -> {brand_id: sentiment}; jo naam brands.csv se match nahi hua woh unlisted mein jaata hai,
    # (category_id, naam) ke hisaab se - "Realme"/"realme" ek hi gine jaate hain, aur count unique answers ka hai
    sentiments, unlisted = {}, {}
    if not os.path.exists(path):
        return sentiments, unlisted
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            category_id = category_of.get(record["query_id"])
            if category_id is None:
                continue
            category_brands = brands_by_category[category_id]
            found = sentiments.setdefault(record["query_id"], {})
            for item in record["brands"]:
                brand_id = match_brand(item["brand"], category_brands, patterns)
                if brand_id is None:
                    entry = unlisted.setdefault((category_id, clean_name(item["brand"]).casefold()),
                                                {"answers": set(), "names": Counter()})
                    entry["answers"].add(record["query_id"])
                    entry["names"][clean_name(item["brand"])] += 1
                else:
                    found.setdefault(brand_id, item["sentiment"])
    return sentiments, unlisted


def pct(x):
    return f"{x:.1%}"


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    queries = pd.read_csv("queries.csv")
    brands = pd.read_csv("brands.csv")
    with open(ANSWERS_FILE, encoding="utf-8") as f:
        answers = pd.DataFrame([json.loads(line) for line in f if line.strip()])
    answers = answers.drop_duplicates("query_id", keep="last")
    df = answers.merge(queries, on="query_id", how="inner")

    patterns = {
        b.brand_id: [compile_alias(a.strip(), b.brand) for a in b.aliases.split("|") if a.strip()]
        for b in brands.itertuples()
    }
    brands_by_category = dict(list(brands.groupby("category_id")))
    sentiments, unlisted = load_sentiments(
        SENTIMENTS_FILE, dict(zip(df["query_id"], df["category_id"])), brands_by_category, patterns)

    rows = []
    llm_only, regex_only = [], 0
    for a in df.itertuples():
        text = normalize(a.answer)
        category_brands = brands_by_category[a.category_id]
        first_seen = {}
        for b in category_brands.itertuples():
            start = first_match(patterns[b.brand_id], text)
            if start is not None:
                first_seen[b.brand_id] = start
        order = sorted(first_seen, key=lambda bid: (first_seen[bid], bid))
        position = {bid: i for i, bid in enumerate(order, start=1)}
        answer_sentiments = sentiments.get(a.query_id)
        for b in category_brands.itertuples():
            mentioned = b.brand_id in position
            sentiment = None
            if answer_sentiments is not None:
                sentiment = answer_sentiments.get(b.brand_id)
                if mentioned and sentiment is None:
                    regex_only += 1
                elif not mentioned and sentiment is not None:
                    llm_only.append(f"{a.query_id}:{b.brand}")
                    sentiment = None  # jo brand mention nahi hua uska sentiment khali rahe
            rows.append({
                "query_id": a.query_id,
                "group_id": a.group_id,
                "category": a.category,
                "query_type": a.query_type,
                "language": a.language,
                "brand_id": b.brand_id,
                "brand": b.brand,
                "origin": b.origin,
                "size_tier": b.size_tier,
                "source": b.source,
                "mentioned": int(mentioned),
                "position": position.get(b.brand_id),
                "sentiment": sentiment,
            })

    mentions = pd.DataFrame(rows).sort_values(["query_id", "brand_id"])
    mentions["position"] = mentions["position"].astype("Int64")
    mentions.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    # ---- Summary ----
    truncated = (df["stop_reason"] == "max_tokens").sum()
    print(f"{len(df)} answers ({len(queries)} queries mein se, {truncated} truncated) "
          f"-> {len(mentions)} rows {OUTPUT_FILE} mein")

    per_answer = mentions.groupby(["query_id", "language"], as_index=False)["mentioned"].sum()
    zero = per_answer[per_answer["mentioned"] == 0]
    print(f"\n0 brands wale answers (suspicious): {len(zero)}")
    if not zero.empty:
        print("  " + ", ".join(f"{r.query_id} ({r.language})" for r in zero.itertuples()))

    print("\nLanguage-wise mention rate:")
    by_language = pd.DataFrame({
        "answers": per_answer.groupby("language").size(),
        "mention_rate": mentions.groupby("language")["mentioned"].mean().map(pct),
        "avg_brands_per_answer": per_answer.groupby("language")["mentioned"].mean().round(2),
    })
    print(by_language.to_string())

    print("\nOrigin-wise mention rate:")
    by_origin = pd.DataFrame({
        "pairs": mentions.groupby("origin").size(),
        "mention_rate": mentions.groupby("origin")["mentioned"].mean().map(pct),
    })
    print(by_origin.to_string())

    # added_from_answers brands answers se chune gaye hain (3+ mentions), isliye unka rate alag dekho
    print("\nSource-wise mention rate:")
    by_source = pd.DataFrame({
        "brands": mentions.groupby("source")["brand_id"].nunique(),
        "pairs": mentions.groupby("source").size(),
        "mention_rate": mentions.groupby("source")["mentioned"].mean().map(pct),
    })
    print(by_source.to_string())

    if not sentiments:
        print(f"\n{SENTIMENTS_FILE} nahi mila - sentiment column khali hai (pehle sentiment.py chalao)")
        return

    covered = mentions["query_id"].isin(sentiments.keys())
    mentioned_rows = mentions[covered & (mentions["mentioned"] == 1)]
    print(f"\nSentiment ({len(sentiments)} answers): {mentioned_rows['sentiment'].notna().sum()} / "
          f"{len(mentioned_rows)} mentioned pairs ko sentiment mila")
    print(f"  regex ne pakda, LLM ne nahi: {regex_only} | LLM ne pakda, regex ne nahi: {len(llm_only)}")
    if llm_only:
        print("  LLM-only examples:", ", ".join(llm_only[:15]))
    print("\nSentiment by language (mentioned pairs):")
    print(pd.crosstab(mentioned_rows["language"], mentioned_rows["sentiment"], normalize="index")
          .map(pct).to_string())
    category_name = dict(zip(brands["category_id"], brands["category"]))
    unlisted_df = pd.DataFrame([
        {
            "category_id": category_id,
            "category": category_name[category_id],
            "brand": entry["names"].most_common(1)[0][0],
            "answers": len(entry["answers"]),
            "spellings": "|".join(entry["names"]),
        }
        for (category_id, _), entry in unlisted.items()
    ]).sort_values(["category_id", "answers", "brand"], ascending=[True, False, True])
    unlisted_df.to_csv(UNLISTED_FILE, index=False, encoding="utf-8")
    print(f"\n{len(unlisted_df)} unlisted brands {UNLISTED_FILE} mein. Sabse zyada aane wale:")
    for r in unlisted_df.sort_values("answers", ascending=False).head(15).itertuples():
        print(f"  {r.answers:>3}  {r.category}: {r.brand}")


if __name__ == "__main__":
    main()
