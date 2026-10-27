import json
import sys

import numpy as np
import pandas as pd

from extract import clean_name, compile_alias, match_brand

SENTIMENTS_FILE = "sentiments.jsonl"
MENTIONS_FILE = "mentions.csv"
N_BOOT = 1000
SEED = 42
LANGUAGES = ["English", "Hindi", "Hinglish"]


def coverage_table(queries, brands):
    # Haiku ne har answer se jo brands nikale (list ke andar + bahar), unme se kitne brands.csv mein hain
    lists = {"all": brands, "original": brands[brands["source"] == "original"]}
    patterns = {b.brand_id: [compile_alias(a, b.brand) for a in b.aliases.split("|")] for b in brands.itertuples()}
    by_category = {name: dict(list(b.groupby("category_id"))) for name, b in lists.items()}
    rows = []
    with open(SENTIMENTS_FILE, encoding="utf-8") as f:
        for record in map(json.loads, f):
            q = queries.loc[record["query_id"]]
            names = {clean_name(b["brand"]) for b in record["brands"]}
            row = {"language": q["language"], "extracted": len(names)}
            for name, cats in by_category.items():
                category_brands = cats[q["category_id"]]
                row[f"in_{name}"] = sum(match_brand(n, category_brands, patterns) is not None for n in names)
            rows.append(row)
    df = pd.DataFrame(rows)
    out = df.groupby("language").agg(answers=("extracted", "size"), avg_brands=("extracted", "mean"),
                                     extracted=("extracted", "sum"), in_all=("in_all", "sum"),
                                     in_original=("in_original", "sum"))
    out.loc["Total"] = [len(df), df["extracted"].mean(), df["extracted"].sum(), df["in_all"].sum(),
                        df["in_original"].sum()]
    out["avg_in_list"] = out["in_all"] / out["answers"]
    out["coverage_all"] = out["in_all"] / out["extracted"]
    out["coverage_original"] = out["in_original"] / out["extracted"]
    return out


def bootstrap_gaps(mentions):
    # group_id par resample: ek sawaal ke teeno language versions saath aate hain (paired)
    counts = (mentions.groupby(["group_id", "language", "origin"])["mentioned"].agg(["sum", "count"])
              .unstack(["language", "origin"]).fillna(0))
    groups = counts.index
    sums = np.stack([[counts[("sum", lang, o)].to_numpy() for o in ["Global", "Indian"]] for lang in LANGUAGES], -1)
    ns = np.stack([[counts[("count", lang, o)].to_numpy() for o in ["Global", "Indian"]] for lang in LANGUAGES], -1)
    # sums/ns shape: (origin=2, groups, language=3) -> (groups, origin, language)
    sums, ns = sums.transpose(1, 0, 2), ns.transpose(1, 0, 2)

    def gap(idx):
        rate = sums[idx].sum(axis=-3) / ns[idx].sum(axis=-3)  # (..., origin, language)
        return rate[..., 0, :] - rate[..., 1, :]

    point = gap(np.arange(len(groups)))
    rng = np.random.default_rng(SEED)
    boot = gap(rng.integers(0, len(groups), size=(N_BOOT, len(groups))))  # (N_BOOT, language)
    rows = {lang: (point[i], *np.percentile(boot[:, i], [2.5, 97.5])) for i, lang in enumerate(LANGUAGES)}
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        rows[f"{LANGUAGES[i]} − {LANGUAGES[j]}"] = (point[i] - point[j],
                                                   *np.percentile(boot[:, i] - boot[:, j], [2.5, 97.5]))
    return pd.DataFrame(rows, index=["gap", "ci_low", "ci_high"]).T


def pct(x, signed=False):
    return f"{x:+.1%}" if signed else f"{x:.1%}"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    queries = pd.read_csv("queries.csv").set_index("query_id")
    brands = pd.read_csv("brands.csv")
    mentions = pd.read_csv(MENTIONS_FILE)

    cov = coverage_table(queries, brands)
    print("1. Haiku extraction: brands per answer aur brands.csv coverage\n")
    print(f"{'language':<10}{'answers':>8}{'avg brands':>12}{'avg in list':>13}{'coverage (271)':>16}"
          f"{'coverage (original 160)':>25}")
    for lang, r in cov.iterrows():
        print(f"{lang:<10}{int(r.answers):>8}{r.avg_brands:>12.2f}{r.avg_in_list:>13.2f}"
              f"{pct(r.coverage_all):>16}{pct(r.coverage_original):>25}")

    print(f"\n2. Global − Indian mention-rate gap, {N_BOOT} bootstrap resamples on group_id, 95% CI")
    for name, data in [("original brands (160)", mentions[mentions["source"] == "original"]),
                       ("all brands (271)", mentions)]:
        print(f"\n{name}:")
        for label, r in bootstrap_gaps(data).iterrows():
            print(f"  {label:<20}{pct(r.gap, True):>8}   [{pct(r.ci_low, True)}, {pct(r.ci_high, True)}]")


if __name__ == "__main__":
    main()
