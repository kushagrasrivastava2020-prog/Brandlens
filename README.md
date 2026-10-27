# BrandLens-India

A pilot study of brand visibility in AI-assistant answers to Indian shopping questions in English, Hindi and Hinglish.

**Headline result.** Indian-origin brands make up 20.0% [14.4, 26.0] of all brands named in English answers. In Hindi answers the figure is 40.9% [36.3, 45.6], and in Hinglish answers 42.8% [38.0, 48.0]. These shares don't depend on any brand list. Hindi and Hinglish don't differ from each other. Every final number is in [results.md](results.md).

## How the data was made

1. **Queries** (`queries.csv`). 15 product categories × 8 query types × 3 languages = 360 queries. The query types are best, budget, indian, beginner, gift, reliable, top3 and value. Each `group_id` links the English, Hindi and Hinglish versions of one question.
2. **Answers** (`collect.py` → `answers.jsonl`). Each query went to `claude-haiku-5-5` as a plain user message: no system prompt, no web search, `max_tokens=2000`. At 800 tokens, 8 of the first 10 answers were cut off; that early run is kept as `answers_800.jsonl` for comparison only. The final set has 359 complete answers and 1 truncated.
3. **Brand mentions** (`extract.py` → `mentions.csv`). There is one row per (query, brand in that category) pair. `mentioned` and `position` come from regex matching on the aliases in `brands.csv`. Latin aliases match on word boundaries; Devanagari aliases match as substrings.
4. **Sentiment and all brands** (`sentiment.py` → `sentiments.jsonl`). Haiku read every answer and returned each brand it names with a sentiment (positive / neutral / negative) as structured JSON. This also captures brands that are not on the list (`unlisted_brands.csv`).
5. **Validation.**
   - 30 random answers were read by hand and compared with `mentions.csv`.
   - The regex results were cross-checked against Haiku's extraction for all 360 answers.
   - Misses and false positives found this way led to alias fixes. Examples: "Big Muscles", POCO → Xiaomi, plain "Voltas" for refrigerators. "Active Noise Cancellation", "Google Play" and "Sleepwell Helix" are now excluded.
   - After the fixes, no brand is found only by Haiku. Only 9 pairs are found only by regex.
6. **Brand list expansion.**
   - Starting point: 160 brands (`brands_original.csv`, never edited).
   - Added: 111 brands that Haiku found in 3 or more answers of a category. The rule was applied the same way in every category, and every addition was reviewed in `draft_flags.csv` (sub-brands, duplicates, ambiguous aliases).
   - Every brand carries a `source` column: `original` or `added_from_answers`.
7. **Origin of off-list brands** (`origin_share.py`). Brands on the list use their `origin` from `brands.csv`. The 259 off-list names were tagged Indian / Global / unknown by Haiku, one call per category, and cached in `unlisted_origins.csv`. 50 random tags were checked by hand.

## Brand list bias

The tracked list decides what counts as a mention, and it isn't neutral:

- **Coverage differs by language.** Answers name about 8 brands in every language. The original list covers 47% of the brands in English answers but 76% in Hindi and Hinglish (`analysis.py`). The reason is that English queries without India or ₹ in them get US-centric answers (Osprey, CeraVe, EltaMD and so on), and the original list was built for the Indian market. So the lower English mention rate, the negative `language_English` SHAP value and the lower English scores in the app all reflect list coverage. None of them shows that the model recommends fewer brands in English.
- **Mention rates and positions are relative to the list.** With more brands on the list, every brand's rate would change, and `position` is counted only among tracked brands.
- **Added brands were selected on the outcome.** The 111 `added_from_answers` brands were picked because they appeared in 3+ answers, and 46 of them are Global-niche. Claude drafted their `size_tier`, and Claude is the same model family that wrote the answers, so the tier labels may share the same "memory". The main analysis uses original brands (`train.py --subset original`), or controls for `source`.
- **The list-independent metric avoids all of this.** `origin_share.py` looks at every brand in every answer, so its headline numbers don't depend on the list.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1. Collect answers | `collect.py [--limit N]` | `answers.jsonl` |
| 2. Brands + sentiment per answer (Haiku, JSON) | `sentiment.py [--limit N]` | `sentiments.jsonl` |
| 3. Mentions, position, sentiment | `extract.py` | `mentions.csv`, `unlisted_brands.csv` |
| 4. Models: majority / logistic / XGBoost, SHAP, charts | `train.py [--subset all\|original]` | `model.pkl` + `figures/`, or `model_original.pkl` + `figures/original/` |
| 5. List coverage, Global − Indian gap (bootstrap CI) | `analysis.py` | console tables |
| 6. Indian-origin share of all brands (bootstrap CI) | `origin_share.py` | `origin_share.csv`, `unlisted_origins.csv` |
| 7. Demo app: "Profile score" and "Check my brand" tabs | `streamlit run app.py` | http://localhost:8501 |

Setup:

```bash
python -m venv venv
venv\Scripts\python -m pip install -r requirements.txt
```

Put your key in `.env` as `ANTHROPIC_API_KEY=...`, or in `.streamlit/secrets.toml` for the app. Both are git-ignored.

pandas is pinned to 2.3.3 because Windows Smart App Control blocked a pandas 3.1.0 DLL on the development machine. All outputs are identical under both versions.

## The app

- **Profile score** runs `model_original.pkl` on a profile (category, origin, size tier) and shows a relative visibility score per language, a query-type chart, and a SHAP breakdown. It never sees brand names.
- **Check my brand** (`brand_check.py`) asks `claude-haiku-5-5` the study's 24 questions for one category right now: 8 query types × 3 languages, the same settings as `collect.py`. It then finds the brand in each answer, using regex with the given aliases plus any known ones from `brands.csv`, and records its position among all brands in the answer. Haiku's sentiment extraction is reused for the sentiment and the other brands. Each check makes 48 API calls, about $0.02. A session is limited to 3 checks, and the key is never shown or logged. The results are indicative: one model, no web search, 8 queries per language. Tested on boAt (wireless earbuds): all 24 answers were read by hand, and all 19 mentions and 5 non-mentions were correct.

The API scripts (`collect.py`, `sentiment.py`, `origin_share.py`) skip work that is already saved, run 4 threads where that helps, retry rate limits and server errors 3 times, and log failures to `errors.log` / `sentiment_errors.log`. Everything after step 2 runs offline.

Models use the features `language`, `category`, `query_type`, `origin` and `size_tier`; the brand name is never a feature. They are evaluated with 5-fold `GroupKFold` on `group_id`, so all three language versions of a question fall on the same side of the split.

## brands.csv rules

### size_tier

Tag it from market knowledge, **never from mention counts**. "It appeared a lot, so it's big" would make the model learn its own target.

- `big`: known to everyone across India
- `mid`: known within its category or segment
- `niche`: small or new, or hard to find in India

### origin

`origin` is where the brand was founded or launched (`Indian` / `Global`), wherever its owner is today. Lakmé is `Indian` (now owned by HUL); Pond's is `Global`.

### Adding brands

- Use the same rule in every category: a brand that `unlisted_brands.csv` shows in **3 or more answers** of that category. Adding brands to only some categories would shift those categories' mention rates and create a fake category effect.
- Sub-brands become aliases of their parent, not new brands (Redmi/POCO → Xiaomi, iQOO → Vivo, Sleepyhead → Duroflex).
- New `brand_id`s go after the existing ones, and existing IDs never change. Mark new brands `source = added_from_answers`.
- Give Roman and Devanagari aliases. Drop a Devanagari form if it is also a common Hindi word (क्लिनिक, मीडिया, ट्रेन).
- Names that are common English or Hinglish words are matched case-sensitively (`AMBIGUOUS` in `extract.py`: Simple, Nothing, Bata, Surface and others).
- Never edit `brands_original.csv`.

## Limitations

- **No live web search.** Answers come from the model's own memory, so they may be out of date and may differ from AI products that search the web.
- **One model, one run, 360 queries.** The bootstrap CIs reflect variation across queries only. Brand-level variation and model randomness aren't included.
- **Mention rates are measured within the tracked brand list, and list coverage differs by language (English 47%, Hindi 76%, Hinglish 76%).** Don't compare mention rates directly across languages; use the origin share instead.
- **Global − Indian gap.** Global brands lead in all three languages (original brands; no CI contains 0). The English gap looks about 14 points larger, but the coverage difference means this can't be confirmed from mention rates. Hindi and Hinglish don't differ (+0.2, the CI contains 0).
- **Origin tags.** 11% of origins in the origin share come from Haiku. 50 were hand-checked (49 correct, 1 unknown). 22 mentions (<1%) have unknown origin and are excluded.
- **Added brands** were selected on the outcome, and Claude drafted their `size_tier` (see Brand list bias).
- **Sentiment** is 89% positive and has too little variation to be a useful outcome.
- **The app** (`app.py`) scores a profile (category, origin, size tier), not a brand. Two brands with the same profile always get the same score, and the score is a relative visibility score within the list, not a probability.
