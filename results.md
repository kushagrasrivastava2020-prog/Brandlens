# BrandLens-India: final results

All numbers below come from the scripts in this repo (run on 2026-10-27). Regenerate them with:

```bash
python analysis.py              # coverage table, Global − Indian gap CIs
python origin_share.py          # Indian-origin share (uses cached Haiku tags in unlisted_origins.csv)
python train.py --subset all
python train.py --subset original
```

Confidence intervals are 95% percentile intervals from 1,000 bootstrap resamples of `group_id` (the three language versions of a question are resampled together). They reflect variation across queries only: one model, one run, about 160 brands. Brand-level variation and model randomness are not included.

## Data

| Item | Value |
|---|---|
| Queries | 360 = 15 categories × 8 query types × 3 languages (English, Hindi, Hinglish); 120 question groups |
| Answer model | `claude-haiku-5-5`, no system prompt, `max_tokens=2000`, no web search |
| Answers | 360 (359 `end_turn`, 1 `max_tokens`) |
| Brand list | 160 original + 111 `added_from_answers` = 271 brands |
| Query × brand pairs | 6,504 (3,840 for original brands) |
| Mention rate | 50.1% (original brands), 39.7% (all brands) |
| Brands extracted by Haiku | 2,903 mentions, 8.1 per answer |

## 1. Headline: Indian-origin share of all brands in answers

This metric does not depend on the brand list. It counts every brand Haiku extracted from an answer, both on and off the list.

| Language | Indian-origin share | 95% CI | Brands (known origin) |
|---|---|---|---|
| English | 20.0% | [14.4%, 26.0%] | 990 |
| Hindi | 40.9% | [36.3%, 45.6%] | 940 |
| Hinglish | 42.8% | [38.0%, 48.0%] | 951 |

| Difference | Points | 95% CI |
|---|---|---|
| Hindi − English | +20.9 | [+15.8, +26.1] |
| Hinglish − English | +22.8 | [+17.8, +28.1] |
| Hinglish − Hindi | +1.9 | [−0.1, +4.4] |

By category (share, 95% CI; 8 question groups per category, so the intervals are wide):

| Category | English | Hindi | Hinglish |
|---|---|---|---|
| air conditioners | 13% [2, 29] | 38% [33, 47] | 34% [28, 40] |
| backpacks | 17% [2, 44] | 36% [25, 50] | 44% [29, 66] |
| face moisturizer | 25% [3, 53] | 46% [31, 66] | 42% [30, 57] |
| laptops | 8% [0, 23] | 7% [0, 19] | 9% [0, 24] |
| mattresses | 31% [7, 67] | 81% [70, 93] | 86% [71, 98] |
| motorcycles | 33% [14, 57] | 50% [40, 62] | 58% [44, 76] |
| protein powder | 30% [3, 61] | 54% [43, 65] | 60% [50, 68] |
| refrigerators | 11% [1, 24] | 31% [24, 39] | 24% [16, 35] |
| running shoes | 19% [2, 39] | 8% [0, 19] | 9% [2, 19] |
| shampoo | 20% [3, 44] | 46% [37, 57] | 53% [40, 66] |
| smartphones | 9% [0, 23] | 10% [0, 25] | 7% [0, 18] |
| smartwatches | 21% [4, 47] | 48% [38, 63] | 48% [35, 68] |
| sunscreen | 27% [2, 58] | 72% [60, 84] | 68% [57, 81] |
| tea | 21% [3, 50] | 62% [58, 67] | 72% [64, 80] |
| wireless earbuds | 18% [1, 39] | 23% [16, 35] | 29% [19, 50] |

Where the origin tags come from:

- 89% of mentions are matched to `brands.csv` and use its `origin`.
- 11% (259 off-list names) were tagged by Haiku and cached in `unlisted_origins.csv`. A hand check covered 50 random tags: 20 drawn from all tags (19 correct, 1 `unknown`) and 30 drawn from tags with a known origin (30 correct).
- 22 mentions (<1%) are `unknown` and are left out of the share. Most of them are Indian tea estates and D2C brands, so the Hindi/Hinglish shares may be 1–2 points low.

## 2. Brand list coverage

How many of the brands Haiku extracted from each answer are on the tracked list.

| Language | Brands per answer | On list per answer | Coverage (271 brands) | Coverage (original 160) |
|---|---|---|---|---|
| English | 8.33 | 6.57 | 78.8% | 47.1% |
| Hindi | 8.00 | 7.57 | 94.6% | 75.8% |
| Hinglish | 8.03 | 7.50 | 93.4% | 76.1% |
| Total | 8.12 | 7.21 | 88.8% | 66.1% |

Answers name about 8 brands in every language. The original list covers 47% of the brands in English answers but 76% in Hindi and Hinglish. Lower mention rates for English are therefore a coverage gap, not evidence that the model mentions fewer brands in English.

## 3. Global − Indian mention-rate gap

Mention rate of Global brands minus mention rate of Indian brands, within the tracked list.

| | Original brands (160) | All brands (271) |
|---|---|---|
| English | +27.3 [+19.2, +35.3] | +26.8 [+18.3, +35.6] |
| Hindi | +13.8 [+7.2, +20.5] | −2.6 [−8.2, +3.7] |
| Hinglish | +13.6 [+6.4, +20.5] | −4.2 [−10.4, +2.3] |
| English − Hindi | +13.6 [+6.7, +20.5] | +29.5 [+22.1, +37.4] |
| English − Hinglish | +13.8 [+7.1, +20.6] | +31.0 [+23.9, +38.7] |
| Hindi − Hinglish | +0.2 [−4.1, +4.8] | +1.6 [−1.7, +5.3] |

- On original brands, Global brands lead in all three languages. No interval contains 0.
- The English gap looks about 14 points larger. Because English coverage is lower, this difference can't be confirmed from mention rates alone. Section 1 points the same way without depending on the list.
- Hindi and Hinglish don't differ.
- With all brands, the Hindi and Hinglish gaps cross 0, so "Indian brands lead in Hindi" is not supported.

Mention rate by language and origin (original brands):

| Language | Global | Indian |
|---|---|---|
| English | 49.7% | 22.4% |
| Hindi | 63.1% | 49.3% |
| Hinglish | 63.4% | 49.8% |

## 4. Model: predicting `mentioned` (0/1)

Features are `language`, `category`, `query_type`, `origin` and `size_tier`; the brand name is not a feature. Evaluation uses 5-fold `GroupKFold` on `group_id`. F1 and accuracy use a 0.5 threshold. Values are the fold mean ± std.

Original brands (160 brands, 3,840 pairs; the app uses this model):

| Model | AUC | F1 | Accuracy |
|---|---|---|---|
| Majority class | 0.500 ± 0.000 | 0.393 ± 0.321 | 48.5% ± 0.7% |
| Logistic regression | 0.753 ± 0.006 | 0.697 ± 0.006 | 69.6% ± 0.9% |
| XGBoost | 0.850 ± 0.017 | 0.757 ± 0.021 | 76.0% ± 2.1% |

All brands (271 brands, 6,504 pairs):

| Model | AUC | F1 | Accuracy |
|---|---|---|---|
| Majority class | 0.500 ± 0.000 | 0.000 ± 0.000 | 60.3% ± 1.5% |
| Logistic regression | 0.709 ± 0.021 | 0.558 ± 0.018 | 67.7% ± 1.5% |
| XGBoost | 0.830 ± 0.011 | 0.663 ± 0.035 | 74.7% ± 1.6% |

- On original brands, XGBoost predicts correctly for 76% of question × brand pairs, against about 50% for always giving the most common answer. Pooled over the five test folds that is 1,435 correct "mentioned", 1,482 correct "not mentioned", 435 wrong "mentioned" and 488 missed mentions.
- On all brands, always answering "not mentioned" already scores 60.3%, so accuracy alone flatters weak models there. AUC doesn't have that problem, which is why it is the main metric.
- The majority baseline's F1 on original brands swings between folds because the mention rate there is about 50%.

Top 5 SHAP features (mean |SHAP|). The value in brackets is the mean SHAP when the feature is present; positive raises the predicted mention.

| # | All brands | Original brands |
|---|---|---|
| 1 | size_tier_big (+0.69) | size_tier_big (+0.61) |
| 2 | language_English (−0.24) | language_English (−0.75) |
| 3 | size_tier_niche (−0.60) | origin_Global (+0.18) |
| 4 | origin_Global (+0.05) | query_type_indian (+0.51) |
| 5 | query_type_indian (+0.14) | size_tier_mid (−0.18) |

- The negative `language_English` value reflects the coverage gap in section 2. Do not read it as "the model recommends less in English".
- The main directions agree across both runs: big +, niche −, Global +, "indian" queries +. Five smaller effects change sign when the added brands are included: budget, top3, running shoes, mattresses and Hinglish (≈0).

## 5. Sentiment of mentioned brands (original brands)

88.8% positive, 10.9% neutral, 0.3% negative. There's too little variation for sentiment to be a useful outcome in this pilot.
