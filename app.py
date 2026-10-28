import os
import re

import altair as alt
import anthropic
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
from dotenv import load_dotenv

import brand_check

MODEL_FILE = "model_original.pkl"
BRANDS_FILE = "brands.csv"
QUERIES_FILE = "queries.csv"
REFERENCE_CHART = os.path.join("figures", "original", "mention_rate_language_origin.png")
FEATURES = ["language", "category", "query_type", "origin", "size_tier"]  # train.py jaisa order
LANGUAGES = ["English", "Hindi", "Hinglish"]
ORIGINS = ["Indian", "Global"]
TIERS = ["big", "mid", "niche"]
NO_BRAND = "(none)"
BAR_COLOR, UP_COLOR, DOWN_COLOR = "#2a78d6", "#2a78d6", "#e34948"
RAISES, LOWERS = "pushed the score up", "pushed the score down"
# query_type ke seedhe naam; budget wala category ke amount se banta hai (question_label)
QUESTION_LABELS = {
    "best": "Best brands?", "indian": "Good Indian brands?", "beginner": "First-time buyer: which to trust?",
    "gift": "Good brand to gift?", "reliable": "Most reliable?", "top3": "Top 3 in India?",
    "value": "Best value for money?",
}
SIZE_WORDS = {"big": "big, well-known", "mid": "mid-sized", "niche": "small or new"}
# analysis.py se: answers ke kitne % brands original list mein hain
LIST_COVERAGE = {"English": "47%", "Hindi": "76%", "Hinglish": "76%"}
DISCLAIMER = ("Pilot study. One model (Claude Haiku), 360 queries, no live web search. "
              "Predictions come from the brand's profile (category, origin, size) only, not from its name. "
              "Scores are measured within the tracked brand list. English answers mention more brands that are "
              "outside the list, so don't compare scores directly across languages.")
MAX_CHECKS = 3  # har browser session mein "Check my brand" ki limit
CHECK_DISCLAIMER = ("Results come from one model (Claude Haiku), without live web search. "
                    "They are based on 8 queries per language, so treat them as indicative.")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model.named_steps["xgbclassifier"])


@st.cache_data
def load_brands():
    brands = pd.read_csv(BRANDS_FILE)
    return brands[brands["source"] == "original"]


@st.cache_data
def load_queries():
    return pd.read_csv(QUERIES_FILE)


def model_levels(model):
    # dropdown options model ke encoder se, taaki training data se bahar ki value na jaaye
    return dict(zip(FEATURES, (list(c) for c in model.named_steps["onehotencoder"].categories_)))


def predict(model, category, origin, size_tier):
    # har language x har query type ke liye ek row; language ka score = 8 query types ka average
    query_types = model_levels(model)["query_type"]
    rows = pd.DataFrame([{"language": lang, "category": category, "query_type": qt, "origin": origin,
                          "size_tier": size_tier} for lang in LANGUAGES for qt in query_types])
    rows["probability"] = model.predict_proba(rows[FEATURES])[:, 1]
    return rows


def shap_by_feature(model, rows):
    # one-hot SHAP ko wapas 5 features mein jodo, phir rows (8 query types) par average
    encoder = model.named_steps["onehotencoder"]
    names = encoder.get_feature_names_out(FEATURES)
    explanation = load_explainer(model)(encoder.transform(rows[FEATURES]))
    values = pd.DataFrame(explanation.values, columns=names)
    contributions = pd.Series({f: values[[c for c in names if c.startswith(f + "_")]].sum(axis=1).mean()
                               for f in FEATURES})
    return contributions, float(np.mean(explanation.base_values))


def question_label(query_type, category):
    if query_type != "budget":
        return QUESTION_LABELS[query_type]
    # budget sawaal mein category ka asli amount, jaise "Best under ₹40,000?"
    queries = load_queries()
    text = queries[(queries["category"] == category) & (queries["query_type"] == "budget")
                   & (queries["language"] == "English")]["query_text"].iloc[0]
    amount = re.search(r"₹[\d.,]+(?: lakh)?", text)
    return f"Best under {amount.group(0)}?" if amount else "Best under a budget?"


def in_ten(p):
    return f"about {round(p * 10)} in 10 answers"


def query_type_chart(rows, category, language):
    queries = load_queries()
    asked = queries[(queries["category"] == category) & (queries["language"] == language)]
    data = rows.assign(
        question=[question_label(qt, category) for qt in rows["query_type"]],
        asked=[asked.loc[asked["query_type"] == qt, "query_text"].iloc[0] for qt in rows["query_type"]],
        label=[f"{p:.0%}" for p in rows["probability"]],
    )
    base = alt.Chart(data).encode(
        y=alt.Y("question:N", title=None, sort="-x", axis=alt.Axis(labelLimit=220)),
        tooltip=[alt.Tooltip("asked:N", title="Exact question asked"),
                 alt.Tooltip("probability:Q", title="Score", format=".0%")],
    )
    # domain 1.15 tak, taaki 100% wale bar ke aage bhi number ki jagah rahe
    bars = base.mark_bar(color=BAR_COLOR, cornerRadiusEnd=4, height=16).encode(
        x=alt.X("probability:Q", title="Score", scale=alt.Scale(domain=[0, 1.15]),
                axis=alt.Axis(format="%", values=[0, 0.25, 0.5, 0.75, 1])))
    labels = base.mark_text(align="left", dx=4, fontSize=11, color="#898781").encode(
        x="probability:Q", text="label:N")
    return (bars + labels).properties(height=280)


def shap_chart(contributions, labels):
    data = contributions.rename("shap").rename_axis("feature").reset_index()
    data["fact"] = data["feature"].map(labels)
    data["direction"] = np.where(data["shap"] >= 0, RAISES, LOWERS)
    bars = alt.Chart(data).mark_bar(cornerRadiusEnd=4, height=18).encode(
        x=alt.X("shap:Q", title="← lowers · raises →"),
        y=alt.Y("fact:N", title=None, sort=alt.EncodingSortField("shap", op="sum", order="descending"),
                axis=alt.Axis(labelLimit=240)),
        color=alt.Color("direction:N", title=None, legend=alt.Legend(orient="bottom"),
                        scale=alt.Scale(domain=[RAISES, LOWERS], range=[UP_COLOR, DOWN_COLOR])),
        tooltip=[alt.Tooltip("fact:N", title="Fact"), alt.Tooltip("direction:N", title="Effect")],
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#898781").encode(x="x:Q")
    return (bars + zero).properties(height=240)


def get_api_key():
    # .env ya .streamlit/secrets.toml se; key kabhi screen par ya log mein nahi jaani chahiye
    load_dotenv()
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        try:
            key = st.secrets["ANTHROPIC_API_KEY"]
        except Exception:
            key = None
    return key or None


def profile_tab(model, brands, levels):
    st.markdown("**How to read this page:** describe a type of brand on the left. On the right you see how often "
                "an AI names brands like it, which questions name it most, and what raised or lowered the score. "
                "It looks at the brand's type only, not its name, so brands of the same type get the same score.")

    def clear_brand():
        st.session_state["brand"] = NO_BRAND

    def autofill():
        # chuna hua original brand -> uska origin aur size_tier
        if st.session_state["brand"] != NO_BRAND:
            row = brands[(brands["category"] == st.session_state["category"])
                         & (brands["brand"] == st.session_state["brand"])].iloc[0]
            st.session_state["origin"], st.session_state["size_tier"] = row["origin"], row["size_tier"]

    left, right = st.columns([1, 2], gap="large")
    with left:
        st.subheader("1. Describe the brand")
        category = st.selectbox("Category", levels["category"], key="category", on_change=clear_brand)
        st.selectbox("Or fill in from a real brand (optional)",
                     [NO_BRAND] + sorted(brands.loc[brands["category"] == category, "brand"]),
                     key="brand", on_change=autofill)
        origin = st.radio("Origin", ORIGINS, key="origin", horizontal=True, on_change=clear_brand,
                          help="Indian = the brand started in India. Global = it started outside India.")
        size_tier = st.radio("Size", TIERS, key="size_tier", horizontal=True, on_change=clear_brand,
                             help="big = known across India, mid = known within its category, niche = small or new")

    rows = predict(model, category, origin, size_tier)
    scores = {lang: rows.loc[rows["language"] == lang, "probability"].mean() for lang in LANGUAGES}
    with right:
        st.subheader("2. How often does the AI name a brand like this?")
        for col, lang in zip(st.columns(3), LANGUAGES):
            col.metric(f"Asked in {lang}", f"{scores[lang]:.0%}",
                       help=f"{LIST_COVERAGE[lang]} of the brands in {lang} answers are on the tracked list")
            col.caption(in_ten(scores[lang]))
        st.markdown(f"**In plain words:** ask an AI about **{category}**, and a **{SIZE_WORDS[size_tier]} "
                    f"{origin}** brand gets named in about **{round(scores['English'] * 10)} of 10** English answers, "
                    f"**{round(scores['Hindi'] * 10)} of 10** Hindi answers and **{round(scores['Hinglish'] * 10)} "
                    "of 10** Hinglish answers.")
        st.caption("This is the model's estimate for this type of brand, learned from 360 real AI answers "
                   "(a relative visibility score for this pilot), not a forecast for one specific brand. "
                   "Why is English lower? English questions that don't mention India often get US brands that "
                   "aren't on our list. So compare types within one language, not one language against another.")

        st.subheader("3. Look closer")
        language = st.radio("Show details for", LANGUAGES, key="language", horizontal=True)
        lang_rows = rows[rows["language"] == language]
        # ek ke neeche ek, full width: side-by-side mein labels aur bars pichak jaate the
        st.markdown(f"**Which questions name it most ({language})**")
        st.caption("Each bar is one of the 8 questions asked. Hover a bar to see the exact question.")
        st.altair_chart(query_type_chart(lang_rows, category, language), width="stretch")

        contributions, base = shap_by_feature(model, lang_rows)
        labels = {"language": f"Language: {language}", "category": f"Category: {category}",
                  "query_type": "Question type (average of 8)", "origin": f"Origin: {origin}",
                  "size_tier": f"Size: {size_tier}"}
        st.markdown(f"**What pushed the score up or down ({language})**")
        st.caption(f"The model starts from an average brand ({1 / (1 + np.exp(-base)):.0%}). Each bar is one "
                   "fact about this brand: blue pushed the score up, red pushed it down, and a longer bar "
                   "means a bigger push. (This breakdown is called SHAP; the bar lengths are in the model's "
                   "own units, not percent.)")
        st.altair_chart(shap_chart(contributions, labels), width="stretch")

    st.subheader("For comparison: real counts")
    st.caption("How often Indian and global brands were actually named in the 360 answers, by language "
               "(the 160 original brands, all categories). This chart doesn't change with your choices.")
    if os.path.exists(REFERENCE_CHART):
        st.image(REFERENCE_CHART, width=560)
    else:
        st.warning(f"{REFERENCE_CHART} not found. Run `python train.py --subset original` first.")

    st.info(DISCLAIMER)


def render_check(result):
    results, info = result["results"], result["info"]
    st.markdown(f"#### {info['brand']} · {info['category']}")
    st.caption("Matched as: " + ", ".join(info["names"]))
    if info["failed"]:
        reasons = ", ".join(f"{name} ×{n}" for name, n in info["errors"].items())
        st.warning(f"{info['failed']} of 24 questions failed after retries and are left out ({reasons}).")

    summary = brand_check.language_summary(results)
    for col, r in zip(st.columns(3), summary.to_dict("records")):
        col.metric(f"{r['Language']}: answers mentioning it", r["Mentioned in"])
    st.dataframe(summary, hide_index=True, width="stretch")
    st.caption("Avg position = order of first appearance among all brands in the answer (1 = named first). "
               "Sentiment is Claude Haiku's reading of how the answer talks about the brand.")

    st.markdown("**Top 10 other brands in these answers**")
    st.dataframe(brand_check.top_competitors(results), hide_index=True, width="stretch")

    st.download_button("Download the answers (CSV)", brand_check.to_csv(results),
                       file_name=f"brand_check_{info['brand']}_{info['category']}.csv".replace(" ", "_"),
                       mime="text/csv")
    with st.expander(f"Read all {len(results)} answers"):
        for r in results.itertuples():
            status = (f"mentioned, position {int(r.position)}, {r.sentiment or 'sentiment not rated'}"
                      if r.mentioned else "not mentioned")
            with st.container(border=True):
                st.markdown(f"**{r.language} · {r.query_type}** ({status})")
                st.caption(r.query_text)
                st.markdown(r.answer)


def check_tab(levels):
    st.subheader("Check my brand")
    st.caption("Asks Claude Haiku the study's 24 questions for one category right now "
               "(8 query types × English, Hindi, Hinglish), then looks for your brand in the answers.")
    api_key = get_api_key()
    st.session_state.setdefault("checks_used", 0)
    st.session_state.setdefault("session_cost", 0.0)

    with st.form("check_form"):
        brand = st.text_input("Brand name", placeholder="e.g. boAt")
        aliases = st.text_input("Other names (optional, comma-separated)", placeholder="e.g. Airdopes, बोट")
        category = st.selectbox("Category", levels["category"], key="check_category")
        run = st.form_submit_button("Run check", type="primary",
                                    disabled=not api_key or st.session_state["checks_used"] >= MAX_CHECKS)
    if not api_key:
        st.warning("No API key found. Add ANTHROPIC_API_KEY to .env or .streamlit/secrets.toml.")

    if run:
        if st.session_state["checks_used"] >= MAX_CHECKS:  # button band hone ke baad bhi double-check
            st.error(f"This session has used all {MAX_CHECKS} checks.")
        elif not brand.strip():
            st.error("Enter a brand name.")
        else:
            st.session_state["checks_used"] += 1
            progress = st.progress(0.0, text="Asking Claude Haiku…")
            try:
                results, info = brand_check.run_check(
                    brand.strip(), aliases.split(","), category, api_key,
                    on_progress=lambda done, total: progress.progress(done / total, text=f"{done}/{total} questions done"))
            except anthropic.AuthenticationError:
                st.error("The API key was rejected. Check ANTHROPIC_API_KEY.")
            except Exception as e:  # sirf error ka type dikhao, message mein kuch sensitive na jaaye
                st.error(f"The check failed ({type(e).__name__}). Try again in a minute.")
            else:
                st.session_state["session_cost"] += info["cost"]
                if results.empty:
                    reasons = ", ".join(f"{name} ×{n}" for name, n in info["errors"].items())
                    st.error(f"All 24 questions failed ({reasons}). Try again in a minute.")
                else:
                    st.session_state["check_result"] = {"results": results, "info": info}
            progress.empty()

    left = MAX_CHECKS - st.session_state["checks_used"]
    st.caption(f"{left} of {MAX_CHECKS} checks left in this session. Each check makes 48 API calls "
               f"(24 answers + 24 brand extractions), about \\${brand_check.EXPECTED_COST:.2f}. If the brand is in "
               "brands.csv for this category, its known names are matched too.")
    if "check_result" in st.session_state:
        info = st.session_state["check_result"]["info"]
        # "\$" zaroori hai: markdown do $ ke beech ke text ko LaTeX maan leta hai
        st.success(f"Cost of the last check: about \\${info['cost']:.3f} ({info['input_tokens']:,} input + "
                   f"{info['output_tokens']:,} output tokens at claude-haiku-5-5 prices). "
                   f"This session so far: \\${st.session_state['session_cost']:.3f}.")
        render_check(st.session_state["check_result"])

    st.info(CHECK_DISCLAIMER)


def main():
    st.set_page_config(page_title="BrandLens-India", page_icon="🔎", layout="wide")
    st.title("BrandLens-India")
    st.caption("How often AI assistants name different kinds of brands when Indians ask for shopping advice "
               "in English, Hindi or Hinglish")

    model = load_model()
    levels = model_levels(model)
    profile, check = st.tabs(["Profile score", "Check my brand"])
    with profile:
        profile_tab(model, load_brands(), levels)
    with check:
        check_tab(levels)


if __name__ == "__main__":
    main()
