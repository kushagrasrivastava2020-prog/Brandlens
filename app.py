import os

import altair as alt
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

MODEL_FILE = "model_original.pkl"
BRANDS_FILE = "brands.csv"
REFERENCE_CHART = os.path.join("figures", "original", "mention_rate_language_origin.png")
FEATURES = ["language", "category", "query_type", "origin", "size_tier"]  # train.py jaisa order
LANGUAGES = ["English", "Hindi", "Hinglish"]
ORIGINS = ["Indian", "Global"]
TIERS = ["big", "mid", "niche"]
NO_BRAND = "(none)"
BAR_COLOR, UP_COLOR, DOWN_COLOR = "#2a78d6", "#2a78d6", "#e34948"
RAISES, LOWERS = "raises score", "lowers score"
# analysis.py se: answers ke kitne % brands original list mein hain
LIST_COVERAGE = {"English": "47%", "Hindi": "76%", "Hinglish": "76%"}
DISCLAIMER = ("Pilot study. One model (Claude Haiku), 360 queries, no live web search. "
              "Predictions come from the brand's profile (category, origin, size) only, not from its name. "
              "Scores are measured within the tracked brand list. English answers mention more brands that are "
              "outside the list, so don't compare scores directly across languages.")


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


def query_type_chart(rows):
    return alt.Chart(rows).mark_bar(color=BAR_COLOR, cornerRadiusEnd=4, height=16).encode(
        x=alt.X("probability:Q", title="Relative visibility score", scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%")),
        y=alt.Y("query_type:N", title=None, sort="-x"),
        tooltip=[alt.Tooltip("query_type:N", title="query type"),
                 alt.Tooltip("probability:Q", title="score", format=".1%")],
    ).properties(height=260)


def shap_chart(contributions):
    data = contributions.rename("shap").rename_axis("feature").reset_index()
    data["direction"] = np.where(data["shap"] >= 0, RAISES, LOWERS)
    bars = alt.Chart(data).mark_bar(cornerRadiusEnd=4, height=18).encode(
        x=alt.X("shap:Q", title="SHAP (effect on log-odds)"),
        y=alt.Y("feature:N", title=None, sort=alt.EncodingSortField("shap", op="sum", order="descending")),
        color=alt.Color("direction:N", title=None,
                        scale=alt.Scale(domain=[RAISES, LOWERS], range=[UP_COLOR, DOWN_COLOR])),
        tooltip=[alt.Tooltip("feature:N"), alt.Tooltip("shap:Q", title="SHAP", format="+.3f")],
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#898781").encode(x="x:Q")
    return (bars + zero).properties(height=220)


def main():
    st.set_page_config(page_title="BrandLens-India", page_icon="🔎", layout="wide")
    st.title("BrandLens-India")
    st.caption("Relative visibility of a brand profile in AI answers (English / Hindi / Hinglish)")
    st.markdown("This app scores a **profile** (category, origin, size tier), not a brand name. "
                "Two brands with the same profile always get the same score.")

    model = load_model()
    brands = load_brands()
    levels = model_levels(model)

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
        st.subheader("Profile")
        category = st.selectbox("Category", levels["category"], key="category", on_change=clear_brand)
        st.selectbox("Autofill from a brand (optional, original brands only)",
                     [NO_BRAND] + sorted(brands.loc[brands["category"] == category, "brand"]),
                     key="brand", on_change=autofill)
        origin = st.radio("Origin", ORIGINS, key="origin", horizontal=True, on_change=clear_brand)
        size_tier = st.radio("Size tier", TIERS, key="size_tier", horizontal=True, on_change=clear_brand,
                             help="big = known across India, mid = known within its category or segment, "
                                  "niche = small or new")

    rows = predict(model, category, origin, size_tier)
    with right:
        st.subheader("Relative visibility score (pilot)")
        st.caption("Model-estimated mention rate within the tracked brand list, averaged over the 8 query types. "
                   "It is not a probability for any real brand: with more brands on the list, every score would drop.")
        for col, lang in zip(st.columns(3), LANGUAGES):
            col.metric(lang, f"{rows.loc[rows['language'] == lang, 'probability'].mean():.0%}",
                       help=f"{LIST_COVERAGE[lang]} of the brands in {lang} answers are on the tracked list")
        st.caption("Tracked-list coverage of brands in answers: "
                   + ", ".join(f"{lang} {c}" for lang, c in LIST_COVERAGE.items())
                   + ". Lower coverage pulls a language's scores down.")

        language = st.radio("Language for details", LANGUAGES, key="language", horizontal=True)
        lang_rows = rows[rows["language"] == language]
        chart_col, shap_col = st.columns(2)
        with chart_col:
            st.markdown(f"**By query type ({language})**")
            st.altair_chart(query_type_chart(lang_rows), width="stretch")
        with shap_col:
            contributions, base = shap_by_feature(model, lang_rows)
            st.markdown(f"**Why this score: SHAP ({language})**")
            st.altair_chart(shap_chart(contributions), width="stretch")
            st.caption(f"Baseline (training average): {1 / (1 + np.exp(-base)):.0%}. Positive values raise the "
                       "score, negative values lower it. query_type is averaged over the 8 types. A negative "
                       "language bar for English partly reflects lower list coverage, not only model behaviour.")

    st.subheader("Reference: language × origin (original brands)")
    if os.path.exists(REFERENCE_CHART):
        st.image(REFERENCE_CHART, width=560)
    else:
        st.warning(f"{REFERENCE_CHART} not found. Run `python train.py --subset original` first.")

    st.info(DISCLAIMER)


if __name__ == "__main__":
    main()
