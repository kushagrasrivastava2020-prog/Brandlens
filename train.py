import argparse
import os
import sys

import joblib
import matplotlib

matplotlib.use("Agg")  # bina GUI ke PNG save karo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.ticker import PercentFormatter
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

MENTIONS_FILE = "mentions.csv"
MODEL_FILE = "model.pkl"
FIG_DIR = "figures"
FEATURES = ["language", "category", "query_type", "origin", "size_tier"]  # brand ka naam jaan-boojh ke nahi
TARGET = "mentioned"
N_FOLDS = 5
SEED = 42

LANGUAGES = ["English", "Hindi", "Hinglish"]
SOURCES = ["original", "added_from_answers"]
TIERS = ["big", "mid", "niche"]

# Chart colors (light mode): surface, ink, aur categorical/diverging slots
SURFACE, INK, INK_2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
ORIGIN_COLORS = {"Global": "#2a78d6", "Indian": "#eb6834"}
TIER_COLOR = "#6250d6"
SENTIMENT_COLORS = {"negative": "#e34948", "neutral": "#c3c2b7", "positive": "#2a78d6"}


def make_models():
    return {
        "Majority class": make_pipeline(OneHotEncoder(handle_unknown="ignore"),
                                        DummyClassifier(strategy="most_frequent")),
        "Logistic regression": make_pipeline(OneHotEncoder(handle_unknown="ignore"),
                                             LogisticRegression(max_iter=1000)),
        "XGBoost": make_pipeline(OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                                 XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                                               colsample_bytree=0.8, eval_metric="logloss", random_state=SEED)),
    }


def cross_validate(df):
    # group_id ke hisaab se split: ek sawaal ke teeno language versions ek hi fold mein rehte hain
    X, y, groups = df[FEATURES], df[TARGET], df["group_id"]
    results = {}
    for name, model in make_models().items():
        aucs, f1s, accs = [], [], []
        for train_idx, test_idx in GroupKFold(n_splits=N_FOLDS).split(X, y, groups):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            proba = model.predict_proba(X.iloc[test_idx])[:, 1]
            pred = (proba >= 0.5).astype(int)
            aucs.append(roc_auc_score(y.iloc[test_idx], proba))
            f1s.append(f1_score(y.iloc[test_idx], pred, zero_division=0))
            accs.append(accuracy_score(y.iloc[test_idx], pred))
        results[name] = {"auc": (np.mean(aucs), np.std(aucs)), "f1": (np.mean(f1s), np.std(f1s)),
                         "acc": (np.mean(accs), np.std(accs))}
    return results


def shap_summary(model, df, path):
    encoder, xgb = model.named_steps["onehotencoder"], model.named_steps["xgbclassifier"]
    X = pd.DataFrame(encoder.transform(df[FEATURES]), columns=encoder.get_feature_names_out(FEATURES))
    explanation = shap.TreeExplainer(xgb)(X)
    plt.figure()
    shap.plots.beeswarm(explanation, max_display=20, show=False)
    plt.title("SHAP: kaunsi feature mention ki probability badhati/ghatati hai", loc="left", fontsize=11)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    values = pd.DataFrame(explanation.values, columns=X.columns)
    return pd.DataFrame({
        "mean_abs_shap": values.abs().mean(),
        # direction: jab feature present (=1) ho tab average SHAP - plus matlab mention ki probability badhti hai
        "shap_when_present": [values.loc[X[c] == 1, c].mean() for c in X.columns],
    }).sort_values("mean_abs_shap", ascending=False)


def style_axes(ax, percent=True):
    ax.set_facecolor(SURFACE)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))


def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def present_sources(df):
    return [s for s in SOURCES if s in set(df["source"])]


def plot_language_origin(df, path):
    # source ke hisaab se alag panel: added brands answers se chune gaye hain, unhe original ke saath mat milao
    sources = present_sources(df)
    fig, axes = plt.subplots(1, len(sources), figsize=(5 * len(sources), 4), sharey=True, facecolor=SURFACE,
                             squeeze=False)
    axes = axes[0]
    x, width = np.arange(len(LANGUAGES)), 0.38
    for ax, source in zip(axes, sources):
        sub = df[df["source"] == source]
        rates = sub.groupby(["language", "origin"])[TARGET].mean().unstack().reindex(
            index=LANGUAGES, columns=list(ORIGIN_COLORS))
        for k, (origin, color) in enumerate(ORIGIN_COLORS.items()):
            bars = ax.bar(x + (k - 0.5) * width, rates[origin], width, color=color,
                          edgecolor=SURFACE, linewidth=2, label=origin)
            ax.bar_label(bars, labels=[f"{v:.0%}" for v in rates[origin]], padding=2, color=INK_2, fontsize=9)
        ax.set_xticks(x, LANGUAGES)
        ax.set_title(f"{source} ({sub['brand_id'].nunique()} brands)", loc="left", color=INK, fontsize=11)
        style_axes(ax)
    axes[0].set_ylabel("Mention rate", color=INK_2)
    # legend title ke neeche ek line mein, taaki bars ya title ke upar na aaye (1 ya 2 panel dono mein)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.06, 1.03), ncol=2,
               frameon=False, labelcolor=INK_2)
    fig.suptitle("Language × origin: average mention rate", x=0.06, y=1.1, ha="left", color=INK, fontsize=13)
    save(fig, path)


def plot_size_tier(df, path):
    sources = present_sources(df)
    fig, axes = plt.subplots(1, len(sources), figsize=(5 * len(sources), 4), sharey=True, facecolor=SURFACE,
                             squeeze=False)
    axes = axes[0]
    for ax, source in zip(axes, sources):
        sub = df[df["source"] == source]
        rates = sub.groupby("size_tier")[TARGET].mean().reindex(TIERS)
        n_brands = sub.groupby("size_tier")["brand_id"].nunique().reindex(TIERS).fillna(0).astype(int)
        bars = ax.bar(TIERS, rates, 0.6, color=TIER_COLOR, edgecolor=SURFACE, linewidth=2)
        ax.bar_label(bars, labels=[f"{v:.0%}" for v in rates], padding=2, color=INK_2, fontsize=9)
        ax.set_xticks(range(len(TIERS)), [f"{t}\n({n} brands)" for t, n in zip(TIERS, n_brands)])
        ax.set_title(source, loc="left", color=INK, fontsize=11)
        style_axes(ax)
    axes[0].set_ylabel("Mention rate", color=INK_2)
    fig.suptitle("size_tier: average mention rate", x=0.06, y=1.04, ha="left", color=INK, fontsize=13)
    save(fig, path)


def plot_sentiment(df, path):
    sub = df[(df[TARGET] == 1) & df["sentiment"].notna()]
    rows = [(lang, origin) for lang in LANGUAGES for origin in ORIGIN_COLORS]
    shares = (sub.groupby(["language", "origin"])["sentiment"].value_counts(normalize=True)
              .unstack().reindex(index=pd.MultiIndex.from_tuples(rows), columns=list(SENTIMENT_COLORS)).fillna(0))
    counts = sub.groupby(["language", "origin"]).size().reindex(pd.MultiIndex.from_tuples(rows))
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=SURFACE)
    y = np.arange(len(rows))[::-1]
    left = np.zeros(len(rows))
    for sentiment, color in SENTIMENT_COLORS.items():
        vals = shares[sentiment].to_numpy()
        ax.barh(y, vals, left=left, color=color, edgecolor=SURFACE, linewidth=2, label=sentiment)
        for yi, l, v in zip(y, left, vals):
            if v >= 0.05:  # chhote segments par label nahi, warna bheed ho jaati hai
                ax.text(l + v / 2, yi, f"{v:.0%}", ha="center", va="center", fontsize=9,
                        color=SURFACE if sentiment != "neutral" else INK)
        left += vals
    ax.set_yticks(y, [f"{lang} · {origin}  (n={n})" for (lang, origin), n in zip(rows, counts)])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    style_axes(ax, percent=False)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.legend(frameon=False, labelcolor=INK_2, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.suptitle("Sentiment of mentioned brands: language × origin", x=0.06, y=1.06, ha="left", color=INK, fontsize=13)
    save(fig, path)


def main():
    parser = argparse.ArgumentParser(description="mentioned (0/1) predict karo: baselines, XGBoost, SHAP, charts")
    parser.add_argument("--subset", choices=["all", "original"], default="all",
                        help="original = sirf source == original brands (answers se jode gaye brands bahar)")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    df = pd.read_csv(MENTIONS_FILE)
    model_file, fig_dir = MODEL_FILE, FIG_DIR
    if args.subset == "original":
        df = df[df["source"] == "original"].reset_index(drop=True)
        model_file, fig_dir = "model_original.pkl", os.path.join(FIG_DIR, "original")
    print(f"subset={args.subset}: {len(df)} rows, {df['brand_id'].nunique()} brands, "
          f"{df['group_id'].nunique()} groups, mention rate {df[TARGET].mean():.1%}")
    print(f"Features: {FEATURES} | {N_FOLDS}-fold GroupKFold on group_id | F1 and accuracy at threshold 0.5\n")

    results = cross_validate(df)
    print(f"{'model':<22}{'AUC':>18}{'F1':>18}{'Accuracy':>20}")
    for name, r in results.items():
        print(f"{name:<22}{r['auc'][0]:>11.3f} ± {r['auc'][1]:.3f}{r['f1'][0]:>11.3f} ± {r['f1'][1]:.3f}"
              f"{r['acc'][0]:>12.1%} ± {r['acc'][1]:.1%}")

    # Final XGBoost poore data par, SHAP aur save ke liye
    model = make_models()["XGBoost"].fit(df[FEATURES], df[TARGET])
    joblib.dump(model, model_file)

    os.makedirs(fig_dir, exist_ok=True)
    importance = shap_summary(model, df, os.path.join(fig_dir, "shap_summary.png"))
    plot_language_origin(df, os.path.join(fig_dir, "mention_rate_language_origin.png"))
    plot_size_tier(df, os.path.join(fig_dir, "mention_rate_size_tier.png"))
    plot_sentiment(df, os.path.join(fig_dir, "sentiment_language_origin.png"))

    print("\nTop 10 features (mean |SHAP|, aur feature present hone par average SHAP):")
    for feature, r in importance.head(10).iterrows():
        print(f"  {r.mean_abs_shap:.3f}  {feature:<28} {r.shap_when_present:+.3f}")
    print(f"\nModel: {model_file} | Figures: {fig_dir}/")


if __name__ == "__main__":
    main()
