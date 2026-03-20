"""Phase 4: Exploratory Data Analysis (EDA) on trader + sentiment merged dataset."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from preprocess_phase2 import (
    SENTIMENT_DATA_PATH,
    TRADER_DATA_PATH,
    preprocess,
)


def standardize_sentiment_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize sentiment labels for consistent analysis."""
    out = df.copy()
    out["classification"] = (
        out["classification"].astype("string").str.strip().str.title().fillna("Unknown")
    )
    return out


def plot_average_profit_by_sentiment(df: pd.DataFrame) -> None:
    """1) Compare average profit across sentiment."""
    metric = df.groupby("classification", dropna=False)["profit"].mean().sort_values()

    plt.figure(figsize=(8, 4))
    metric.plot(kind="bar", color=["#d62728" if x == "Fear" else "#2ca02c" if x == "Greed" else "#7f7f7f" for x in metric.index])
    plt.title("Average Profit (closedPnL) by Sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Average Profit")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()

    best_sentiment = metric.idxmax()
    diff = float(metric.max() - metric.min())
    print(
        f"Interpretation: Average trade profit is highest during {best_sentiment}. "
        f"The spread between best and worst sentiment is {diff:.4f}, suggesting sentiment-linked performance differences."
    )


def plot_win_rate_by_sentiment(df: pd.DataFrame) -> None:
    """2) Compute and plot win rate by sentiment."""
    win_rate = (df.groupby("classification", dropna=False)["is_profit"].mean() * 100).sort_values()

    plt.figure(figsize=(8, 4))
    win_rate.plot(kind="bar", color=["#d62728" if x == "Fear" else "#2ca02c" if x == "Greed" else "#7f7f7f" for x in win_rate.index])
    plt.title("Win Rate (% Profitable Trades) by Sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Win Rate (%)")
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()

    best_sentiment = win_rate.idxmax()
    gap = float(win_rate.max() - win_rate.min())
    print(
        f"Interpretation: {best_sentiment} has the highest win rate. "
        f"The win-rate gap across sentiment states is {gap:.2f} percentage points."
    )


def plot_leverage_by_sentiment(df: pd.DataFrame) -> None:
    """3) Analyze leverage usage across sentiment."""
    if "leverage" not in df.columns or df["leverage"].dropna().empty:
        print("Interpretation: Leverage column is unavailable or empty, so leverage analysis is skipped.")
        return

    subset = df[["classification", "leverage"]].dropna().copy()

    plt.figure(figsize=(9, 5))
    subset.boxplot(column="leverage", by="classification", grid=False)
    plt.title("Leverage Distribution by Sentiment")
    plt.suptitle("")
    plt.xlabel("Sentiment")
    plt.ylabel("Leverage")
    plt.tight_layout()
    plt.show()

    lev_summary = subset.groupby("classification")["leverage"].median().sort_values()
    high = lev_summary.idxmax()
    print(
        f"Interpretation: Median leverage is highest during {high}. "
        "This indicates traders take relatively more risk exposure in that sentiment regime."
    )


def _resolve_side_column(df: pd.DataFrame) -> str | None:
    """Find side-like column used for buy/sell behavior analysis."""
    for col in ["side", "direction"]:
        if col in df.columns:
            return col
    return None


def plot_buy_sell_behavior_by_sentiment(df: pd.DataFrame) -> None:
    """4) Analyze buy vs sell behavior across sentiment."""
    side_col = _resolve_side_column(df)
    if side_col is None:
        print("Interpretation: Neither side nor direction column is present, so buy/sell analysis is skipped.")
        return

    tmp = df[["classification", side_col]].dropna().copy()
    tmp[side_col] = tmp[side_col].astype("string").str.strip().str.upper()
    tmp[side_col] = tmp[side_col].replace({"BUY": "BUY", "SELL": "SELL"})

    behavior = pd.crosstab(tmp["classification"], tmp[side_col], normalize="index") * 100
    behavior = behavior.reindex(columns=[c for c in ["BUY", "SELL"] if c in behavior.columns])

    plt.figure(figsize=(9, 5))
    behavior.plot(kind="bar", stacked=True, figsize=(9, 5), color=["#1f77b4", "#ff7f0e"])
    plt.title("Buy vs Sell Behavior by Sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Share of Trades (%)")
    plt.legend(title="Side")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()

    if {"BUY", "SELL"}.issubset(behavior.columns):
        buy_heaviest = behavior["BUY"].idxmax()
        print(
            f"Interpretation: {buy_heaviest} shows the strongest BUY bias. "
            "The stacked distribution highlights how directional preference shifts across sentiment regimes."
        )
    else:
        print(
            "Interpretation: Trade-side composition differs by sentiment, "
            "indicating regime-dependent directional behavior."
        )


def plot_profit_distribution(df: pd.DataFrame) -> None:
    """5) Show distribution of profits with histogram and boxplot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram (overall)
    axes[0].hist(df["profit"].dropna(), bins=60, color="#4c72b0", alpha=0.85)
    axes[0].set_title("Profit Distribution (Histogram)")
    axes[0].set_xlabel("Profit")
    axes[0].set_ylabel("Trade Count")
    axes[0].grid(alpha=0.2)

    # Boxplot by sentiment
    grouped = [
        grp["profit"].dropna().values
        for _, grp in df.groupby("classification", dropna=False)
    ]
    labels = [str(k) for k, _ in df.groupby("classification", dropna=False)]

    axes[1].boxplot(grouped, labels=labels, showfliers=False)
    axes[1].set_title("Profit Distribution by Sentiment (Boxplot)")
    axes[1].set_xlabel("Sentiment")
    axes[1].set_ylabel("Profit")
    axes[1].grid(alpha=0.2)

    plt.tight_layout()
    plt.show()

    mean_profit = df["profit"].mean()
    median_profit = df["profit"].median()
    print(
        f"Interpretation: Profit distribution is centered around mean={mean_profit:.4f}, median={median_profit:.4f}. "
        "The boxplot reveals regime-wise spread differences and potential asymmetry in outcomes."
    )


def run_phase4_eda(trader_path: Path, sentiment_path: Path) -> None:
    """Execute all requested EDA analyses for phase 4."""
    _, _, merged = preprocess(trader_path=trader_path, sentiment_path=sentiment_path)
    merged = standardize_sentiment_labels(merged)

    # Keep sentiment categories relevant for requested comparison.
    merged = merged[merged["classification"].isin(["Fear", "Greed", "Unknown"])].copy()

    print("\nPHASE 4 EDA STARTED")
    print(f"Merged rows available: {len(merged):,}")

    plot_average_profit_by_sentiment(merged)
    plot_win_rate_by_sentiment(merged)
    plot_leverage_by_sentiment(merged)
    plot_buy_sell_behavior_by_sentiment(merged)
    plot_profit_distribution(merged)


if __name__ == "__main__":
    run_phase4_eda(
        trader_path=TRADER_DATA_PATH,
        sentiment_path=SENTIMENT_DATA_PATH,
    )
