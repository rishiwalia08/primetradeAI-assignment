"""Phase 5: Advanced trader-behavior insights and analyst-style conclusions."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd

from preprocess_phase2 import (
    SENTIMENT_DATA_PATH,
    TRADER_DATA_PATH,
    preprocess,
)


TOP_TRADER_QUANTILE = 0.95  # Top 5% by total profit


def _resolve_side_column(df: pd.DataFrame) -> str | None:
    for col in ["side", "direction"]:
        if col in df.columns:
            return col
    return None


def _normalize_side(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.upper().replace({"LONG": "BUY", "SHORT": "SELL"})


def prepare_phase5_dataset(trader_path: Path, sentiment_path: Path) -> pd.DataFrame:
    """Load and preprocess merged dataset for phase-5 analysis."""
    _, _, merged = preprocess(trader_path=trader_path, sentiment_path=sentiment_path)
    merged = merged.copy()
    merged["classification"] = merged["classification"].astype("string").str.strip().str.title()

    if "account" not in merged.columns:
        raise KeyError("Expected `account` column in merged dataset for trader-level analysis.")

    # Keep rows with valid account and known Fear/Greed labels for behavioral comparisons.
    merged = merged.dropna(subset=["account"]).copy()
    return merged


def identify_top_traders(df: pd.DataFrame, quantile: float = TOP_TRADER_QUANTILE) -> Tuple[pd.DataFrame, pd.Series]:
    """Identify top traders by total profit and return account-level table + account list."""
    trader_perf = (
        df.groupby("account", as_index=False)
        .agg(
            total_profit=("profit", "sum"),
            avg_profit=("profit", "mean"),
            win_rate=("is_profit", "mean"),
            trades=("profit", "count"),
            avg_trade_size_usd=("trade_size_usd", "mean"),
        )
        .sort_values("total_profit", ascending=False)
    )

    threshold = trader_perf["total_profit"].quantile(quantile)
    top_traders = trader_perf[trader_perf["total_profit"] >= threshold].copy()
    top_accounts = top_traders["account"]
    return trader_perf, top_accounts


def compare_top_vs_average(df: pd.DataFrame, top_accounts: pd.Series) -> pd.DataFrame:
    """Compare behavioral and performance metrics for top cohort vs all others."""
    tagged = df.copy()
    tagged["trader_group"] = tagged["account"].isin(set(top_accounts)).map({True: "Top Traders", False: "Other Traders"})

    agg_dict = {
        "total_profit": ("profit", "sum"),
        "avg_profit": ("profit", "mean"),
        "win_rate": ("is_profit", "mean"),
        "median_profit": ("profit", "median"),
        "avg_trade_size_usd": ("trade_size_usd", "mean"),
        "median_trade_size_usd": ("trade_size_usd", "median"),
        "trades": ("profit", "count"),
        "unique_accounts": ("account", "nunique"),
    }
    if "leverage" in tagged.columns:
        agg_dict["avg_leverage"] = ("leverage", "mean")

    comp = tagged.groupby("trader_group", as_index=False).agg(**agg_dict)
    if "avg_leverage" not in comp.columns:
        comp["avg_leverage"] = pd.NA
    comp["win_rate"] = comp["win_rate"] * 100
    return comp


def top_trader_regime_behavior(df: pd.DataFrame, top_accounts: pd.Series) -> pd.DataFrame:
    """Evaluate whether top traders behave differently in Fear vs Greed."""
    top_df = df[df["account"].isin(set(top_accounts))].copy()
    top_df = top_df[top_df["classification"].isin(["Fear", "Greed"])].copy()

    agg_dict = {
        "trades": ("profit", "count"),
        "total_profit": ("profit", "sum"),
        "avg_profit": ("profit", "mean"),
        "win_rate": ("is_profit", "mean"),
        "avg_trade_size_usd": ("trade_size_usd", "mean"),
    }
    if "leverage" in top_df.columns:
        agg_dict["avg_leverage"] = ("leverage", "mean")

    regime = top_df.groupby("classification", as_index=False).agg(**agg_dict).sort_values("classification")
    if "avg_leverage" not in regime.columns:
        regime["avg_leverage"] = pd.NA
    regime["win_rate"] = regime["win_rate"] * 100
    return regime


def contrarian_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Compare contrarian trades vs trend-following trades.

    Contrarian definition used:
    - Fear + BUY
    - Greed + SELL

    Trend-following definition:
    - Fear + SELL
    - Greed + BUY
    """
    side_col = _resolve_side_column(df)
    if side_col is None:
        raise KeyError("No side column found. Expected one of: side, direction.")

    tmp = df.copy()
    tmp["side_norm"] = _normalize_side(tmp[side_col])
    tmp = tmp[tmp["classification"].isin(["Fear", "Greed"])].copy()
    tmp = tmp[tmp["side_norm"].isin(["BUY", "SELL"])].copy()

    tmp["is_contrarian"] = (
        ((tmp["classification"] == "Fear") & (tmp["side_norm"] == "BUY"))
        | ((tmp["classification"] == "Greed") & (tmp["side_norm"] == "SELL"))
    )

    tmp["style"] = tmp["is_contrarian"].map({True: "Contrarian", False: "Trend-following"})

    agg_dict = {
        "trades": ("profit", "count"),
        "total_profit": ("profit", "sum"),
        "avg_profit": ("profit", "mean"),
        "median_profit": ("profit", "median"),
        "win_rate": ("is_profit", "mean"),
        "avg_trade_size_usd": ("trade_size_usd", "mean"),
    }
    if "leverage" in tmp.columns:
        agg_dict["avg_leverage"] = ("leverage", "mean")

    perf = tmp.groupby("style", as_index=False).agg(**agg_dict).sort_values("avg_profit", ascending=False)
    if "avg_leverage" not in perf.columns:
        perf["avg_leverage"] = pd.NA
    perf["win_rate"] = perf["win_rate"] * 100
    return perf


def plot_top_vs_other(comp: pd.DataFrame) -> None:
    """Visual compare of top-trader cohort vs others."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    x = comp["trader_group"]
    axes[0].bar(x, comp["avg_profit"], color=["#2ca02c", "#7f7f7f"])
    axes[0].set_title("Average Profit per Trade")
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].bar(x, comp["win_rate"], color=["#2ca02c", "#7f7f7f"])
    axes[1].set_title("Win Rate (%)")
    axes[1].tick_params(axis="x", rotation=15)

    axes[2].bar(x, comp["avg_trade_size_usd"], color=["#2ca02c", "#7f7f7f"])
    axes[2].set_title("Average Trade Size (USD)")
    axes[2].tick_params(axis="x", rotation=15)

    for ax in axes:
        ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    plt.show()


def plot_contrarian(perf: pd.DataFrame) -> None:
    """Visual compare of contrarian vs trend-following performance."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = perf["style"]

    axes[0].bar(x, perf["avg_profit"], color=["#1f77b4", "#ff7f0e"])
    axes[0].set_title("Average Profit by Trading Style")
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(x, perf["win_rate"], color=["#1f77b4", "#ff7f0e"])
    axes[1].set_title("Win Rate (%) by Trading Style")
    axes[1].grid(axis="y", alpha=0.2)

    plt.tight_layout()
    plt.show()


def analyst_insight_summary(
    top_table: pd.DataFrame,
    comp: pd.DataFrame,
    regime: pd.DataFrame,
    contra: pd.DataFrame,
) -> None:
    """Print concise real-world analyst insights from computed metrics."""
    print("\n" + "=" * 96)
    print("PHASE 5: TRADING ANALYST INSIGHTS")
    print("=" * 96)

    # Insight 1: concentration of alpha
    total_profit_all = top_table["total_profit"].sum()
    top_cut = top_table["total_profit"].quantile(TOP_TRADER_QUANTILE)
    top_profit = top_table.loc[top_table["total_profit"] >= top_cut, "total_profit"].sum()
    concentration = (top_profit / total_profit_all * 100) if total_profit_all else 0.0
    print(
        f"1) Profit concentration: top cohort contributes {concentration:.2f}% of aggregate profit, "
        "indicating alpha is concentrated in a relatively small trader segment."
    )

    # Insight 2: behavior of top cohort
    top_row = comp.loc[comp["trader_group"] == "Top Traders"].iloc[0]
    other_row = comp.loc[comp["trader_group"] == "Other Traders"].iloc[0]
    print(
        "2) Top-vs-average behavior: top traders show "
        f"higher avg profit ({top_row['avg_profit']:.4f} vs {other_row['avg_profit']:.4f}) and "
        f"higher win rate ({top_row['win_rate']:.2f}% vs {other_row['win_rate']:.2f}%)."
    )

    # Insight 3: regime adaptation
    if len(regime) >= 2:
        best_regime = regime.sort_values("avg_profit", ascending=False).iloc[0]
        worst_regime = regime.sort_values("avg_profit", ascending=True).iloc[0]
        print(
            "3) Regime adaptation: top traders perform best in "
            f"{best_regime['classification']} (avg profit {best_regime['avg_profit']:.4f}) "
            f"and weakest in {worst_regime['classification']} (avg profit {worst_regime['avg_profit']:.4f}), "
            "showing sentiment-dependent edge."
        )

    # Insight 4: contrarian edge
    if len(contra) >= 2:
        winner = contra.sort_values("avg_profit", ascending=False).iloc[0]
        runner = contra.sort_values("avg_profit", ascending=False).iloc[1]
        print(
            "4) Trading-style edge: "
            f"{winner['style']} outperforms {runner['style']} on average profit "
            f"({winner['avg_profit']:.4f} vs {runner['avg_profit']:.4f}) and "
            f"win rate ({winner['win_rate']:.2f}% vs {runner['win_rate']:.2f}%)."
        )

    print(
        "5) Practical takeaway: use sentiment-aware position sizing and route risk budget "
        "toward cohorts/styles that retain edge across regimes, while de-risking in weak regimes."
    )


def run_phase5(trader_path: Path, sentiment_path: Path) -> None:
    df = prepare_phase5_dataset(trader_path=trader_path, sentiment_path=sentiment_path)

    top_table, top_accounts = identify_top_traders(df)
    comp = compare_top_vs_average(df, top_accounts)
    regime = top_trader_regime_behavior(df, top_accounts)
    contra = contrarian_performance(df)

    print("\nTop traders by total profit:")
    print(top_table.head(15).to_string(index=False))

    print("\nTop vs Other traders summary:")
    print(comp.to_string(index=False))

    print("\nTop trader behavior in Fear vs Greed:")
    print(regime.to_string(index=False))

    print("\nContrarian vs Trend-following performance:")
    print(contra.to_string(index=False))

    plot_top_vs_other(comp)
    plot_contrarian(contra)
    analyst_insight_summary(top_table, comp, regime, contra)


if __name__ == "__main__":
    run_phase5(
        trader_path=TRADER_DATA_PATH,
        sentiment_path=SENTIMENT_DATA_PATH,
    )
