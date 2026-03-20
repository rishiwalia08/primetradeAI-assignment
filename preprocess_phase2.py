"""Phase 2 preprocessing pipeline: Hyperliquid trader data + Fear/Greed sentiment."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


WORKSPACE_DIR = Path("/Users/rishiwalia/Documents/document")
TRADER_DATA_PATH = WORKSPACE_DIR / "historical_data (1).csv"
SENTIMENT_DATA_PATH = WORKSPACE_DIR / "fear_greed_index.csv"


def load_dataset(path: Path) -> pd.DataFrame:
    """Load CSV/Parquet/JSON file into a pandas DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        return pd.read_json(path)

    raise ValueError(f"Unsupported file type: {suffix}. Use csv/parquet/json.")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all column names to snake_case for consistent downstream logic."""
    normalized = []
    for col in df.columns:
        c = re.sub(r"[^0-9a-zA-Z]+", "_", str(col).strip().lower())
        c = re.sub(r"_+", "_", c).strip("_")
        normalized.append(c)
    df.columns = normalized
    return df


def parse_trader_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Parse trader timestamp with a preference for readable local timestamp columns."""
    if "timestamp_ist" in df.columns:
        df["trade_datetime"] = pd.to_datetime(
            df["timestamp_ist"],
            format="%d-%m-%Y %H:%M",
            errors="coerce",
        )
    elif "time" in df.columns:
        df["trade_datetime"] = pd.to_datetime(df["time"], errors="coerce")
    elif "timestamp" in df.columns:
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        # Hyperliquid-like epoch timestamps are usually milliseconds.
        df["trade_datetime"] = pd.to_datetime(ts, unit="ms", errors="coerce", utc=True)
    else:
        raise KeyError("No valid trader datetime column found (expected timestamp_ist/time/timestamp).")

    df["date"] = pd.to_datetime(df["trade_datetime"], errors="coerce").dt.normalize()
    return df


def parse_sentiment_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Parse sentiment date and standardize classification values."""
    if "date" not in df.columns:
        raise KeyError("Sentiment dataset must contain a date column.")

    df["sentiment_datetime"] = pd.to_datetime(df["date"], errors="coerce")
    df["date"] = df["sentiment_datetime"].dt.normalize()

    if "classification" not in df.columns:
        raise KeyError("Sentiment dataset must contain a classification column.")

    df["classification"] = (
        df["classification"].astype("string").str.strip().str.title().replace({"": pd.NA})
    )
    return df


def coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert selected columns to numeric dtype safely."""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def apply_trader_missing_value_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in trader data with explicit business rules."""
    required = ["date", "closed_pnl", "size_tokens", "execution_price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Trader dataset missing required columns: {missing}")

    # Drop rows that cannot produce target features.
    df = df.dropna(subset=required).copy()
    return df


def apply_sentiment_missing_value_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in sentiment data."""
    df = df.dropna(subset=["date"]).copy()
    df["classification"] = df["classification"].ffill().bfill().fillna("Unknown")
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create project-required features.

    Required outputs:
    - profit = closedPnL
    - is_profit (boolean)
    - trade_size_usd = size * execution_price
    """
    df["profit"] = df["closed_pnl"]
    df["is_profit"] = df["profit"] > 0
    df["trade_size_usd"] = df["size_tokens"] * df["execution_price"]
    return df


def merge_trader_with_sentiment(
    trader_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Phase 3: merge on aligned daily date key and return merge quality metrics."""
    merged_df = trader_df.merge(
        sentiment_df[["date", "classification"]],
        on="date",
        how="left",
        validate="many_to_one",
    )

    null_sentiment_count = int(merged_df["classification"].isna().sum())
    total_rows = len(merged_df)
    null_sentiment_pct = float(null_sentiment_count / total_rows * 100) if total_rows else 0.0

    merged_df["classification"] = merged_df["classification"].fillna("Unknown")

    merge_metrics = {
        "rows_after_merge": total_rows,
        "null_sentiment_count_before_fill": null_sentiment_count,
        "null_sentiment_pct_before_fill": round(null_sentiment_pct, 4),
    }
    return merged_df, merge_metrics


def print_merge_summary(merged_df: pd.DataFrame, merge_metrics: Dict[str, float]) -> None:
    """Print merge validation summary and sample merged rows."""
    print("\n" + "=" * 90)
    print("PHASE 3 - MERGE VALIDATION")
    print("=" * 90)
    for k, v in merge_metrics.items():
        print(f"{k}: {v}")

    sample_cols = [
        c
        for c in ["account", "coin", "trade_datetime", "date", "profit", "trade_size_usd", "classification"]
        if c in merged_df.columns
    ]
    print("\nMerged sample rows:")
    print(merged_df[sample_cols].head(10))


def print_basic_info(df: pd.DataFrame, name: str) -> None:
    """Print required diagnostics: head, describe, and null counts."""
    print("\n" + "=" * 90)
    print(f"{name} - HEAD")
    print("=" * 90)
    print(df.head())

    print(f"\n{name} - DESCRIBE")
    print(df.describe(include="all").transpose())

    print(f"\n{name} - NULL COUNTS")
    print(df.isna().sum().sort_values(ascending=False))


def preprocess(trader_path: Path, sentiment_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run end-to-end preprocessing and return clean trader, clean sentiment, merged data."""
    trader_df = load_dataset(trader_path)
    sentiment_df = load_dataset(sentiment_path)

    trader_df = normalize_columns(trader_df)
    sentiment_df = normalize_columns(sentiment_df)

    trader_df = parse_trader_datetime(trader_df)
    sentiment_df = parse_sentiment_datetime(sentiment_df)

    trader_df = coerce_numeric(
        trader_df,
        ["execution_price", "size_tokens", "size_usd", "closed_pnl", "leverage", "start_position"],
    )

    trader_df = apply_trader_missing_value_rules(trader_df)
    sentiment_df = apply_sentiment_missing_value_rules(sentiment_df)

    # Keep latest sentiment label per date if duplicates exist.
    sentiment_df = (
        sentiment_df.sort_values("sentiment_datetime")
        .drop_duplicates(subset=["date"], keep="last")
        .copy()
    )

    trader_df = create_features(trader_df)

    merged_df, merge_metrics = merge_trader_with_sentiment(trader_df, sentiment_df)
    print_merge_summary(merged_df, merge_metrics)

    return trader_df, sentiment_df, merged_df


if __name__ == "__main__":
    trader_clean, sentiment_clean, merged = preprocess(
        trader_path=TRADER_DATA_PATH,
        sentiment_path=SENTIMENT_DATA_PATH,
    )

    print_basic_info(trader_clean, "TRADER (CLEAN)")
    print_basic_info(sentiment_clean, "SENTIMENT (CLEAN)")
    print_basic_info(merged, "MERGED (FINAL)")
