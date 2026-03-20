"""Phase 6: Simple predictive model + trader clustering (meaningful baseline approach)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocess_phase2 import (
    SENTIMENT_DATA_PATH,
    TRADER_DATA_PATH,
    preprocess,
)


RANDOM_STATE = 42


def load_merged_dataset(trader_path: Path, sentiment_path: Path) -> pd.DataFrame:
    """Return merged dataset from Phase 2 pipeline."""
    _, _, merged = preprocess(trader_path=trader_path, sentiment_path=sentiment_path)
    merged = merged.copy()
    merged["classification"] = merged["classification"].astype("string").str.strip().str.title().fillna("Unknown")
    return merged


# -----------------------------------------------------------------------------
# Option A: Predict whether a trade is profitable
# -----------------------------------------------------------------------------
def option_a_predict_profitability(df: pd.DataFrame) -> Tuple[Pipeline, Dict[str, float], pd.DataFrame]:
    """Train a baseline classifier to predict `is_profit`."""

    feature_cols_num = [
        c
        for c in ["leverage", "trade_size_usd", "size_tokens", "execution_price", "start_position", "fee"]
        if c in df.columns
    ]
    feature_cols_cat = [c for c in ["classification", "side", "direction", "coin"] if c in df.columns]

    if "is_profit" not in df.columns:
        raise KeyError("`is_profit` column is required for Option A.")

    model_df = df[feature_cols_num + feature_cols_cat + ["is_profit"]].copy()
    model_df["target"] = model_df["is_profit"].astype(int)

    X = model_df[feature_cols_num + feature_cols_cat]
    y = model_df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, feature_cols_num),
            ("cat", categorical_pipe, feature_cols_cat),
        ]
    )

    clf = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }

    # Feature importance proxy using logistic coefficients.
    prep = clf.named_steps["prep"]
    model = clf.named_steps["model"]
    feature_names = prep.get_feature_names_out()
    coefs = model.coef_[0]

    importance = (
        pd.DataFrame({"feature": feature_names, "coefficient": coefs})
        .assign(abs_coefficient=lambda d: d["coefficient"].abs())
        .sort_values("abs_coefficient", ascending=False)
        .reset_index(drop=True)
    )

    print("\n" + "=" * 96)
    print("PHASE 6 - OPTION A (Prediction): Profitability Classifier")
    print("=" * 96)
    print("Selected features:")
    print(f"- Numeric: {feature_cols_num}")
    print(f"- Categorical: {feature_cols_cat}")
    print("\nModel choice: Logistic Regression")
    print("Reason: simple, interpretable baseline for binary outcomes and mixed tabular features.")

    print("\nEvaluation metrics:")
    for k, v in metrics.items():
        print(f"- {k}: {v:.4f}")

    print("\nTop coefficient-based drivers (absolute effect):")
    print(importance.head(12).to_string(index=False))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return clf, metrics, importance


# -----------------------------------------------------------------------------
# Option B: Cluster traders by behavior
# -----------------------------------------------------------------------------
def _label_cluster_profiles(centroids_df: pd.DataFrame) -> Dict[int, str]:
    """Assign human-readable profile names to KMeans clusters."""
    labels: Dict[int, str] = {}

    # High leverage + large size => risk-takers
    risk_cluster = centroids_df.sort_values(["avg_leverage", "avg_trade_size_usd"], ascending=False).index[0]
    labels[int(risk_cluster)] = "Risk-takers"

    # Low leverage + small size => conservative
    conservative_candidates = centroids_df.drop(index=risk_cluster)
    conservative_cluster = conservative_candidates.sort_values(
        ["avg_leverage", "avg_trade_size_usd"], ascending=True
    ).index[0]
    labels[int(conservative_cluster)] = "Conservative"

    # Remaining => consistent
    for idx in centroids_df.index:
        if int(idx) not in labels:
            labels[int(idx)] = "Consistent"

    return labels


def option_b_cluster_traders(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster traders into behavior groups."""
    if "account" not in df.columns:
        raise KeyError("`account` column is required for clustering.")

    agg_dict = {
        "total_profit": ("profit", "sum"),
        "avg_profit": ("profit", "mean"),
        "win_rate": ("is_profit", "mean"),
        "avg_trade_size_usd": ("trade_size_usd", "mean"),
        "profit_std": ("profit", "std"),
        "trades": ("profit", "count"),
    }
    if "leverage" in df.columns:
        agg_dict["avg_leverage"] = ("leverage", "mean")

    trader_agg = df.groupby("account", as_index=False).agg(**agg_dict)
    if "avg_leverage" not in trader_agg.columns:
        trader_agg["avg_leverage"] = 0.0
    trader_agg = trader_agg.fillna(0.0)

    # Keep moderately active traders for more stable behavior signatures.
    trader_agg = trader_agg[trader_agg["trades"] >= 5].copy()

    cluster_features = [
        "avg_profit",
        "win_rate",
        "avg_trade_size_usd",
        "avg_leverage",
        "profit_std",
        "trades",
    ]

    X = trader_agg[cluster_features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    trader_agg["cluster"] = kmeans.fit_predict(X_scaled)

    centroids_scaled = pd.DataFrame(kmeans.cluster_centers_, columns=cluster_features)
    centroids = pd.DataFrame(scaler.inverse_transform(centroids_scaled), columns=cluster_features)

    profile_map = _label_cluster_profiles(centroids)
    trader_agg["profile"] = trader_agg["cluster"].map(profile_map)

    profile_summary = (
        trader_agg.groupby("profile", as_index=False)
        .agg(
            traders=("account", "nunique"),
            avg_profit=("avg_profit", "mean"),
            avg_win_rate=("win_rate", "mean"),
            avg_trade_size_usd=("avg_trade_size_usd", "mean"),
            avg_leverage=("avg_leverage", "mean"),
            avg_trades=("trades", "mean"),
        )
        .sort_values("avg_profit", ascending=False)
    )

    print("\n" + "=" * 96)
    print("PHASE 6 - OPTION B (Clustering): Trader Profiles")
    print("=" * 96)
    print("Selected clustering features:")
    print(f"- {cluster_features}")
    print("\nModel choice: KMeans (k=3)")
    print("Reason: simple, interpretable segmentation for behavioral cohorts.")

    print("\nCluster profile summary:")
    print(profile_summary.to_string(index=False))

    # Visualization
    plt.figure(figsize=(8, 5))
    for profile_name, grp in trader_agg.groupby("profile"):
        plt.scatter(
            grp["avg_leverage"],
            grp["win_rate"] * 100,
            alpha=0.65,
            label=profile_name,
        )

    plt.title("Trader Clusters: Leverage vs Win Rate")
    plt.xlabel("Average Leverage")
    plt.ylabel("Win Rate (%)")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return trader_agg, profile_summary


def print_key_findings(metrics: Dict[str, float], importance: pd.DataFrame, profile_summary: pd.DataFrame) -> None:
    """Concise analyst-style interpretation of model and clustering output."""
    print("\n" + "=" * 96)
    print("KEY FINDINGS (Simple but Meaningful)")
    print("=" * 96)

    print(
        "1) Predictive signal exists: baseline classifier achieves "
        f"accuracy={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}, ROC-AUC={metrics['roc_auc']:.3f}, "
        "which indicates trade outcome is partially explainable using sentiment + risk/execution features."
    )

    top_features = importance.head(5)["feature"].tolist()
    print(
        "2) Most influential drivers include: "
        f"{top_features}. These variables should be prioritized in risk controls and strategy tuning."
    )

    best_profile = profile_summary.sort_values("avg_profit", ascending=False).iloc[0]
    print(
        "3) Segmentation insight: highest-performing cohort is "
        f"{best_profile['profile']} with avg_profit={best_profile['avg_profit']:.4f}, "
        f"avg_win_rate={best_profile['avg_win_rate'] * 100:.2f}% and avg_leverage={best_profile['avg_leverage']:.2f}."
    )

    print(
        "4) Practical use: combine model probabilities with cluster profile tags to set dynamic position sizing "
        "(e.g., higher confidence thresholds for high-risk clusters)."
    )


def run_phase6(trader_path: Path, sentiment_path: Path) -> None:
    """Run both options (A + B) for a complete and interview-ready Phase 6."""
    df = load_merged_dataset(trader_path=trader_path, sentiment_path=sentiment_path)

    model, metrics, importance = option_a_predict_profitability(df)
    _ = model  # keep variable for clarity in pipeline output

    _, profile_summary = option_b_cluster_traders(df)

    print_key_findings(metrics, importance, profile_summary)


if __name__ == "__main__":
    run_phase6(
        trader_path=TRADER_DATA_PATH,
        sentiment_path=SENTIMENT_DATA_PATH,
    )
