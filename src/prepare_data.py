# src/part1_prepare_data.py
#
# Description: Loads raw transaction and label CSVs, cleans the data,
#              engineers features, and outputs a model-ready training set.
#
# Changelog:
# 15/03/2026  Initial version from starter template
# 15/03/2026  Added null/non-numeric handling on amount column
# 15/03/2026  Added duplicate transaction_id detection and removal
# 15/03/2026  Added outlier detection on amounts using IQR method
# 15/03/2026  Added data quality report with nulls, coverage, and default rate
# 15/03/2026  Added 3 custom features: debit_credit_ratio, txn_amount_std, num_unique_merchants
# 15/03/2026  Added merchant category flags (grocery, gambling, streaming etc.)
# 15/03/2026  Removed intermediate all_desc column from final output
# 15/03/2026  Renamed defaulted_within_90d to defaulted to match brief
# 15/03/2026  Renamed txn_count to num_transactions to match brief
# 18/03/2026  Refactored into importable functions for testability and reuse

from pathlib import Path
import pandas as pd
import numpy as np
import re

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Merchant categories grouped by financial signal.
# Adding a new merchant is a config change, not a code change.
MERCHANT_CATEGORIES = {
    "salary":    ["payroll", "salary", "wages"],
    "rent":      ["rent"],
    "grocery":   ["tesco", "sainsbury", "asda", "aldi", "lidl", "morrisons", "waitrose"],
    "streaming": ["netflix", "spotify", "disney", "amazon prime", "apple tv"],
    "gambling":  ["bet365", "william hill", "paddy power", "betfair", "ladbrokes"],
    "transfer":  ["transfer", "faster payment", "standing order"],
    "bonus":     ["bonus"],
}


def clean_text(s: str) -> str:
    """Lowercase, remove non-alpha characters, collapse whitespace."""
    s = s.lower()
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def explore_data(tx: pd.DataFrame, labels: pd.DataFrame) -> None:
    """Print a data quality report covering nulls, coverage, outliers, and default rate."""
    print("\n========== DATA QUALITY REPORT ==========")
    print(f"Transactions: {tx.shape[0]} rows, {tx.shape[1]} columns")
    print(f"Labels:       {labels.shape[0]} rows, {labels.shape[1]} columns")

    # Check nulls across all columns
    null_counts = tx.isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls > 0:
        print(f"\nNull values found:")
        for col, count in null_counts[null_counts > 0].items():
            print(f"  {col}: {count} nulls ({count/len(tx):.1%})")
    else:
        print(f"\nNull values: none across any column ✓")

    # Check customer coverage between the two files
    tx_custs = set(tx["customer_id"].unique())
    label_custs = set(labels["customer_id"].unique())
    in_tx_not_labels = tx_custs - label_custs
    in_labels_not_tx = label_custs - tx_custs
    if in_tx_not_labels:
        print(f"WARNING: {len(in_tx_not_labels)} customer(s) in transactions have no label")
    if in_labels_not_tx:
        print(f"WARNING: {len(in_labels_not_tx)} customer(s) in labels have no transactions")
    if not in_tx_not_labels and not in_labels_not_tx:
        print(f"Customer coverage: all {len(tx_custs)} customers have matching labels ✓")

    # Transactions per customer (flag anyone with very few)
    txn_per_cust = tx.groupby("customer_id").size()
    thin = txn_per_cust[txn_per_cust < 3]
    if len(thin) > 0:
        print(f"WARNING: {len(thin)} customer(s) have fewer than 3 transactions: {list(thin.index)}")
        print(f"  Features for these customers may be unreliable")

    # Default rate
    default_rate = labels.iloc[:, -1].mean()
    print(f"Default rate: {default_rate:.1%}")
    print("==========================================\n")


def clean_transactions(tx: pd.DataFrame) -> pd.DataFrame:
    """Handle nulls, duplicates, and flag outliers. Returns cleaned DataFrame."""
    df = tx.copy()

    # Handle null / non-numeric amounts
    # Coerce to numeric so any blanks or text become NaN, then drop them.
    # In production data, amounts can be empty, malformed, or contain currency symbols.
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    n_bad_amounts = df["amount"].isna().sum()
    if n_bad_amounts > 0:
        print(f"WARNING: {n_bad_amounts} rows with null/invalid amounts - dropping them")
        df = df.dropna(subset=["amount"])
    else:
        print("Amounts: no nulls or invalid values found ✓")

    # Check for and remove duplicate transactions
    # Duplicate transaction IDs can occur from feed retries or double imports.
    # Keeping the first occurrence, dropping the rest.
    n_dupes = df["transaction_id"].duplicated(keep="first").sum()
    if n_dupes > 0:
        print(f"WARNING: {n_dupes} duplicate transaction_id(s) found - removing them")
        df = df.drop_duplicates(subset=["transaction_id"], keep="first")
    else:
        print("Duplicates: no duplicate transaction IDs found ✓")

    # Flag amount outliers using the IQR method
    # We log them but don't remove them because in financial data, large values
    # (e.g. salary, rent) are legitimate. Removing them would destroy real signal.
    q1 = df["amount"].quantile(0.25)
    q3 = df["amount"].quantile(0.75)
    iqr = q3 - q1
    outlier_mask = (df["amount"] < q1 - 1.5 * iqr) | (df["amount"] > q3 + 1.5 * iqr)
    n_outliers = outlier_mask.sum()
    if n_outliers > 0:
        print(f"NOTE: {n_outliers} amount outlier(s) detected (1.5x IQR method) - kept in dataset")
    else:
        print("Outliers: no amount outliers detected ✓")

    return df


def build_features(tx: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transactions into one row per customer with all features."""
    tx["clean_desc"] = tx["description"].fillna("").apply(clean_text)

    agg = (
        tx.groupby("customer_id")
        .agg(
            txn_count=("transaction_id", "count"),
            total_debit=("amount", lambda x: x[x < 0].sum()),
            total_credit=("amount", lambda x: x[x > 0].sum()),
            avg_amount=("amount", "mean"),
            all_desc=("clean_desc", lambda x: " ".join(x)),
        )
        .reset_index()
    )

    # Keyword flags: required by the pre-trained model
    keywords = ["rent", "netflix", "tesco", "payroll", "bonus"]
    for kw in keywords:
        agg[f"kw_{kw}"] = agg["all_desc"].str.contains(rf"\b{kw}\b").astype(int)

    # Merchant category flags
    # The brief asks us to flag common merchant categories, not just individual
    # keywords. Grouping merchants into categories is more maintainable and
    # captures broader spending patterns.
    for category, terms in MERCHANT_CATEGORIES.items():
        pattern = "|".join(terms)
        agg[f"has_{category}"] = agg["all_desc"].str.contains(pattern).astype(int)

    # Custom features (1-3 of our own design)
    # These go beyond the mandatory four and capture deeper credit risk signals.

    # Debit to credit ratio: how much of their income are they spending?
    # A ratio above 1.0 means they're spending more than they earn, which is
    # a strong indicator of potential default. We handle the edge case where
    # total_credit is 0 (no income) by filling with NaN to avoid division errors.
    agg["debit_credit_ratio"] = (
        agg["total_debit"].abs() / agg["total_credit"].replace(0, float("nan"))
    ).round(4)

    # Transaction amount standard deviation: spending volatility.
    # Customers with erratic amounts (big swings between transactions) tend
    # to have less stable finances. Customers with only 1 transaction get 0.
    txn_std = tx.groupby("customer_id")["amount"].std().fillna(0).reset_index()
    txn_std.columns = ["customer_id", "txn_amount_std"]
    txn_std["txn_amount_std"] = txn_std["txn_amount_std"].round(2)
    agg = agg.merge(txn_std, on="customer_id", how="left")

    # Number of unique merchants: spending diversity.
    # Customers who only transact with 1-2 merchants may have limited income
    # sources or very constrained finances.
    n_merchants = tx.groupby("customer_id")["clean_desc"].nunique().reset_index()
    n_merchants.columns = ["customer_id", "num_unique_merchants"]
    agg = agg.merge(n_merchants, on="customer_id", how="left")

    # Drop the all_desc column
    # It was only needed for keyword/category matching above.
    agg = agg.drop(columns=["all_desc"])

    return agg


def build_training_set(tx_path: Path, labels_path: Path) -> pd.DataFrame:
    """End-to-end pipeline: load, explore, clean, build features, merge with labels."""
    tx = pd.read_csv(tx_path, parse_dates=["txn_timestamp"])
    labels = pd.read_csv(labels_path)

    # Explore
    explore_data(tx, labels)

    # Clean
    tx = clean_transactions(tx)

    # Features
    features = build_features(tx)

    # Merge with labels
    df = features.merge(labels, on="customer_id", how="left")

    # Rename target column to match the brief's expected output
    df = df.rename(columns={"defaulted_within_90d": "defaulted"})

    # Rename columns to match the brief's expected format
    # The kw_ columns are kept for backward compatibility with the pre-trained
    # model (it expects exactly those 9 features). If we retrained the model,
    # we would drop kw_ and use the has_ category columns instead.
    df = df.rename(columns={"txn_count": "num_transactions"})

    return df


if __name__ == "__main__":
    df = build_training_set(
        DATA_DIR / "transactions.csv",
        DATA_DIR / "labels.csv",
    )
    df.to_csv(ARTIFACTS_DIR / "training_set.csv", index=False)
    print("wrote artifacts/training_set.csv")
