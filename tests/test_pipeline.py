# tests/test_pipeline.py
#
# Tests for the data pipeline functions in prepare_data.py.
# Run with: pytest tests/ -v

import pandas as pd
import numpy as np
from src.prepare_data import clean_text, clean_transactions, build_features


def test_clean_text_lowercases():
    assert clean_text("TESCO STORE 123") == "tesco store"

def test_clean_text_strips_whitespace():
    assert clean_text("  SALARY PAYMENT  ") == "salary payment"

def test_clean_text_handles_empty():
    assert clean_text("") == ""


def test_clean_transactions_drops_duplicates():
    """Duplicate transaction IDs should be removed, keeping the first."""
    df = pd.DataFrame({
        "transaction_id": ["T1", "T1", "T2"],
        "customer_id": ["C1", "C1", "C1"],
        "amount": [100.0, 100.0, 200.0],
        "description": ["test", "test", "test"],
    })
    result = clean_transactions(df)
    assert len(result) == 2


def test_clean_transactions_coerces_bad_amounts():
    """Non-numeric amounts should be dropped."""
    df = pd.DataFrame({
        "transaction_id": ["T1", "T2"],
        "customer_id": ["C1", "C1"],
        "amount": ["100.0", "bad"],
        "description": ["test", "test"],
    })
    result = clean_transactions(df)
    assert len(result) == 1
    assert result.iloc[0]["amount"] == 100.0


def test_build_features_returns_expected_columns():
    """Feature matrix should contain all mandatory and custom columns."""
    df = pd.DataFrame({
        "transaction_id": ["T1", "T2", "T3"],
        "customer_id": ["C1", "C1", "C2"],
        "amount": [-50.0, 1000.0, -200.0],
        "description": ["TESCO STORE", "SALARY PAYMENT", "RENT FEB"],
    })
    result = build_features(df)

    # Mandatory features
    assert "txn_count" in result.columns
    assert "total_debit" in result.columns
    assert "total_credit" in result.columns
    assert "avg_amount" in result.columns

    # Custom features
    assert "debit_credit_ratio" in result.columns
    assert "txn_amount_std" in result.columns
    assert "num_unique_merchants" in result.columns

    # Category flags
    assert "has_grocery" in result.columns
    assert "has_salary" in result.columns
    assert "has_rent" in result.columns


def test_build_features_one_row_per_customer():
    """Should aggregate to one row per customer."""
    df = pd.DataFrame({
        "transaction_id": ["T1", "T2", "T3", "T4"],
        "customer_id": ["C1", "C1", "C2", "C2"],
        "amount": [-50.0, 1000.0, -200.0, 500.0],
        "description": ["TESCO", "SALARY", "RENT", "SALARY"],
    })
    result = build_features(df)
    assert len(result) == 2
