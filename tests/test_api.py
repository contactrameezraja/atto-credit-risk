# tests/test_api.py
#
# Basic tests for the credit risk prediction API.
# Run with: pytest tests/ -v

import pytest
from fastapi.testclient import TestClient
from src.app import app


@pytest.fixture(scope="module")
def client():
    """Create a test client with the model loaded."""
    with TestClient(app) as c:
        yield c


# Health endpoint

def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200

def test_health_shows_model_loaded(client):
    resp = client.get("/health")
    assert resp.json()["model_loaded"] is True


# Predict endpoint: valid requests

def test_predict_returns_200_with_valid_data(client):
    resp = client.post("/predict", json={
        "customer_id": "CUST_TEST",
        "num_transactions": 3,
        "total_debit": -65.98,
        "total_credit": 2500.0,
        "avg_amount": 811.34,
    })
    assert resp.status_code == 200

def test_predict_returns_required_fields(client):
    resp = client.post("/predict", json={"customer_id": "CUST_TEST"})
    data = resp.json()
    assert "customer_id" in data
    assert "probability" in data
    assert "prediction" in data

def test_predict_returns_correct_customer_id(client):
    resp = client.post("/predict", json={"customer_id": "CUST_0001"})
    assert resp.json()["customer_id"] == "CUST_0001"

def test_probability_between_0_and_1(client):
    resp = client.post("/predict", json={"customer_id": "CUST_TEST"})
    prob = resp.json()["probability"]
    assert 0.0 <= prob <= 1.0

def test_prediction_is_binary(client):
    resp = client.post("/predict", json={"customer_id": "CUST_TEST"})
    assert resp.json()["prediction"] in (0, 1)

def test_predict_accepts_all_part1_features(client):
    """The API should accept every feature that the Part 1 pipeline produces."""
    resp = client.post("/predict", json={
        "customer_id": "CUST_FULL",
        "num_transactions": 3,
        "total_debit": -65.98,
        "total_credit": 2500.0,
        "avg_amount": 811.34,
        "kw_rent": 0,
        "kw_netflix": 1,
        "kw_tesco": 1,
        "kw_payroll": 1,
        "kw_bonus": 0,
        "has_salary": 1,
        "has_rent": 0,
        "has_grocery": 1,
        "has_streaming": 1,
        "has_gambling": 0,
        "has_transfer": 0,
        "has_bonus": 0,
        "debit_credit_ratio": 0.0264,
        "txn_amount_std": 1462.48,
        "num_unique_merchants": 3,
    })
    assert resp.status_code == 200


# Predict endpoint: invalid requests

def test_missing_customer_id_returns_422(client):
    resp = client.post("/predict", json={"num_transactions": 10})
    assert resp.status_code == 422

def test_invalid_keyword_flag_returns_422(client):
    """kw_rent must be 0 or 1, not 5."""
    resp = client.post("/predict", json={"customer_id": "BAD", "kw_rent": 5})
    assert resp.status_code == 422

def test_negative_total_credit_returns_422(client):
    """total_credit can't be negative."""
    resp = client.post("/predict", json={"customer_id": "BAD", "total_credit": -100})
    assert resp.status_code == 422
