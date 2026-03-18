# src/part2_app.py
#
# Description: FastAPI service that loads a pre-trained credit risk model
#              and serves default predictions via a POST /predict endpoint.
#
# Changelog:
# 16/03/2026  Initial version from starter template
# 16/03/2026  Added customer_id to request and response
# 16/03/2026  Replaced deprecated on_event with lifespan context manager
# 16/03/2026  Added input validation with Pydantic Field constraints
# 16/03/2026  Renamed fields to match brief (num_transactions, has_rent etc.)
# 16/03/2026  Added structured logging with latency tracking per prediction
# 16/03/2026  Added PredictionResponse model to document output contract
# 16/03/2026  Improved health endpoint to report model status
# 16/03/2026  Added try/except around model prediction for clean error handling
# 18/03/2026  Aligned API schema with all Part 1 features (kw_, has_, custom)
#
# Usage: uvicorn src.app:app --host 0.0.0.0 --port 8000

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
from contextlib import asynccontextmanager
import logging
import time
import joblib

# Structured logging so every prediction leaves an audit trail.
# In production this would feed into something like Datadog or CloudWatch.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "model.joblib"

model = None

# Use lifespan context manager instead of deprecated on_event("startup").
# The model is loaded once when the app starts, not on every request.
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError("Model file not found. Please place model.joblib in artifacts/")
    model = joblib.load(MODEL_PATH)
    log.info("Model loaded from %s", MODEL_PATH)
    yield

app = FastAPI(title="ML Inference Service", lifespan=lifespan)

# Input validation: proper types and constraints so bad data gets rejected
# before it reaches the model. In production, garbage in = garbage out.
# Accepts all features produced by the Part 1 pipeline (training_set.csv).
class CustomerFeatures(BaseModel):
    # Included so predictions can be traced back to a customer.
    customer_id: str

    # Core aggregation features
    num_transactions: int = Field(0, ge=0, description="Number of transactions")
    total_debit: float = Field(0.0, le=0, description="Sum of debits (negative)")
    total_credit: float = Field(0.0, ge=0, description="Sum of credits (positive)")
    avg_amount: float = Field(0.0, description="Mean transaction amount")

    # Keyword flags (used by the current pre-trained model)
    kw_rent: int = Field(0, ge=0, le=1)
    kw_netflix: int = Field(0, ge=0, le=1)
    kw_tesco: int = Field(0, ge=0, le=1)
    kw_payroll: int = Field(0, ge=0, le=1)
    kw_bonus: int = Field(0, ge=0, le=1)

    # Merchant category flags (not used by current model, available for retraining)
    has_salary: int = Field(0, ge=0, le=1)
    has_rent: int = Field(0, ge=0, le=1)
    has_grocery: int = Field(0, ge=0, le=1)
    has_streaming: int = Field(0, ge=0, le=1)
    has_gambling: int = Field(0, ge=0, le=1)
    has_transfer: int = Field(0, ge=0, le=1)
    has_bonus: int = Field(0, ge=0, le=1)

    # Custom features (not used by current model, available for retraining)
    debit_credit_ratio: float = Field(0.0, ge=0)
    txn_amount_std: float = Field(0.0, ge=0)
    num_unique_merchants: int = Field(0, ge=0)

# Response model documents the exact output contract.
# FastAPI also uses this to generate accurate API docs automatically.
class PredictionResponse(BaseModel):
    customer_id: str
    probability: float = Field(description="Default probability between 0 and 1")
    prediction: int = Field(description="0 = no default, 1 = default")

# Health endpoint reports whether the model is loaded.
# A load balancer should only route traffic here if model_loaded is true.
@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "unavailable",
        "model_loaded": model is not None,
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    start = time.perf_counter()

    # The pre-trained model expects these 9 features in this exact order.
    # Category flags and custom features are accepted by the API but not
    # passed to the current model. They would be used after retraining.
    X = [[
        payload.num_transactions,
        payload.total_debit,
        payload.total_credit,
        payload.avg_amount,
        payload.kw_rent,
        payload.kw_netflix,
        payload.kw_tesco,
        payload.kw_payroll,
        payload.kw_bonus,
    ]]
    # Catch model errors and return a clean message.
    # The actual error is logged for debugging but not exposed to the caller.
    try:
        proba = model.predict_proba(X)[0][1]
    except Exception as e:
        log.error("Prediction failed for customer=%s: %s", payload.customer_id, e)
        raise HTTPException(status_code=500, detail="Prediction failed. Please check input data.")

    pred = int(proba >= 0.5)

    elapsed_ms = (time.perf_counter() - start) * 1000
    log.info("customer=%s prob=%.4f pred=%d latency=%.1fms",
             payload.customer_id, proba, pred, elapsed_ms)

    return PredictionResponse(
        customer_id=payload.customer_id,
        probability=round(proba, 4),
        prediction=pred,
    )
