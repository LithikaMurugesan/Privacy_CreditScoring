"""
src/api/server.py
=================
FastAPI REST API for credit score prediction.

Endpoints:
  GET  /          — health check + API info
  POST /predict   — predict credit score from customer features
  GET  /model/info — model architecture info

The API loads the trained model from experiments/results/ if available,
otherwise uses a freshly initialized CreditNet (untrained, for demo only).

Running the API:
    python run_api.py
    # or: uvicorn src.api.server:app --host 0.0.0.0 --port 8000

Testing with curl:
    curl -X POST http://localhost:8000/predict \\
      -H "Content-Type: application/json" \\
      -d '{"income": 50000, "age": 35, "loan_amount": 200000,
           "loan_tenure": 36, "existing_loans": 1,
           "on_time_ratio": 0.85, "credit_utilization": 0.3,
           "employment_score": 0.75, "savings_ratio": 0.2,
           "num_enquiries": 1}'
"""

import os
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

# FastAPI — graceful fallback if not installed
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None
    BaseModel = object

from src.models.model import CreditNet
from src.data.data_generator import FEATURE_NAMES, load_all_data


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────────────────────────────────────

if FASTAPI_AVAILABLE:
    class PredictRequest(BaseModel):
        """Customer feature inputs for credit scoring."""
        income:             float = Field(..., gt=0,    description="Monthly income in INR")
        age:                float = Field(..., ge=18,   description="Age in years")
        loan_amount:        float = Field(..., gt=0,    description="Loan amount in INR")
        loan_tenure:        float = Field(..., gt=0,    description="Loan tenure in months")
        existing_loans:     int   = Field(0,  ge=0,    description="Number of existing loans")
        on_time_ratio:      float = Field(0.8, ge=0, le=1, description="On-time payment ratio [0,1]")
        credit_utilization: float = Field(0.3, ge=0, le=1, description="Credit utilization [0,1]")
        employment_score:   float = Field(0.7, ge=0, le=1, description="Employment stability [0,1]")
        savings_ratio:      float = Field(0.2, ge=0, le=1, description="Savings as fraction of income")
        num_enquiries:      int   = Field(1,  ge=0,    description="Credit enquiries last 6 months")

    class PredictResponse(BaseModel):
        """Credit scoring prediction output."""
        cibil_score:     int
        default_probability: float
        risk_label:      str
        credit_decision: str
        suggested_rate:  str
        model_source:    str


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def prob_to_cibil(prob: float) -> int:
    """Convert default probability [0,1] to CIBIL score [300,900]."""
    return int(900 - prob * 600)


def score_to_label(score: int) -> tuple:
    """Return (risk_label, credit_decision, suggested_rate) for a CIBIL score."""
    if score >= 750:
        return "Excellent", "Approved", "8-10%"
    if score >= 650:
        return "Good", "Approved", "10-13%"
    if score >= 550:
        return "Fair", "Conditional Approval", "13-18%"
    return "Poor", "Rejected", "18%+"


def load_model_and_scaler():
    """
    Load the best available trained model.

    Priority:
      1. experiments/results/best_model.pt (from run_experiment.py)
      2. Freshly initialized CreditNet (untrained — demo only)

    Returns (model, scaler, model_source)
    """
    model = CreditNet(input_dim=10)
    model_source = "untrained (demo)"

    # Try to load saved model
    model_path = os.path.join("experiments", "results", "best_model.pt")
    if os.path.exists(model_path):
        try:
            state = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state)
            model_source = f"trained ({model_path})"
        except Exception as e:
            pass  # Fall through to untrained

    # Fit scaler on HDFC data (balanced distribution)
    all_data = load_all_data()
    scaler   = StandardScaler()
    scaler.fit(all_data["HDFC"][FEATURE_NAMES].values)

    return model, scaler, model_source


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────────────────────

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="FL Credit Scoring API",
        description=(
            "Privacy-Preserving Credit Scoring using Federated Learning + "
            "Differential Privacy. Model trained across 6 Indian banks without "
            "sharing raw customer data."
        ),
        version="1.0.0",
    )

    # CORS — allow Streamlit dashboard to call this API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Load model once at startup
    _model, _scaler, _model_source = load_model_and_scaler()
    _model.eval()

    @app.get("/", tags=["Health"])
    def root():
        """API health check and basic info."""
        return {
            "status": "ok",
            "api":    "FL Credit Scoring API",
            "version": "1.0.0",
            "model_source": _model_source,
            "endpoints": ["/predict", "/model/info"],
        }

    @app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
    def predict(request: PredictRequest) -> PredictResponse:
        """
        Predict credit score for a given customer.

        The model was trained using Federated Learning across 5 banks
        with Differential Privacy — raw bank data was never centralised.

        Returns CIBIL score (300-900), default probability, and credit decision.
        """
        try:
            features = np.array([[
                request.income,
                request.age,
                request.loan_amount,
                request.loan_tenure,
                request.existing_loans,
                request.on_time_ratio,
                request.credit_utilization,
                request.employment_score,
                request.savings_ratio,
                request.num_enquiries,
            ]], dtype=np.float32)

            X_scaled = _scaler.transform(features).astype(np.float32)

            with torch.no_grad():
                prob = _model(torch.from_numpy(X_scaled)).item()

            score               = prob_to_cibil(prob)
            label, decision, rate = score_to_label(score)

            return PredictResponse(
                cibil_score          = score,
                default_probability  = round(float(prob), 4),
                risk_label           = label,
                credit_decision      = decision,
                suggested_rate       = rate,
                model_source         = _model_source,
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/model/info", tags=["Model"])
    def model_info():
        """Return model architecture details."""
        n_params = sum(p.numel() for p in _model.parameters())
        return {
            "architecture":  "CreditNet (3-layer MLP)",
            "input_dim":     10,
            "hidden_layers": [64, 32, 16],
            "output":        "sigmoid (P(default))",
            "norm_layer":    "GroupNorm (Opacus-compatible)",
            "parameters":    n_params,
            "model_source":  _model_source,
            "features":      FEATURE_NAMES,
        }

else:
    app = None  # FastAPI not available
