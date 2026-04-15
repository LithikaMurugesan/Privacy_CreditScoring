# Privacy-Preserving Credit Scoring
## Federated Learning + Differential Privacy on Non-IID Financial Data

> **Final Year Engineering Project** — Full Production Implementation

---

## Quick Start (3 Steps)

```bash
# Step 1 — Install dependencies
pip install -r requirements.txt

# Step 2 — Launch dashboard
streamlit run app.py

# Step 3 — Open browser at http://localhost:8501
```

---

## Project Structure

```
dev-branch/
├── src/                          # All production source code
│   ├── data/
│   │   └── data_generator.py     # Synthetic Non-IID bank data (6 banks)
│   ├── models/
│   │   └── model.py              # CreditNet — 3-layer MLP (GroupNorm, Opacus-compatible)
│   ├── federated/
│   │   ├── fl_engine.py          # Unified FL Engine (all 4 modes)
│   │   ├── flower_client.py      # Flower NumPyClient per bank
│   │   ├── flower_simulation.py  # flwr.simulation wrapper (Streamlit-compatible)
│   │   ├── strategies.py         # FedAvg + FedProx Flower strategies
│   │   ├── baseline.py           # Centralized + local-only baselines
│   │   └── comparison.py         # Run all 4 modes side-by-side
│   ├── privacy/
│   │   ├── dp_custom.py          # Manual gradient clipping + Gaussian noise
│   │   ├── dp_opacus.py          # Opacus PrivacyEngine wrapper
│   │   └── dp_manager.py         # Unified DP backend toggle
│   ├── utils/
│   │   ├── fl_logger.py          # Structured FL round logger + CSV export
│   │   ├── helpers.py            # local_train, evaluate_model, fed_avg
│   │   └── plots.py              # All Plotly chart builders
│   └── api/
│       └── server.py             # FastAPI /predict REST endpoint
├── configs/
│   └── default.yaml              # Centralized hyperparameter config
├── experiments/
│   └── results/                  # Auto-saved CSVs, plots, model weights
├── app.py                        # Streamlit dashboard (7 pages)
├── run_experiment.py             # CLI: run all 4 modes, save results
├── run_api.py                    # Launch FastAPI server
└── requirements.txt
```

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| Overview | Architecture, bank profiles, FL/DP concept explanations |
| Data Explorer | Non-IID income distributions, correlation heatmaps |
| FL Training | Live training with real-time accuracy charts |
| Baseline Comparison | FL+DP vs Centralized vs Local-only |
| Performance Comparison | All 4 modes + Privacy-Accuracy tradeoff chart |
| Privacy Analysis | Epsilon budget, DP implementation details |
| Credit Predictor | CIBIL score prediction using trained FL model |

---

## Training Modes

| Mode | FL | DP | Best For |
|------|----|----|---------| 
| Centralized | No | No | Upper bound (research baseline) |
| FedAvg | Yes | No | Measuring FL overhead |
| FedAvg + DP | Yes | Yes | Our main approach |
| FedProx + DP | Yes (FedProx) | Yes | Non-IID heterogeneous data |

**Select mode** from the sidebar dropdown in the dashboard.

---

## Federated Learning Architecture

```
┌─────────────────────────────────────────────────┐
│                FL SERVER (aggregator)            │
│         FedAvg / FedProx Strategy               │
└──────┬──────────────┬──────────────┬────────────┘
       │              │              │
  ┌────▼───┐    ┌─────▼──┐    ┌─────▼──┐
  │  SBI   │    │  HDFC  │    │  Axis  │  ... (6 banks)
  │ Client │    │ Client │    │ Client │
  │        │    │        │    │        │
  │Local   │    │Local   │    │Local   │
  │Dataset │    │Dataset │    │Dataset │
  │ (Non-  │    │ (Non-  │    │ (Non-  │
  │  IID)  │    │  IID)  │    │  IID)  │
  └────────┘    └────────┘    └────────┘
  
  RAW DATA NEVER LEAVES THE BANK
  Only model weights (floats) are transmitted
```

**One FL Round:**
1. Server broadcasts global model weights
2. Each bank trains locally for `local_epochs` epochs
3. Each bank sends updated weights back (optionally with DP noise)
4. Server aggregates using weighted FedAvg
5. Repeat for `num_rounds` rounds

---

## Differential Privacy — How It Works

DP-SGD adds privacy at the gradient level:

```
Step 1: Gradient Clipping
  total_norm = ||grad||₂
  clip_coef  = C / max(total_norm, C)
  grad = grad * clip_coef     ← bounds per-sample sensitivity to C

Step 2: Gaussian Noise
  noise ~ N(0, (σ × C / batch_size)²)
  grad  = grad + noise        ← hides individual contributions

Step 3: Privacy Accounting
  ε ≈ RDP(α=2) + log(1/δ) / (2 × RDP)
```

**Privacy budget ε:**
- `ε < 1` — Very strong (medical)
- `ε 1–3` — Strong (banking, our target)  
- `ε 3–7` — Moderate (general ML)
- `ε > 7` — Weak

**Two backends:**
- **Custom** (`dp_custom.py`): manual implementation, no extra dependency
- **Opacus** (`dp_opacus.py`): Meta's production library, exact per-sample gradients

---

## CLI Usage

```bash
# Run all 4 training modes and save comparison CSV + plot
python run_experiment.py

# Run with custom banks and rounds
python run_experiment.py --banks SBI HDFC Axis --rounds 10 --epochs 3

# Run a single mode
python run_experiment.py --mode fedavg_dp

# Use Flower framework as FL backend
python run_experiment.py --fl-backend flower

# Use Opacus for DP
python run_experiment.py --dp-backend opacus
```

Output saved to `experiments/results/`:
- `{timestamp}_comparison.csv` — accuracy, AUC, epsilon per mode
- `{timestamp}_comparison.png` — bar chart comparison
- `best_model.pt` — trained model weights (loaded by Credit Predictor)

---

## REST API

```bash
# Start the API server
python run_api.py

# Interactive docs → http://localhost:8000/docs
```

**POST /predict** — Credit score prediction:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "income": 50000,
    "age": 35,
    "loan_amount": 200000,
    "loan_tenure": 36,
    "existing_loans": 1,
    "on_time_ratio": 0.85,
    "credit_utilization": 0.3,
    "employment_score": 0.75,
    "savings_ratio": 0.2,
    "num_enquiries": 1
  }'
```

Response:
```json
{
  "cibil_score": 742,
  "default_probability": 0.263,
  "risk_label": "Good",
  "credit_decision": "Approved",
  "suggested_rate": "10-13%",
  "model_source": "trained (experiments/results/best_model.pt)"
}
```

---

## Tech Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| Neural Network | PyTorch 2.x | CreditNet MLP, DP-SGD training |
| Federated Learning | Flower (flwr) | NumPyClient, FedAvg/FedProx strategies |
| Differential Privacy | Opacus 1.4+ | PrivacyEngine, per-sample gradients |
| Dashboard | Streamlit 1.32+ | 7-page interactive UI |
| Charts | Plotly | All visualizations |
| REST API | FastAPI + Uvicorn | /predict endpoint |
| Data | scikit-learn + numpy | StandardScaler, metrics |

---

## Non-IID Data — Bank Profiles

| Bank | Income (avg) | Default Rate | Segment |
|------|-------------|--------------|---------|
| SBI  | INR 28,000 | 28% | Rural / Agricultural |
| HDFC | INR 62,000 | 18% | IT Professionals |
| Axis | INR 48,000 | 22% | Business Owners |
| PNB  | INR 32,000 | 30% | Government Employees |
| ICICI| INR 82,000 | 14% | Urban Professionals |
| Kotak| INR 55,000 | 16% | Mixed Retail |

These heterogeneous distributions create the Non-IID challenge that FedProx addresses.

---


