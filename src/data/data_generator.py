"""
src/data/data_generator.py
==========================
Synthetic bank data generator for FL credit-scoring simulation.

Each bank has a distinct Non-IID profile:
  - Different income distributions
  - Different age distributions
  - Different default rates

This heterogeneity (Non-IID) is the core challenge for Federated Learning:
  local models training on these skewed distributions tend to drift apart,
  which is why FedProx (with its proximal regulariser) outperforms vanilla FedAvg.

This module is intentionally decoupled from Streamlit — it can be imported
by any training script, CLI tool, or the Flower client.
"""

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Feature names (order must match CreditNet input_dim=10)
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "income",           # monthly income in INR
    "age",              # age in years
    "loan_amount",      # requested loan amount in INR
    "loan_tenure",      # loan duration in months
    "existing_loans",   # number of currently active loans
    "on_time_ratio",    # fraction of past payments made on time [0,1]
    "credit_utilization",  # credit utilisation ratio [0,1]
    "employment_score", # employment stability score [0,1]
    "savings_ratio",    # savings as fraction of monthly income [0,1]
    "num_enquiries",    # number of credit enquiries in last 6 months
]

# ─────────────────────────────────────────────────────────────────────────────
# Bank profiles — 6 Indian banks with distinct customer segments
# These create the Non-IID heterogeneity in the federated setting.
# ─────────────────────────────────────────────────────────────────────────────
BANK_PROFILES = {
    # Rural / agricultural — lower income, higher default
    "SBI":   dict(income_mean=28000, income_std=12000, age_mean=42, default_rate=0.28, n=1200),
    # Urban IT professionals — high income, lower default
    "HDFC":  dict(income_mean=62000, income_std=22000, age_mean=34, default_rate=0.18, n=1000),
    # Business owners / SME — medium income, medium default
    "Axis":  dict(income_mean=48000, income_std=18000, age_mean=37, default_rate=0.22, n=900),
    # Government employees — lower income but stable, highest default
    "PNB":   dict(income_mean=32000, income_std=14000, age_mean=45, default_rate=0.30, n=800),
    # Urban young professionals — highest income, lowest default
    "ICICI": dict(income_mean=82000, income_std=28000, age_mean=31, default_rate=0.14, n=1100),
    # Mixed retail — balanced
    "Kotak": dict(income_mean=55000, income_std=20000, age_mean=35, default_rate=0.16, n=950),
}

# ─────────────────────────────────────────────────────────────────────────────
# Bank display colours (used across dashboard and plots)
# ─────────────────────────────────────────────────────────────────────────────
BANK_COLORS = {
    "SBI":   "#f97316",
    "HDFC":  "#3b82f6",
    "Axis":  "#22c55e",
    "PNB":   "#a855f7",
    "ICICI": "#ec4899",
    "Kotak": "#eab308",
}


def generate_bank_data(bank_name: str, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic financial data for a single bank.

    The default/no-default label is derived from a logistic model of income,
    payment history, credit utilisation, and existing debt — calibrated to
    match each bank's target default_rate.

    Parameters
    ----------
    bank_name : str  — must be a key in BANK_PROFILES
    seed      : int  — random seed for reproducibility

    Returns
    -------
    pd.DataFrame with columns = FEATURE_NAMES + ["default"]
    """
    p   = BANK_PROFILES[bank_name]
    rng = np.random.RandomState(seed + hash(bank_name) % 100)
    n   = p["n"]

    # ── Feature generation ────────────────────────────────────────────────
    income             = rng.normal(p["income_mean"], p["income_std"], n).clip(5000, 200000)
    age                = rng.normal(p["age_mean"], 8, n).clip(21, 65)
    loan_amount        = rng.exponential(150000, n).clip(10000, 2000000)
    loan_tenure        = rng.choice([12, 24, 36, 48, 60, 84, 120], n)
    existing_loans     = rng.choice([0, 1, 2, 3], n, p=[0.45, 0.30, 0.17, 0.08])
    on_time_ratio      = rng.beta(6, 2, n)          # right-skewed — most pay on time
    credit_utilization = rng.beta(2, 5, n)           # left-skewed — most don't max credit
    employment_score   = rng.uniform(0, 1, n)
    savings_ratio      = rng.beta(3, 5, n)
    num_enquiries      = rng.poisson(2, n).clip(0, 15)

    X = np.column_stack([
        income, age, loan_amount, loan_tenure,
        existing_loans, on_time_ratio, credit_utilization,
        employment_score, savings_ratio, num_enquiries,
    ])

    # ── Label generation (logistic model calibrated to bank default rate) ─
    log_odds = (
        -3.0
        - 0.00003 * income
        + 0.01    * age
        + 0.000001 * loan_amount
        - 2.5     * on_time_ratio
        + 1.2     * credit_utilization
        + 0.4     * existing_loans
        + rng.normal(0, 0.5, n)   # noise
    )
    prob = 1 / (1 + np.exp(-log_odds))
    # Re-calibrate to match the bank's target default rate
    prob = prob * (p["default_rate"] / prob.mean())
    prob = prob.clip(0.01, 0.99)
    y    = (rng.uniform(0, 1, n) < prob).astype(np.float32)

    df              = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["default"]   = y
    return df


def load_all_data(seed: int = 42) -> dict:
    """
    Load synthetic data for all 6 banks.

    Returns
    -------
    dict[bank_name -> pd.DataFrame]  (cached by Streamlit when imported via app.py)
    """
    return {b: generate_bank_data(b, seed=seed) for b in BANK_PROFILES}
