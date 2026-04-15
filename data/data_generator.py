import numpy as np
import pandas as pd
import streamlit as st

FEATURE_NAMES = [
    "income", "age", "loan_amount", "loan_tenure",
    "existing_loans", "on_time_ratio", "credit_utilization",
    "employment_score", "savings_ratio", "num_enquiries"
]

BANK_PROFILES = {
    "SBI":   dict(income_mean=28000, income_std=12000, age_mean=42, default_rate=0.28, n=1200),
    "HDFC":  dict(income_mean=62000, income_std=22000, age_mean=34, default_rate=0.18, n=1000),
    "Axis":  dict(income_mean=48000, income_std=18000, age_mean=37, default_rate=0.22, n=900),
    "PNB":   dict(income_mean=32000, income_std=14000, age_mean=45, default_rate=0.30, n=800),
    # ── 5th Bank (NEW): ICICI — urban high-income, young professionals, low default
    "ICICI": dict(income_mean=82000, income_std=28000, age_mean=31, default_rate=0.14, n=1100),
    # ── 6th Bank (NEW): Kotak — mixed retail
    "Kotak": dict(income_mean=55000, income_std=20000, age_mean=35, default_rate=0.16, n=950),
}

def generate_bank_data(bank_name, seed=42):
    p = BANK_PROFILES[bank_name]
    rng = np.random.RandomState(seed + hash(bank_name) % 100)
    n   = p["n"]

    income              = rng.normal(p["income_mean"], p["income_std"], n).clip(5000, 200000)
    age                 = rng.normal(p["age_mean"], 8, n).clip(21, 65)
    loan_amount         = rng.exponential(150000, n).clip(10000, 2000000)
    loan_tenure         = rng.choice([12,24,36,48,60,84,120], n)
    existing_loans      = rng.choice([0,1,2,3], n, p=[0.45,0.30,0.17,0.08])
    on_time_ratio       = rng.beta(6, 2, n)
    credit_utilization  = rng.beta(2, 5, n)
    employment_score    = rng.uniform(0, 1, n)
    savings_ratio       = rng.beta(3, 5, n)
    num_enquiries       = rng.poisson(2, n).clip(0, 15)

    X = np.column_stack([
        income, age, loan_amount, loan_tenure,
        existing_loans, on_time_ratio, credit_utilization,
        employment_score, savings_ratio, num_enquiries
    ])

    # Target: default (1) or no default (0)
    log_odds = (
        -3.0
        - 0.00003 * income
        + 0.01    * age
        + 0.000001* loan_amount
        - 2.5     * on_time_ratio
        + 1.2     * credit_utilization
        + 0.4     * existing_loans
        + rng.normal(0, 0.5, n)
    )
    prob  = 1 / (1 + np.exp(-log_odds))
    prob  = prob * (p["default_rate"] / prob.mean())
    prob  = prob.clip(0.01, 0.99)
    y     = (rng.uniform(0,1,n) < prob).astype(np.float32)

    df    = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["default"] = y
    return df

@st.cache_data
def load_all_data():
    return {b: generate_bank_data(b) for b in BANK_PROFILES}
