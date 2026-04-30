import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import copy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from sklearn.preprocessing import StandardScaler

import streamlit as st
from streamlit_option_menu import option_menu

from src.data.data_generator import load_all_data, FEATURE_NAMES, BANK_PROFILES, BANK_COLORS
from src.models.model import CreditNet, get_weights, set_weights
from src.federated.fl_engine import FLEngine
from src.federated.baseline import train_centralized_baseline, train_local_only_baselines
from src.federated.comparison import run_comparison
from src.privacy.dp_custom import compute_epsilon
from src.utils.fl_logger import FLLogger
from src.utils import plots as P
from src.ui.styles import apply_custom_styles
from src.ui.components import (
    icon,
    icon_header,
    icon_status,
    render_log_tab,
    render_export_tab
)
from src.ui.score_calculator import render_score_calculator
from src.ui.login import render_login
from src.ui.overview import render_overview
from src.ui.sidebar import render_sidebar
from src.ui.data_explorer import render_data_explorer
from src.ui.privacy_analysis import render_privacy_analysis
from src.ui.fl_training import render_fl_training
from src.ui.baseline_comparison import render_baseline_comparison
from src.ui.comparison import render_performance_comparison

st.set_page_config(
    page_title="FL Credit Scoring",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_styles()
def prob_to_cibil(prob):
    return int(900 - prob * 600)

def score_label(score):
    if score >= 750: return "Excellent", "#22c55e"
    if score >= 650: return "Good", "#84cc16"
    if score >= 550: return "Fair", "#f97316"
    return "Poor", "#ef4444"

@st.cache_data
def _load_data():
    return load_all_data()

all_data = _load_data()

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if "username" not in st.session_state:
    st.session_state["username"] = ""

if not st.session_state["authenticated"]:
    render_login()
    st.stop()

def _init_state(key, val):
    if key not in st.session_state:
        st.session_state[key] = val

_init_state("training_mode", "fedavg_dp")
_init_state("sel_banks", ["SBI", "HDFC", "Axis", "PNB", "ICICI"])
_init_state("num_rounds", 8)
_init_state("local_epochs", 2)
_init_state("lr", 0.001)
_init_state("fl_backend", "custom")
_init_state("noise_mult", 1.1)
_init_state("max_norm", 1.0)
_init_state("dp_backend", "custom")
_init_state("mu_fedprox", 0.01)
page = render_sidebar()

training_mode = st.session_state.get("training_mode", "fedavg_dp")
sel_banks     = st.session_state.get("sel_banks", ["SBI", "HDFC", "Axis", "PNB", "ICICI"])
num_rounds    = st.session_state.get("num_rounds", 8)
local_epochs  = st.session_state.get("local_epochs", 2)
lr            = st.session_state.get("lr", 0.001)
fl_backend    = st.session_state.get("fl_backend", "custom")
noise_mult    = st.session_state.get("noise_mult", 1.1)
max_norm      = st.session_state.get("max_norm", 1.0)
dp_backend    = st.session_state.get("dp_backend", "custom")
mu_fedprox    = st.session_state.get("mu_fedprox", 0.01)

use_dp = training_mode in ("fedavg_dp", "fedprox_dp")
use_fedprox = training_mode == "fedprox_dp"

if page == "Overview":
    render_overview(BANK_PROFILES)
elif page == "Data Explorer":
    render_data_explorer(
        all_data,
        BANK_PROFILES,
        FEATURE_NAMES
    )
elif page == "FL Training":
    render_fl_training(
    training_mode,
    fl_backend,
    dp_backend,
    use_dp,
    sel_banks,
    num_rounds,
    local_epochs,
    lr,
    noise_mult,
    max_norm,
    mu_fedprox,
    all_data
)

elif page == "Baseline Comparison":
    render_baseline_comparison(
        sel_banks=sel_banks,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        lr=lr,
        all_data=all_data
    )

elif page == "Performance Comparison":
    render_performance_comparison(
        sel_banks,
        all_data,
        num_rounds,
        local_epochs,
        lr,
        noise_mult,
        max_norm,
        mu_fedprox,
        fl_backend,
        dp_backend
    )

elif page == "Privacy Analysis":
    render_privacy_analysis(
        use_dp,
        noise_mult,
        local_epochs,
        num_rounds,
        icon_status
    )

elif page == "Credit Predictor":
    icon_header("fa-credit-card", "Credit Score Predictor")
    model_ready = "trained_model" in st.session_state

    if not model_ready:
        model_path = "experiments/results/best_model.pt"

        if os.path.exists(model_path):
            try:
                m = CreditNet(10)
                m.load_state_dict(torch.load(model_path, map_location="cpu"))

                st.session_state["trained_model"] = m
                st.session_state["scalers"] = {
                    "HDFC": StandardScaler().fit(
                        all_data["HDFC"][FEATURE_NAMES].values
                    )
                }

                model_ready = True
                st.success("Model loaded successfully from trained model.")

            except Exception as e:
                st.warning(f"Model loading failed: {str(e)}")

    if not model_ready:
        st.warning(
            "Train the FL model first from FL Training page. "
            "Using demo model for now."
        )

    st.divider()

    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.subheader("Customer Details")

        income = st.number_input(
            "Monthly Income (INR)",
            min_value=5000,
            max_value=300000,
            value=45000,
            step=1000
        )

        loan_amount = st.number_input(
            "Loan Amount (INR)",
            min_value=10000,
            max_value=3000000,
            value=200000,
            step=10000
        )

        loan_tenure = st.slider(
            "Loan Tenure (Months)",
            6, 120, 36
        )

        existing_loans = st.selectbox(
            "Existing Loans",
            [0, 1, 2, 3]
        )

        on_time_pct = st.slider(
            "On-time Payment %",
            0, 100, 85
        )

        credit_util = st.slider(
            "Credit Utilization %",
            0, 100, 30
        )

        employment = st.selectbox(
            "Employment Type",
            [
                "Government (Salaried)",
                "Private (Salaried)",
                "Self-Employed",
                "Business Owner",
                "Freelancer",
                "Unemployed"
            ]
        )

        age = st.slider(
            "Age",
            21, 65, 34
        )

        savings_pct = st.slider(
            "Savings % of Income",
            0, 60, 20
        )

        enquiries = st.slider(
            "Credit Enquiries (Last 6 Months)",
            0, 10, 1
        )

        predict = st.button(
            "Predict Credit Score",
            use_container_width=True,
            type="primary"
        )

    render_score_calculator(
        col_result=col_result,
        predict=predict,
        income=income,
        age=age,
        loan_amount=loan_amount,
        loan_tenure=loan_tenure,
        existing_loans=existing_loans,
        on_time_pct=on_time_pct,
        credit_util=credit_util,
        savings_pct=savings_pct,
        enquiries=enquiries,
        employment=employment,
        model_ready=model_ready,
        all_data=all_data,
        FEATURE_NAMES=FEATURE_NAMES,
        CreditNet=CreditNet,
        prob_to_cibil=prob_to_cibil,
        score_label=score_label,
        icon_status=icon_status,
        st=st,
        np=np,
        go=go,
        torch=torch,
        StandardScaler=StandardScaler
    )
