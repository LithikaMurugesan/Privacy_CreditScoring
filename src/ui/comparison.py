import streamlit as st
import pandas as pd
from src.federated.comparison import run_comparison
from src.utils import plots as P
from src.ui.components import icon_header


def render_performance_comparison(
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
):

    icon_header("fa-trophy", "Performance Comparison — All 4 Modes")
    st.caption("Centralized vs FedAvg vs FedAvg+DP vs FedProx+DP")

    st.markdown(
        """
        <style>
        div.stButton > button[kind="primary"] {
            background-color: #ef4444;
            color: white;
            border: none;
        }

        div.stButton > button[kind="primary"]:hover {
            background-color: #dc2626;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    if len(sel_banks) < 2:
        st.warning("Select at least 2 banks in the sidebar.")
        return

    rows = st.session_state.get("comparison_rows")

    if st.button("Run Performance Comparison", type="primary", use_container_width=True):

        with st.spinner("Running comparison..."):
            rows = run_comparison(
                banks=sel_banks,
                all_data=all_data,
                num_rounds=num_rounds,
                local_epochs=local_epochs,
                lr=lr,
                noise_mult=noise_mult,
                max_norm=max_norm,
                mu=mu_fedprox,
                fl_backend=fl_backend,
                dp_backend=dp_backend,
            )

        st.session_state["comparison_rows"] = rows
        st.success("Comparison completed!")

    rows = st.session_state.get("comparison_rows")

    if not rows:
        st.info("Run comparison to see results")
        return

    df = pd.DataFrame(rows)

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            P.mode_comparison_bar(df.to_dict("records"), "Accuracy"),
            use_container_width=True
        )

    with col2:
        st.plotly_chart(
            P.mode_comparison_bar(df.to_dict("records"), "AUC-ROC"),
            use_container_width=True
        )

    st.divider()

    st.plotly_chart(
        P.privacy_tradeoff_scatter(df.to_dict("records")),
        use_container_width=True
    )

    st.divider()

    st.dataframe(df, use_container_width=True)
