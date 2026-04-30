import streamlit as st
from src.ui.components import icon_header
from src.utils import plots as P


def render_data_explorer(all_data, BANK_PROFILES, FEATURE_NAMES):
    icon_header(
        "fa-database",
        "Data Explorer — Non-IID Bank Data"
    )

    tab1, tab2, tab3 = st.tabs([
        "Income Distribution",
        "Feature Correlation",
        "Raw Sample"
    ])

    with tab1:
        st.plotly_chart(
            P.income_distribution(all_data),
            use_container_width=True,
            key="de_income"
        )

        c1, c2 = st.columns(2)

        with c1:
            st.plotly_chart(
                P.default_rate_bars(all_data),
                use_container_width=True,
                key="de_default"
            )

        with c2:
            st.plotly_chart(
                P.dataset_size_pie(all_data),
                use_container_width=True,
                key="de_pie"
            )

    with tab2:
        bk = st.selectbox(
            "Select Bank",
            list(BANK_PROFILES.keys()),
            key="de_corr_bank"
        )

        st.plotly_chart(
            P.correlation_heatmap(
                all_data[bk],
                bk,
                FEATURE_NAMES
            ),
            use_container_width=True,
            key="de_heatmap"
        )

    with tab3:
        bk2 = st.selectbox(
            "Select Bank",
            list(BANK_PROFILES.keys()),
            key="de_raw_bank"
        )

        st.dataframe(
            all_data[bk2].head(50),
            use_container_width=True
        )
