import streamlit as st
import pandas as pd

from src.ui.components import icon_header, icon_status
from src.utils import plots as P
from src.privacy.dp_custom import compute_epsilon


def render_privacy_analysis(
    use_dp,
    noise_mult,
    local_epochs,
    num_rounds,
    icon_status_func,
):
    icon_header(
        "fa-shield-halved",
        "Differential Privacy Analysis"
    )

    cur_eps = (
        compute_epsilon(
            noise_mult,
            64 / 1200,
            local_epochs * 18 * num_rounds
        )
        if use_dp else 99
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("epsilon vs Accuracy Tradeoff")

        st.plotly_chart(
            P.epsilon_vs_accuracy_curve(
                cur_eps if use_dp else None
            ),
            use_container_width=True,
            key="pa_eps_acc"
        )

    with col2:
        st.subheader(
            "Privacy Budget Consumption per Round"
        )

        if "epsilon_log" in st.session_state:
            st.plotly_chart(
                P.epsilon_budget_bars(
                    st.session_state["epsilon_log"],
                    cur_eps
                ),
                use_container_width=True,
                key="pa_eps_budget",
            )
        else:
            st.info(
                "Run FL Training first to see "
                "the live budget chart."
            )

    st.divider()

    icon_header(
        "fa-magnifying-glass",
        "DP Implementation (DP-SGD)",
        level=3
    )

    tab_custom, tab_opacus = st.tabs(
        ["Custom DP", "Opacus DP"]
    )

    with tab_custom:
        st.info("Custom DP-SGD implementation details.")

    with tab_opacus:
        st.info("Opacus Privacy Engine integration details.")

    st.divider()

    st.dataframe(
        pd.DataFrame({
            "epsilon range": [
                "epsilon < 1",
                "epsilon = 1-3",
                "epsilon = 3-7",
                "epsilon > 7"
            ],
            "Privacy Level": [
                "Very Strong",
                "Strong",
                "Moderate",
                "Weak"
            ],
            "Accuracy Drop": [
                "~15%",
                "~5-10%",
                "~2-5%",
                "<2%"
            ],
            "Use Case": [
                "Medical/Legal",
                "Banking",
                "General ML",
                "Public data"
            ],
        }),
        use_container_width=True,
        hide_index=True
    )

    if use_dp:
        if cur_eps < 3:
            icon_status_func(
                "fa-circle-check",
                f"epsilon~{cur_eps:.3f} — Strong Privacy",
                "#22c55e",
                "#052e16"
            )

        elif cur_eps < 7:
            icon_status_func(
                "fa-triangle-exclamation",
                f"epsilon~{cur_eps:.3f} — Moderate Privacy",
                "#f97316",
                "#1c0a00"
            )

        else:
            icon_status_func(
                "fa-circle-xmark",
                f"epsilon~{cur_eps:.3f} — Weak Privacy",
                "#ef4444",
                "#1a0000"
            )

    else:
        icon_status_func(
            "fa-circle-xmark",
            "DP is OFF — gradients are not protected.",
            "#ef4444",
            "#1a0000"
        )
