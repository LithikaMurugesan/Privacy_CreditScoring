import streamlit as st
from src.ui.components import icon_header


def render_overview(BANK_PROFILES):
    icon_header(
        "fa-chart-line",
        "Privacy-Preserving Credit Scoring Dashboard"
    )

    c1, c2, c3, c4 = st.columns(4)

    total = sum(
        BANK_PROFILES[b]["n"]
        for b in BANK_PROFILES
    )

    c1.markdown(
        '''
        <div class="kpi">
            <div class="kpi-val">5</div>
            <div class="kpi-lbl">Banks Federated</div>
        </div>
        ''',
        unsafe_allow_html=True
    )

    c2.markdown(
        f'''
        <div class="kpi">
            <div class="kpi-val">{total:,}</div>
            <div class="kpi-lbl">Total Samples</div>
        </div>
        ''',
        unsafe_allow_html=True
    )

    c3.markdown(
        '''
        <div class="kpi">
            <div class="kpi-val">epsilon~2.0</div>
            <div class="kpi-lbl">Privacy Budget</div>
        </div>
        ''',
        unsafe_allow_html=True
    )

    c4.markdown(
        '''
        <div class="kpi">
            <div class="kpi-val">0 bytes</div>
            <div class="kpi-lbl">Raw Data Shared</div>
        </div>
        ''',
        unsafe_allow_html=True
    )

    st.divider()

    icon_header(
        "fa-sliders",
        "Experiment Configuration",
        level=3
    )

    tab_gen, tab_fl, tab_dp = st.tabs(
        [
            "General Settings",
            "FL Settings",
            "Privacy Settings"
        ]
    )

    with tab_gen:
        col1, col2 = st.columns(2)

        with col1:
            st.selectbox(
                "Training Mode",
                options=[
                    "fedavg_dp",
                    "fedavg",
                    "fedprox_dp",
                    "centralized"
                ],
                format_func=lambda m: {
                    "centralized": "Centralized (no FL)",
                    "fedavg": "FedAvg (no DP)",
                    "fedavg_dp": "FedAvg + DP",
                    "fedprox_dp": "FedProx + DP",
                }[m],
                key="training_mode",
            )

        with col2:
            st.multiselect(
                "Participating Banks",
                list(BANK_PROFILES.keys()),
                key="sel_banks"
            )

    with tab_fl:
        fc1, fc2, fc3, fc4 = st.columns(4)

        with fc1:
            st.slider(
                "FL Rounds",
                3,
                15,
                key="num_rounds"
            )

        with fc2:
            st.slider(
                "Local Epochs",
                1,
                5,
                key="local_epochs"
            )

        with fc3:
            st.select_slider(
                "Learning Rate",
                [0.0001, 0.001, 0.005, 0.01],
                key="lr"
            )

        with fc4:
            st.selectbox(
                "FL Backend",
                options=["custom", "flower"],
                format_func=lambda b:
                    "Custom Loop"
                    if b == "custom"
                    else "Flower Framework",
                key="fl_backend"
            )

    with tab_dp:
        use_dp_tab = st.session_state[
            "training_mode"
        ] in (
            "fedavg_dp",
            "fedprox_dp"
        )

        use_fedprox_tab = (
            st.session_state["training_mode"]
            == "fedprox_dp"
        )

        pc1, pc2, pc3 = st.columns(3)

        with pc1:
            st.slider(
                "Noise Multiplier (sigma)",
                0.5,
                2.0,
                step=0.1,
                key="noise_mult",
                disabled=not use_dp_tab
            )

            st.slider(
                "Max Grad Norm (C)",
                0.5,
                2.0,
                step=0.1,
                key="max_norm",
                disabled=not use_dp_tab
            )

        with pc2:
            st.selectbox(
                "DP Backend",
                options=["custom", "opacus"],
                format_func=lambda b:
                    "Custom (Manual)"
                    if b == "custom"
                    else "Opacus PrivacyEngine",
                key="dp_backend",
                disabled=not use_dp_tab
            )

            st.slider(
                "Proximal mu (FedProx)",
                0.001,
                0.5,
                step=0.001,
                format="%.3f",
                key="mu_fedprox",
                disabled=not use_fedprox_tab
            )

        with pc3:
            st.info(
                "Differential Privacy (DP) protects "
                "gradient updates during training."
            )
