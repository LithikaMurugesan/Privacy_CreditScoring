import streamlit as st
import os
import torch

from src.federated.fl_engine import FLEngine
from src.utils.fl_logger import FLLogger
from src.ui.components import (
    icon_header,
    render_log_tab,
    render_export_tab
)
from src.utils import plots as P


def render_fl_training(
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
):

    icon_header("fa-cpu", "Federated Learning Training")

    mode_labels = {
        "centralized": "Centralized (no FL)",
        "fedavg": "FedAvg (no DP)",
        "fedavg_dp": "FedAvg + Differential Privacy",
        "fedprox_dp": "FedProx + Differential Privacy",
    }

    st.caption(
        f"Mode: **{mode_labels.get(training_mode, training_mode)}** | "
        f"Backend: {fl_backend.title()} | "
        f"DP: {dp_backend.title() if use_dp else 'OFF'}"
    )

    if len(sel_banks) < 2 and training_mode != "centralized":
        st.warning("Select at least 2 banks in the sidebar.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Banks", len(sel_banks))
    c2.metric("FL Rounds", num_rounds)
    c3.metric("Local Epochs", local_epochs)
    c4.metric("DP", "ON" if use_dp else "OFF")

    run_btn = st.button(
        "Start FL Training",
        use_container_width=True,
        type="primary"
    )

    if not run_btn and "fl_logger" in st.session_state:
        logger = st.session_state["fl_logger"]
        summary = logger.summary()

        st.success(
            f"Previous run — Accuracy: "
            f"**{summary.get('final_global_acc', 0):.2%}** | "
            f"epsilon: {summary.get('final_epsilon', 0):.3f}"
        )

        tab_c, tab_l, tab_e = st.tabs(["Charts", "FL Logs", "Export CSV"])

        with tab_c:
            h = st.session_state.get("bank_history", {})
            ga = st.session_state.get("global_acc", [])
            el = st.session_state.get("epsilon_log", [])

            if h and ga:
                st.plotly_chart(
                    P.fl_accuracy_chart(h, ga, sel_banks),
                    use_container_width=True,
                    key="prev_acc"
                )

                col1, col2 = st.columns(2)

                with col1:
                    loss_h = {
                        b: {"loss": h[b].get("loss", [])}
                        for b in sel_banks if b in h
                    }
                    st.plotly_chart(
                        P.fl_loss_chart(loss_h, sel_banks),
                        use_container_width=True,
                        key="prev_loss"
                    )

                with col2:
                    st.plotly_chart(
                        P.auc_vs_epsilon_chart(
                            st.session_state.get("global_auc", []),
                            el
                        ),
                        use_container_width=True,
                        key="prev_auc"
                    )

        with tab_l:
            render_log_tab(logger)

        with tab_e:
            render_export_tab(logger)

    elif run_btn:

        fl_logger = FLLogger(
            experiment_name=f"{training_mode}_{num_rounds}",
            save_dir="experiments/results"
        )

        engine = FLEngine()

        bank_history = {
            b: {"acc": [], "auc": [], "loss": []}
            for b in sel_banks
        }

        global_acc, global_auc, epsilon_log = [], [], []

        progress = st.progress(0)
        status = st.empty()
        chart = st.empty()

        def progress_cb(rnd, total, metrics):
            progress.progress(rnd / total)

            status.info(
                f"Round {rnd}/{total} | "
                f"acc={metrics.get('acc',0):.3f} | "
                f"auc={metrics.get('auc',0):.3f} | "
                f"eps={metrics.get('eps',0):.3f}"
            )

            bh = metrics.get("bank_history", {})

            for b in sel_banks:
                if b in bh:
                    bank_history[b]["acc"] = bh[b].get("acc", [])
                    bank_history[b]["auc"] = bh[b].get("auc", [])
                    bank_history[b]["loss"] = bh[b].get("loss", [])

            global_acc.append(metrics.get("acc", 0))
            global_auc.append(metrics.get("auc", 0))
            epsilon_log.append(metrics.get("eps", 0))

            chart.plotly_chart(
                P.fl_accuracy_chart(bank_history, global_acc, sel_banks),
                use_container_width=True,
                key=f"live_{rnd}"
            )

        with st.spinner("Training FL model..."):
            result = engine.run(
                mode=training_mode,
                banks=sel_banks,
                all_data=all_data,
                num_rounds=num_rounds,
                local_epochs=local_epochs,
                lr=lr,
                noise_mult=noise_mult,
                max_norm=max_norm,
                mu=mu_fedprox,
                fl_backend=fl_backend,
                dp_backend=dp_backend if use_dp else "custom",
                metrics_logger=fl_logger,
                progress_cb=progress_cb,
            )

        progress.progress(1.0)

        os.makedirs("experiments/results", exist_ok=True)
        torch.save(result["model"].state_dict(), "experiments/results/best_model.pt")

        st.session_state.update({
            "fl_logger": fl_logger,
            "bank_history": bank_history,
            "global_acc": global_acc,
            "global_auc": global_auc,
            "epsilon_log": epsilon_log,
            "sel_banks": sel_banks,
            "trained_model": result["model"],
            "last_mode": training_mode
        })

        st.success(
            f"Training complete — Accuracy: {result.get('final_acc',0):.2%} | "
            f"AUC: {result.get('final_auc',0):.3f}"
        )

        tab_c, tab_l, tab_e = st.tabs(["Charts", "FL Logs", "Export CSV"])

        with tab_c:
            st.plotly_chart(
                P.fl_accuracy_chart(bank_history, global_acc, sel_banks),
                use_container_width=True,
                key="final_acc"
            )

            col1, col2 = st.columns(2)

            with col1:
                loss_h = {
                    b: {"loss": bank_history[b].get("loss", [])}
                    for b in sel_banks
                }
                st.plotly_chart(
                    P.fl_loss_chart(loss_h, sel_banks),
                    use_container_width=True,
                    key="final_loss"
                )

            with col2:
                st.plotly_chart(
                    P.auc_vs_epsilon_chart(global_auc, epsilon_log),
                    use_container_width=True,
                    key="final_auc"
                )

        with tab_l:
            render_log_tab(fl_logger)

        with tab_e:
            render_export_tab(fl_logger)

    else:
        st.info("Configure settings and click **Start FL Training**")
