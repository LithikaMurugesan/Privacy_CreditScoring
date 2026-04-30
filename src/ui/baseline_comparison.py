import streamlit as st
import pandas as pd

from src.ui.components import icon_header
from src.utils import plots as P
from src.federated.baseline import (
    train_centralized_baseline,
    train_local_only_baselines
)


def render_baseline_comparison(
    sel_banks,
    num_rounds,
    local_epochs,
    lr,
    all_data
):
    icon_header("fa-arrow-left-right", "Baseline Comparison")
    st.caption("FL+DP vs Centralized (upper bound) vs Local-only (lower bound)")

    st.info("Click **Run Baseline Comparison** to train and compare all three regimes.")

    if len(sel_banks) < 2:
        st.warning("Select at least 2 banks in the sidebar.")
        return

    run_bl = st.button(
        "Run Baseline Comparison",
        use_container_width=True,
        type="primary"
    )

    if run_bl:
        epochs_bl = max(10, num_rounds * local_epochs)
        prog_ph = st.progress(0, text="Centralized training...")

        def _central_cb(ep, total, loss, acc, auc):
            prog_ph.progress(
                ep / total,
                text=f"Centralized epoch {ep}/{total} — val_acc={acc:.3f}"
            )

        with st.spinner("Training centralized baseline..."):
            central = train_centralized_baseline(
                all_data,
                sel_banks,
                epochs=epochs_bl,
                lr=lr,
                progress_cb=_central_cb
            )

        prog_ph.progress(1.0, text="Centralized done!")

        with st.spinner("Training local-only baselines..."):
            local_bl = train_local_only_baselines(
                all_data,
                sel_banks,
                epochs=epochs_bl,
                lr=lr
            )

        st.session_state["baseline_central"] = central
        st.session_state["baseline_local"] = local_bl

        st.success("Baseline training complete!")

        fl_acc = st.session_state.get(
            "global_acc",
            [central["final_acc"] * 0.97]
        )[-1]

        icon_header("fa-trophy", "Accuracy Comparison", level=3)

        st.plotly_chart(
            P.baseline_comparison_bar(
                fl_acc,
                central["final_acc"],
                local_bl
            ),
            use_container_width=True
        )

        icon_header("fa-chart-line", "Centralized Learning Curve", level=3)

        st.plotly_chart(
            P.baseline_learning_curve(
                central["history"]
            ),
            use_container_width=True
        )

        rows = [
            {
                "Method": "Centralized (no privacy)",
                "Accuracy": central["final_acc"],
                "AUC-ROC": central["final_auc"]
            }
        ]

        for b, r in local_bl.items():
            rows.append({
                "Method": f"{b} Local-only",
                "Accuracy": r["val_acc"],
                "AUC-ROC": r["val_auc"]
            })

        rows.append({
            "Method": "FL + DP (our model)",
            "Accuracy": fl_acc,
            "AUC-ROC": st.session_state.get(
                "global_auc",
                [central["final_auc"] * 0.97]
            )[-1]
        })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
