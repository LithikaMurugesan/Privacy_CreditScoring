
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

BANK_COLORS = {
    "SBI":  "#f97316",
    "HDFC": "#3b82f6",
    "Axis": "#22c55e",
    "PNB":  "#a855f7",
}

_DARK = "plotly_dark"


def fl_accuracy_chart(history: dict, global_acc: list, sel_banks: list) -> go.Figure:
    """Live accuracy chart — local per-bank (dashed) + global (solid white)."""
    x = list(range(1, len(global_acc) + 1))
    fig = go.Figure()
    for b in sel_banks:
        if history.get(b, {}).get("acc"):
            fig.add_trace(go.Scatter(
                x=x, y=history[b]["acc"],
                name=f"{b} (local)",
                line=dict(color=BANK_COLORS.get(b, "gray"), dash="dot"),
                opacity=0.75,
            ))
    fig.add_trace(go.Scatter(
        x=x, y=global_acc,
        name="🌐 Global (FedAvg)",
        line=dict(color="white", width=3),
    ))
    fig.update_layout(
        title="Accuracy per FL Round",
        xaxis_title="Round", yaxis_title="Accuracy",
        yaxis=dict(range=[0.5, 1.0]),
        template=_DARK, height=380,
        legend=dict(orientation="h", y=-0.28),
    )
    return fig


def fl_loss_chart(history: dict, sel_banks: list) -> go.Figure:
   
    fig = go.Figure()
    for b in sel_banks:
        losses = history.get(b, {}).get("loss", [])
        if losses:
            fig.add_trace(go.Scatter(
                x=list(range(1, len(losses) + 1)),
                y=losses,
                name=b,
                line=dict(color=BANK_COLORS.get(b, "gray"), width=2),
            ))
    fig.update_layout(
        title="Local Training Loss per Bank",
        xaxis_title="Round", yaxis_title="Loss",
        template=_DARK, height=300,
        legend=dict(orientation="h", y=-0.3),
    )
    return fig


def auc_vs_epsilon_chart(global_auc: list, epsilon_log: list) -> go.Figure:
    """Dual-axis: AUC-ROC (left) vs ε privacy budget (right)."""
    x = list(range(1, len(global_auc) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=global_auc, name="AUC-ROC",
        line=dict(color="#38bdf8", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=epsilon_log, name="ε consumed",
        yaxis="y2",
        line=dict(color="#f87171", dash="dot", width=2),
    ))
    fig.update_layout(
        title="Global AUC-ROC vs Privacy Budget (ε) per Round",
        xaxis_title="FL Round",
        yaxis=dict(title="AUC-ROC", range=[0.5, 1.0]),
        yaxis2=dict(title="ε (privacy budget)", overlaying="y", side="right", showgrid=False),
        template=_DARK, height=340,
        legend=dict(orientation="h", y=-0.3),
    )
    return fig


def baseline_comparison_bar(
    fl_acc: float,
    central_acc: float,
    local_results: dict,
) -> go.Figure:

    labels = ["Centralized\n(no privacy)"]
    values = [central_acc]
    colors = ["#94a3b8"]

    for b, r in local_results.items():
        labels.append(f"{b} Local\n(isolated)")
        values.append(r["val_acc"])
        colors.append(BANK_COLORS.get(b, "gray"))

    labels.append("FL + DP\n(our model)")
    values.append(fl_acc)
    colors.append("#38bdf8")

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f"{v:.2%}" for v in values],
        textposition="outside",
    ))
    fig.add_hline(y=central_acc, line_dash="dash", line_color="#94a3b8",
                  annotation_text="Centralized ceiling")
    fig.update_layout(
        title="Accuracy Comparison: Centralized vs FL+DP vs Local-Only",
        yaxis=dict(title="Accuracy", range=[0.5, 1.05]),
        template=_DARK, height=380,
    )
    return fig


def baseline_learning_curve(history: dict) -> go.Figure:

    ep = history["epoch"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ep, y=history["acc"], name="Train acc",
                              line=dict(color="#38bdf8", width=2)))
    fig.add_trace(go.Scatter(x=ep, y=history["val_acc"], name="Val acc",
                              line=dict(color="#4ade80", width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=ep, y=history["loss"], name="Train loss",
                              yaxis="y2",
                              line=dict(color="#f87171", width=1.5, dash="dash")))
    fig.update_layout(
        title="Centralized Baseline — Learning Curve",
        xaxis_title="Epoch",
        yaxis=dict(title="Accuracy"),
        yaxis2=dict(title="Loss", overlaying="y", side="right", showgrid=False),
        template=_DARK, height=340,
        legend=dict(orientation="h", y=-0.3),
    )
    return fig


def epsilon_vs_accuracy_curve(current_eps: float | None = None) -> go.Figure:
    eps_vals = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0]
    acc_vals = [0.70, 0.76, 0.80, 0.83, 0.85, 0.87, 0.88, 0.89, 0.89]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eps_vals, y=acc_vals,
        mode="lines+markers",
        line=dict(color="#38bdf8", width=3),
        name="Accuracy curve",
    ))
    if current_eps and current_eps < 50:
        fig.add_vline(x=min(current_eps, 10), line_dash="dash", line_color="#f87171",
                      annotation_text=f"Your ε≈{current_eps:.2f}")
    fig.add_hrect(y0=0.84, y1=0.91, fillcolor="#22c55e", opacity=0.07,
                  annotation_text="Good zone")
    fig.update_layout(
        xaxis_title="ε (epsilon)", yaxis_title="Model Accuracy",
        template=_DARK, height=320,
    )
    return fig


def epsilon_budget_bars(epsilon_log: list, budget_limit: float) -> go.Figure:
    rounds = list(range(1, len(epsilon_log) + 1))
    bar_colors = [
        "#22c55e" if v < budget_limit * 0.5
        else "#f97316" if v < budget_limit * 0.85
        else "#ef4444"
        for v in epsilon_log
    ]
    fig = go.Figure(go.Bar(x=rounds, y=epsilon_log, marker_color=bar_colors))
    fig.add_hline(y=budget_limit, line_dash="dash", line_color="white",
                  annotation_text=f"Budget ε={budget_limit}")
    fig.update_layout(
        title="Privacy Budget Consumption per Round",
        xaxis_title="FL Round", yaxis_title="ε consumed",
        template=_DARK, height=300,
    )
    return fig


def income_distribution(all_data: dict) -> go.Figure:
    fig = go.Figure()
    for b, df in all_data.items():
        fig.add_trace(go.Histogram(
            x=df["income"], name=b,
            marker_color=BANK_COLORS.get(b, "gray"),
            opacity=0.7, nbinsx=40,
        ))
    fig.update_layout(
        barmode="overlay",
        title="Income Distribution per Bank (Non-IID!)",
        xaxis_title="Monthly Income (₹)",
        template=_DARK, height=380,
    )
    return fig


def default_rate_bars(all_data: dict) -> go.Figure:
    rates  = {b: df["default"].mean() * 100 for b, df in all_data.items()}
    fig = go.Figure(go.Bar(
        x=list(rates.keys()),
        y=list(rates.values()),
        marker_color=[BANK_COLORS.get(b, "gray") for b in rates],
        text=[f"{v:.1f}%" for v in rates.values()],
        textposition="outside",
    ))
    fig.update_layout(title="Default Rate per Bank", yaxis_title="%",
                      template=_DARK, height=300)
    return fig


def dataset_size_pie(all_data: dict) -> go.Figure:
    sizes = {b: len(df) for b, df in all_data.items()}
    fig = go.Figure(go.Pie(
        labels=list(sizes.keys()),
        values=list(sizes.values()),
        marker_colors=[BANK_COLORS.get(b, "gray") for b in sizes],
        hole=0.4,
    ))
    fig.update_layout(title="Dataset Size per Bank", template=_DARK, height=300)
    return fig


def correlation_heatmap(df, bank_name: str, feature_names: list) -> go.Figure:
    corr = df[feature_names + ["default"]].corr()
    fig  = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                     title=f"Feature Correlation — {bank_name}")
    fig.update_layout(template=_DARK, height=450)
    return fig
