"""
src/utils/plots.py
==================
All Plotly chart builders for the Streamlit dashboard.

Each function takes data arguments and returns a go.Figure — no Streamlit
calls inside this module so it can be used by report-generation scripts too.

Charts available:
  FL Training:
    fl_accuracy_chart()      — local + global accuracy per round
    fl_loss_chart()          — per-bank training loss per round
    auc_vs_epsilon_chart()   — AUC vs privacy budget (dual axis)

  Baseline Comparison:
    baseline_comparison_bar()   — Centralized vs FL+DP vs Local-only
    baseline_learning_curve()   — Centralized train/val curves

  Privacy Analysis:
    epsilon_vs_accuracy_curve()  — tradeoff curve
    epsilon_budget_bars()        — budget consumption per round
    privacy_tradeoff_scatter()   — NEW: scatter of modes (ε vs accuracy)

  Data Explorer:
    income_distribution()    — histogram per bank
    default_rate_bars()      — bar chart of default rates
    dataset_size_pie()       — donut chart of dataset sizes
    correlation_heatmap()    — feature correlation for one bank

  Performance Comparison:
    mode_comparison_bar()    — NEW: bar for all 4 training modes (accuracy)
    mode_comparison_auc()    — NEW: bar for all 4 training modes (AUC)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
BANK_COLORS = {
    "SBI":   "#f97316",
    "HDFC":  "#3b82f6",
    "Axis":  "#22c55e",
    "PNB":   "#a855f7",
    "ICICI": "#ec4899",
    "Kotak": "#eab308",
}

MODE_COLORS = {
    "Centralized (no privacy)": "#94a3b8",
    "FedAvg":                   "#22c55e",
    "FedAvg + DP":              "#38bdf8",
    "FedProx + DP":             "#f97316",
}

_DARK = "plotly_dark"


# ═════════════════════════════════════════════════════════════════════════════
# FL TRAINING CHARTS
# ═════════════════════════════════════════════════════════════════════════════

def fl_accuracy_chart(history: dict, global_acc: list, sel_banks: list) -> go.Figure:
    """
    Live accuracy chart — per-bank (dashed) + global FedAvg (solid white).

    Parameters
    ----------
    history    : dict[bank -> {"acc": [...], ...}]
    global_acc : list of floats — global model accuracy per round
    sel_banks  : list of bank names to plot
    """
    x = list(range(1, len(global_acc) + 1))
    fig = go.Figure()

    for b in sel_banks:
        accs = history.get(b, {}).get("acc", [])
        if accs:
            fig.add_trace(go.Scatter(
                x=x[:len(accs)], y=accs,
                name=f"{b} (local)",
                line=dict(color=BANK_COLORS.get(b, "#888"), dash="dot"),
                opacity=0.75,
            ))

    fig.add_trace(go.Scatter(
        x=x, y=global_acc,
        name="Global (FedAvg)",
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
    """Per-bank training loss across FL rounds."""
    fig = go.Figure()
    for b in sel_banks:
        losses = history.get(b, {}).get("loss", [])
        if losses:
            fig.add_trace(go.Scatter(
                x=list(range(1, len(losses) + 1)),
                y=losses,
                name=b,
                line=dict(color=BANK_COLORS.get(b, "#888"), width=2),
            ))
    fig.update_layout(
        title="Local Training Loss per Bank",
        xaxis_title="Round", yaxis_title="Loss",
        template=_DARK, height=300,
        legend=dict(orientation="h", y=-0.3),
    )
    return fig


def auc_vs_epsilon_chart(global_auc: list, epsilon_log: list) -> go.Figure:
    """Dual-axis: AUC-ROC (left axis) vs privacy budget ε (right axis)."""
    x = list(range(1, len(global_auc) + 1))
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x, y=global_auc, name="AUC-ROC",
        line=dict(color="#38bdf8", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=epsilon_log, name="epsilon consumed",
        yaxis="y2",
        line=dict(color="#f87171", dash="dot", width=2),
    ))

    fig.update_layout(
        title="Global AUC-ROC vs Privacy Budget (epsilon) per Round",
        xaxis_title="FL Round",
        yaxis=dict(title="AUC-ROC", range=[0.5, 1.0]),
        yaxis2=dict(title="epsilon (privacy budget)", overlaying="y", side="right", showgrid=False),
        template=_DARK, height=340,
        legend=dict(orientation="h", y=-0.3),
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# BASELINE COMPARISON CHARTS
# ═════════════════════════════════════════════════════════════════════════════

def baseline_comparison_bar(
    fl_acc: float,
    central_acc: float,
    local_results: dict,
) -> go.Figure:
    """
    Bar chart: Centralized (upper bound) vs FL+DP vs Local-only (lower bound).
    """
    labels = ["Centralized\n(no privacy)"]
    values = [central_acc]
    colors = ["#94a3b8"]

    for b, r in local_results.items():
        labels.append(f"{b} Local\n(isolated)")
        values.append(r["val_acc"])
        colors.append(BANK_COLORS.get(b, "#888"))

    labels.append("FL + DP\n(our model)")
    values.append(fl_acc)
    colors.append("#38bdf8")

    fig = go.Figure(go.Bar(
        x=labels, y=values,
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
    """Centralized baseline — train vs validation accuracy + loss per epoch."""
    ep = history["epoch"]
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=ep, y=history["acc"],     name="Train Accuracy",
                             line=dict(color="#38bdf8", width=2)))
    fig.add_trace(go.Scatter(x=ep, y=history["val_acc"], name="Val Accuracy",
                             line=dict(color="#4ade80", width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=ep, y=history["loss"],    name="Train Loss",
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


# ═════════════════════════════════════════════════════════════════════════════
# PERFORMANCE COMPARISON CHARTS  (NEW)
# ═════════════════════════════════════════════════════════════════════════════

def mode_comparison_bar(rows: list, metric: str = "Accuracy") -> go.Figure:
    """
    Bar chart comparing all 4 training modes on a given metric.

    Parameters
    ----------
    rows   : list of dicts with keys: Method, Accuracy, AUC-ROC, ...
    metric : str — "Accuracy" or "AUC-ROC"
    """
    methods = [r["Method"] for r in rows]
    values  = [r[metric]   for r in rows]
    colors  = [MODE_COLORS.get(m, "#6366f1") for m in methods]

    fmt = "{:.2%}" if metric == "Accuracy" else "{:.4f}"
    fig = go.Figure(go.Bar(
        x=methods, y=values,
        marker_color=colors,
        text=[fmt.format(v) for v in values],
        textposition="outside",
    ))

    y_range = [0, 1.05] if metric == "Accuracy" else [0, 1]
    tick_fmt = ".0%" if metric == "Accuracy" else ".3f"

    fig.update_layout(
        title=f"{metric} by Training Mode",
        yaxis=dict(title=metric, range=y_range, tickformat=tick_fmt),
        xaxis_title="",
        template=_DARK, height=420,
        margin=dict(t=60, b=10),
    )
    return fig


def privacy_tradeoff_scatter(rows: list) -> go.Figure:
    """
    NEW — Scatter plot: Privacy budget (epsilon) vs Accuracy.

    Shows the fundamental privacy-accuracy tradeoff. Modes without DP
    are plotted at the right edge (high epsilon = low privacy).
    """
    fig = go.Figure()

    for r in rows:
        eps = r.get("Epsilon")
        acc = r.get("Accuracy", 0)
        method = r.get("Method", "")

        # Modes without DP are shown at epsilon=10 (placeholder for "infinite")
        x_val = eps if eps is not None else 10.0
        label = f"epsilon={eps:.2f}" if eps is not None else "No DP"

        fig.add_trace(go.Scatter(
            x=[x_val], y=[acc * 100],
            mode="markers+text",
            marker=dict(size=18, color=MODE_COLORS.get(method, "#6366f1")),
            text=[method],
            textposition="top center",
            name=method,
            hovertemplate=f"<b>{method}</b><br>epsilon={label}<br>Accuracy={acc:.2%}<extra></extra>",
        ))

    # Add privacy zones as background
    fig.add_vrect(x0=0, x1=3, fillcolor="#22c55e", opacity=0.05,
                  annotation_text="Strong Privacy Zone", annotation_position="top left")
    fig.add_vrect(x0=3, x1=7, fillcolor="#f97316", opacity=0.05,
                  annotation_text="Moderate", annotation_position="top left")
    fig.add_vrect(x0=7, x1=11, fillcolor="#ef4444", opacity=0.05)

    fig.update_layout(
        title="Privacy-Accuracy Tradeoff — lower epsilon = stronger privacy",
        xaxis_title="epsilon (privacy budget) — lower is more private",
        yaxis_title="Accuracy (%)",
        xaxis=dict(range=[0, 11]),
        template=_DARK, height=420,
        showlegend=False,
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# PRIVACY ANALYSIS CHARTS
# ═════════════════════════════════════════════════════════════════════════════

def epsilon_vs_accuracy_curve(current_eps: float = None) -> go.Figure:
    """Theoretical epsilon vs accuracy curve with current epsilon marker."""
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
        fig.add_vline(
            x=min(current_eps, 10),
            line_dash="dash", line_color="#f87171",
            annotation_text=f"Current epsilon~{current_eps:.2f}",
        )

    fig.add_hrect(y0=0.84, y1=0.91, fillcolor="#22c55e", opacity=0.07,
                  annotation_text="Good accuracy zone")

    fig.update_layout(
        xaxis_title="epsilon (epsilon)", yaxis_title="Model Accuracy",
        template=_DARK, height=320,
    )
    return fig


def epsilon_budget_bars(epsilon_log: list, budget_limit: float) -> go.Figure:
    """Color-coded bars showing epsilon consumption per FL round."""
    rounds = list(range(1, len(epsilon_log) + 1))
    bar_colors = [
        "#22c55e" if v < budget_limit * 0.5
        else "#f97316" if v < budget_limit * 0.85
        else "#ef4444"
        for v in epsilon_log
    ]

    fig = go.Figure(go.Bar(x=rounds, y=epsilon_log, marker_color=bar_colors))
    fig.add_hline(y=budget_limit, line_dash="dash", line_color="white",
                  annotation_text=f"Budget epsilon={budget_limit}")

    fig.update_layout(
        title="Privacy Budget Consumption per Round",
        xaxis_title="FL Round", yaxis_title="epsilon consumed",
        template=_DARK, height=300,
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# DATA EXPLORER CHARTS
# ═════════════════════════════════════════════════════════════════════════════

def income_distribution(all_data: dict) -> go.Figure:
    """Overlapping income histograms — shows Non-IID nature of bank data."""
    fig = go.Figure()
    for b, df in all_data.items():
        fig.add_trace(go.Histogram(
            x=df["income"], name=b,
            marker_color=BANK_COLORS.get(b, "#888"),
            opacity=0.7, nbinsx=40,
        ))
    fig.update_layout(
        barmode="overlay",
        title="Income Distribution per Bank (Non-IID)",
        xaxis_title="Monthly Income (INR)",
        template=_DARK, height=380,
    )
    return fig


def default_rate_bars(all_data: dict) -> go.Figure:
    """Bar chart showing default rate heterogeneity across banks."""
    rates = {b: df["default"].mean() * 100 for b, df in all_data.items()}
    fig = go.Figure(go.Bar(
        x=list(rates.keys()),
        y=list(rates.values()),
        marker_color=[BANK_COLORS.get(b, "#888") for b in rates],
        text=[f"{v:.1f}%" for v in rates.values()],
        textposition="outside",
    ))
    fig.update_layout(title="Default Rate per Bank", yaxis_title="%",
                      template=_DARK, height=300)
    return fig


def dataset_size_pie(all_data: dict) -> go.Figure:
    """Donut chart showing dataset size distribution across banks."""
    sizes = {b: len(df) for b, df in all_data.items()}
    fig = go.Figure(go.Pie(
        labels=list(sizes.keys()),
        values=list(sizes.values()),
        marker_colors=[BANK_COLORS.get(b, "#888") for b in sizes],
        hole=0.4,
    ))
    fig.update_layout(title="Dataset Size per Bank", template=_DARK, height=300)
    return fig


def correlation_heatmap(df, bank_name: str, feature_names: list) -> go.Figure:
    """Feature correlation heatmap for a single bank."""
    corr = df[feature_names + ["default"]].corr()
    fig  = px.imshow(
        corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title=f"Feature Correlation — {bank_name}",
    )
    fig.update_layout(template=_DARK, height=450)
    return fig
