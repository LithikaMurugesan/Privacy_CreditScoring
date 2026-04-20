
import argparse
import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run FL Credit Scoring experiment"
    )
    parser.add_argument("--banks",       nargs="+",
                        default=["SBI", "HDFC", "Axis", "PNB", "ICICI"],
                        help="Banks to include in FL")
    parser.add_argument("--rounds",      type=int,   default=8,
                        help="Number of FL rounds")
    parser.add_argument("--epochs",      type=int,   default=2,
                        help="Local epochs per round")
    parser.add_argument("--lr",          type=float, default=0.001)
    parser.add_argument("--noise-mult",  type=float, default=1.1,
                        help="DP noise multiplier (sigma)")
    parser.add_argument("--max-norm",    type=float, default=1.0,
                        help="Gradient clipping norm (C)")
    parser.add_argument("--mu",          type=float, default=0.01,
                        help="FedProx proximal coefficient")
    parser.add_argument("--fl-backend",  choices=["flower", "custom"],
                        default="custom", help="FL backend to use")
    parser.add_argument("--dp-backend",  choices=["opacus", "custom"],
                        default="custom", help="DP backend to use")
    parser.add_argument("--mode",        default="all",
                        choices=["all", "centralized", "fedavg", "fedavg_dp", "fedprox_dp"],
                        help="Training mode (default: all)")
    parser.add_argument("--save-dir",    default="experiments/results")
    return parser.parse_args()


def save_comparison_plot(rows: list, save_path: str):
    """Save a bar chart comparing accuracy across all modes."""
    methods = [r["Method"] for r in rows]
    accs    = [r["Accuracy"] for r in rows]
    aucs    = [r["AUC-ROC"]  for r in rows]

    colors = ["#94a3b8", "#22c55e", "#38bdf8", "#f97316"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0f172a")

    for ax in axes:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")

    # Accuracy
    bars = axes[0].bar(methods, accs, color=colors[:len(methods)])
    axes[0].set_title("Accuracy by Training Mode", color="white")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim([0, 1])
    for bar, v in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.01,
                     f"{v:.2%}", ha="center", va="bottom", color="white", fontsize=9)

    # AUC
    bars2 = axes[1].bar(methods, aucs, color=colors[:len(methods)])
    axes[1].set_title("AUC-ROC by Training Mode", color="white")
    axes[1].set_ylabel("AUC-ROC")
    axes[1].set_ylim([0, 1])
    for bar, v in zip(bars2, aucs):
        axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.01,
                     f"{v:.4f}", ha="center", va="bottom", color="white", fontsize=9)

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    logger.info(f"Comparison plot saved to {save_path}")


def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("Loading bank data...")
    from src.data.data_generator import load_all_data
    all_data = load_all_data()

    if args.mode == "all":
        from src.federated.comparison import run_comparison
        logger.info("Running full comparison (all 4 modes)...")
        rows = run_comparison(
            banks        = args.banks,
            all_data     = all_data,
            num_rounds   = args.rounds,
            local_epochs = args.epochs,
            lr           = args.lr,
            noise_mult   = args.noise_mult,
            max_norm     = args.max_norm,
            mu           = args.mu,
            fl_backend   = args.fl_backend,
            dp_backend   = args.dp_backend,
            verbose      = True,
        )
    else:
        from src.federated.fl_engine import FLEngine
        from src.utils.fl_logger import FLLogger

        fl_logger = FLLogger(
            experiment_name=f"{args.mode}_{timestamp}",
            save_dir=args.save_dir,
        )
        engine = FLEngine()
        result = engine.run(
            mode         = args.mode,
            banks        = args.banks,
            all_data     = all_data,
            num_rounds   = args.rounds,
            local_epochs = args.epochs,
            lr           = args.lr,
            noise_mult   = args.noise_mult,
            max_norm     = args.max_norm,
            mu           = args.mu,
            fl_backend   = args.fl_backend,
            dp_backend   = args.dp_backend,
            metrics_logger = fl_logger,
        )
        rows = [{
            "Method":     args.mode,
            "Accuracy":   result["final_acc"],
            "AUC-ROC":    result["final_auc"],
            "Privacy":    f"DP (eps~{result['final_epsilon']:.3f})" if result["final_epsilon"] > 0 else "None",
            "Data Shared": "Weights only",
            "Epsilon":    result["final_epsilon"],
        }]

        # Save model
        model_path = os.path.join(args.save_dir, "best_model.pt")
        import torch
        torch.save(result["model"].state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")

        # Save logger CSVs
        fl_logger.save_to_disk()

    # Save comparison CSV
    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.save_dir, f"{timestamp}_comparison.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")

    # Save comparison plot
    plot_path = os.path.join(args.save_dir, f"{timestamp}_comparison.png")
    try:
        save_comparison_plot(rows, plot_path)
    except Exception as e:
        logger.warning(f"Could not save plot: {e}")

    # Print summary
    print("\n" + "=" * 55)
    print("  EXPERIMENT RESULTS")
    print("=" * 55)
    df_display = df[["Method", "Accuracy", "AUC-ROC", "Privacy"]].copy()
    df_display["Accuracy"] = df_display["Accuracy"].map("{:.2%}".format)
    df_display["AUC-ROC"]  = df_display["AUC-ROC"].map("{:.4f}".format)
    print(df_display.to_string(index=False))
    print(f"\nResults directory: {args.save_dir}")


if __name__ == "__main__":
    main()
