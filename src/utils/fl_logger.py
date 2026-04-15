"""
src/utils/fl_logger.py
======================
Structured Federated Learning logger.

Collects per-round, per-bank training metrics and global model metrics.
Supports:
  - CSV export (per-client, global, combined)
  - Human-readable round summaries
  - Auto-save to experiments/results/ directory

Usage:
    logger = FLLogger(experiment_name="fedavg_dp_run1")
    logger.log_client(round=1, bank="SBI", acc=0.82, auc=0.87, ...)
    logger.log_global(round=1, g_acc=0.84, g_auc=0.88, epsilon=1.2)
    logger.save_to_disk()   # saves CSVs to experiments/results/
"""

import os
import pandas as pd
from datetime import datetime


class FLLogger:
    """
    Structured logger for Federated Learning training runs.

    Stores per-round, per-bank training metrics as a structured log.
    Supports CSV export and summary statistics.

    Parameters
    ----------
    experiment_name : str  — label for saved files (default: timestamp)
    save_dir        : str  — directory to save CSVs (default: experiments/results)
    """

    def __init__(
        self,
        experiment_name: str = "",
        save_dir: str = "experiments/results",
    ):
        self.records   = []   # per-bank metrics (one per bank per round)
        self.g_records = []   # global model metrics (one per round)
        self.started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.save_dir   = save_dir
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Per-bank log ────────────────────────────────────────────────────────
    def log_client(
        self,
        round_num: int,
        bank: str,
        acc: float,
        auc: float,
        loss: float,
        n_samples: int,
        epsilon: float,
    ) -> None:
        """Log one bank's training metrics for a given FL round."""
        self.records.append({
            "round":     round_num,
            "bank":      bank,
            "accuracy":  round(acc,  6),
            "auc_roc":   round(auc,  6),
            "loss":      round(loss, 6),
            "n_samples": n_samples,
            "epsilon":   round(epsilon, 4),
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        })

    # ── Global model log ────────────────────────────────────────────────────
    def log_global(
        self,
        round_num: int,
        g_acc: float,
        g_auc: float,
        epsilon: float,
        delta: float = 1e-5,
    ) -> None:
        """Log global model metrics after aggregation for a given FL round."""
        self.g_records.append({
            "round":      round_num,
            "global_acc": round(g_acc, 6),
            "global_auc": round(g_auc, 6),
            "epsilon":    round(epsilon, 4),
            "delta":      delta,
            "privacy_ok": epsilon < 3.0,
            "timestamp":  datetime.now().strftime("%H:%M:%S"),
        })

    # ── DataFrames ──────────────────────────────────────────────────────────
    def client_df(self) -> pd.DataFrame:
        """Return per-bank records as a DataFrame."""
        return pd.DataFrame(self.records) if self.records else pd.DataFrame()

    def global_df(self) -> pd.DataFrame:
        """Return global model records as a DataFrame."""
        return pd.DataFrame(self.g_records) if self.g_records else pd.DataFrame()

    # ── CSV export ──────────────────────────────────────────────────────────
    def client_csv(self) -> bytes:
        """Encode per-bank DataFrame as CSV bytes for download."""
        return self.client_df().to_csv(index=False).encode()

    def global_csv(self) -> bytes:
        """Encode global model DataFrame as CSV bytes for download."""
        return self.global_df().to_csv(index=False).encode()

    def combined_csv(self) -> bytes:
        """Merge client + global logs on round number and encode as CSV bytes."""
        gdf = self.global_df().rename(
            columns=lambda c: f"global_{c}" if c != "round" else c
        )
        cdf = self.client_df()
        if gdf.empty or cdf.empty:
            return b""
        merged = cdf.merge(gdf, on="round", how="left")
        return merged.to_csv(index=False).encode()

    def save_to_disk(self) -> str:
        """
        Save training logs to experiments/results/ directory.

        Files saved:
          - {experiment_name}_client_log.csv
          - {experiment_name}_global_log.csv

        Returns
        -------
        str — directory where files were saved
        """
        os.makedirs(self.save_dir, exist_ok=True)
        prefix = os.path.join(self.save_dir, self.experiment_name)

        cdf = self.client_df()
        gdf = self.global_df()

        if not cdf.empty:
            cdf.to_csv(f"{prefix}_client_log.csv", index=False)
        if not gdf.empty:
            gdf.to_csv(f"{prefix}_global_log.csv", index=False)

        return self.save_dir

    # ── Summary ─────────────────────────────────────────────────────────────
    def summary(self) -> dict:
        """Return a summary dict of the training run."""
        if not self.g_records:
            return {}
        gdf = self.global_df()
        cdf = self.client_df()
        return {
            "total_rounds":     len(gdf),
            "final_global_acc": gdf["global_acc"].iloc[-1],
            "final_global_auc": gdf["global_auc"].iloc[-1],
            "final_epsilon":    gdf["epsilon"].iloc[-1],
            "privacy_ok":       bool(gdf["privacy_ok"].iloc[-1]),
            "banks_trained":    cdf["bank"].nunique() if not cdf.empty else 0,
            "total_samples":    int(cdf.groupby("bank")["n_samples"].first().sum())
                                if not cdf.empty else 0,
            "started_at":       self.started_at,
            "experiment_name":  self.experiment_name,
        }

    # ── Human-readable round log lines ──────────────────────────────────────
    def round_lines(self, round_num: int) -> list:
        """Return log lines for a single FL round (for display in terminal/UI)."""
        lines = [f"=== Round {round_num} ==="]
        for r in self.records:
            if r["round"] == round_num:
                lines.append(
                    f"  {r['bank']:5s} -> acc={r['accuracy']:.4f} | "
                    f"auc={r['auc_roc']:.4f} | loss={r['loss']:.4f} | "
                    f"n={r['n_samples']:,} | eps={r['epsilon']:.3f}"
                )
        for g in self.g_records:
            if g["round"] == round_num:
                lines.append(
                    f"  [Global] -> acc={g['global_acc']:.4f} | "
                    f"auc={g['global_auc']:.4f} | eps={g['epsilon']:.4f}"
                )
        return lines
