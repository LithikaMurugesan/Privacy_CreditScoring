
import pandas as pd
import io
from datetime import datetime


class FLLogger:
    """
    Stores per-round, per-bank training metrics as a structured log.
    Supports CSV export and summary statistics.
    """

    def __init__(self):
        self.records  = []   # list of dicts (one per bank per round)
        self.g_records = []  # global model metrics per round
        self.started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    
    def log_client(self, round_num: int, bank: str, acc: float, auc: float,
                   loss: float, n_samples: int, epsilon: float):
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
    def log_global(self, round_num: int, g_acc: float, g_auc: float,
                   epsilon: float, delta: float = 1e-5):
        self.g_records.append({
            "round":        round_num,
            "global_acc":   round(g_acc, 6),
            "global_auc":   round(g_auc, 6),
            "epsilon":      round(epsilon, 4),
            "delta":        delta,
            "privacy_ok":   epsilon < 3.0,
            "timestamp":    datetime.now().strftime("%H:%M:%S"),
        })

    def client_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.records) if self.records else pd.DataFrame()

    def global_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.g_records) if self.g_records else pd.DataFrame()

    def client_csv(self) -> bytes:
        return self.client_df().to_csv(index=False).encode()

    def global_csv(self) -> bytes:
        return self.global_df().to_csv(index=False).encode()

    def combined_csv(self) -> bytes:
        """Single CSV: global metrics + per-bank detail side by side."""
        gdf = self.global_df().rename(columns=lambda c: f"global_{c}" if c != "round" else c)
        cdf = self.client_df()
        if gdf.empty or cdf.empty:
            return b""
        merged = cdf.merge(gdf, on="round", how="left")
        return merged.to_csv(index=False).encode()

    def summary(self) -> dict:
        if not self.g_records:
            return {}
        gdf = self.global_df()
        cdf = self.client_df()
        return {
            "total_rounds":    len(gdf),
            "final_global_acc": gdf["global_acc"].iloc[-1],
            "final_global_auc": gdf["global_auc"].iloc[-1],
            "final_epsilon":    gdf["epsilon"].iloc[-1],
            "privacy_ok":       bool(gdf["privacy_ok"].iloc[-1]),
            "banks_trained":    cdf["bank"].nunique() if not cdf.empty else 0,
            "total_samples":    int(cdf.groupby("bank")["n_samples"].first().sum()) if not cdf.empty else 0,
            "started_at":       self.started_at,
        }
    def round_lines(self, round_num: int) -> list[str]:
        lines = [f"━━ Round {round_num} ━━"]
        for r in self.records:
            if r["round"] == round_num:
                lines.append(
                    f"  {r['bank']:4s} → acc={r['accuracy']:.4f} | "
                    f"auc={r['auc_roc']:.4f} | loss={r['loss']:.4f} | "
                    f"n={r['n_samples']:,} | ε={r['epsilon']:.3f}"
                )
        for g in self.g_records:
            if g["round"] == round_num:
                lines.append(
                    f"  🌐 Global → acc={g['global_acc']:.4f} | "
                    f"auc={g['global_auc']:.4f} | ε={g['epsilon']:.4f}"
                )
        return lines
