"""
src/federated/fl_engine.py
==========================
Unified FL Engine — single entry point for all training modes.

This is the main orchestrator. The Streamlit app, CLI tool, and API
all call FLEngine.run() without knowing which backend is being used.

Supported modes:
  "centralized"  — pool all data, train a single model (no FL, no privacy)
  "fedavg"       — federated FedAvg, no DP
  "fedavg_dp"    — federated FedAvg + Differential Privacy
  "fedprox_dp"   — federated FedProx + Differential Privacy

Supported FL backends:
  "flower"  — uses Flower simulation (flwr.simulation.start_simulation)
  "custom"  — uses in-process custom FL loop (no Flower dependency)

Supported DP backends:
  "opacus"  — Opacus PrivacyEngine (accurate per-sample DP)
  "custom"  — manual gradient clipping + Gaussian noise
  "none"    — no DP

Usage:
    engine = FLEngine()
    results = engine.run(
        mode="fedavg_dp",
        banks=["SBI", "HDFC", "Axis"],
        all_data=all_data,
        num_rounds=8,
        local_epochs=2,
        lr=0.001,
        noise_mult=1.1,
        max_norm=1.0,
        mu=0.01,
        fl_backend="flower",
        dp_backend="custom",
        logger=fl_logger,
        progress_cb=None,
    )
"""

import copy
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.models.model import CreditNet, get_weights, set_weights
from src.utils.helpers import local_train, evaluate_model, fed_avg
from src.privacy.dp_custom import compute_epsilon
from src.data.data_generator import FEATURE_NAMES

logger = logging.getLogger(__name__)


class FLEngine:
    """
    Unified Federated Learning Engine.

    Dispatches to the correct backend (Flower or custom loop)
    based on the mode and fl_backend parameters.
    """

    def run(
        self,
        mode: str,
        banks: list,
        all_data: dict,
        num_rounds: int = 8,
        local_epochs: int = 2,
        lr: float = 0.001,
        noise_mult: float = 1.1,
        max_norm: float = 1.0,
        mu: float = 0.01,
        fl_backend: str = "custom",
        dp_backend: str = "custom",
        metrics_logger=None,
        progress_cb=None,
    ) -> dict:
        """
        Run one complete training session.

        Parameters
        ----------
        mode        : str  — "centralized" | "fedavg" | "fedavg_dp" | "fedprox_dp"
        banks       : list — bank names to include
        all_data    : dict — bank_name -> pd.DataFrame
        num_rounds  : int  — FL communication rounds
        local_epochs: int  — local training epochs per round
        lr          : float — Adam learning rate
        noise_mult  : float — DP noise multiplier (σ)
        max_norm    : float — gradient clipping norm (C)
        mu          : float — FedProx regularisation coefficient
        fl_backend  : str  — "flower" | "custom"
        dp_backend  : str  — "opacus" | "custom" | "none"
        metrics_logger : FLLogger — optional structured logger
        progress_cb : callable(round, total, metrics) — UI progress callback

        Returns
        -------
        dict with keys:
          mode, final_acc, final_auc, final_epsilon,
          model, history (optional), acc_history, auc_history,
          epsilon_history, bank_history
        """
        assert mode in ("centralized", "fedavg", "fedavg_dp", "fedprox_dp"), \
            f"Unknown mode: {mode}"

        logger.info(f"FLEngine.run() — mode={mode} fl_backend={fl_backend} dp_backend={dp_backend}")

        if mode == "centralized":
            return self._run_centralized(banks, all_data, num_rounds, local_epochs, lr)

        # FL modes
        use_dp      = mode in ("fedavg_dp", "fedprox_dp")
        use_fedprox = mode == "fedprox_dp"

        config = dict(
            local_epochs = local_epochs,
            lr           = lr,
            use_dp       = use_dp,
            noise_mult   = noise_mult,
            max_norm     = max_norm,
            dp_backend   = dp_backend if use_dp else "none",
            use_fedprox  = use_fedprox,
            mu           = mu,
        )

        if fl_backend == "flower":
            from src.federated.flower_simulation import run_flower_simulation
            result = run_flower_simulation(
                banks            = banks,
                all_data         = all_data,
                num_rounds       = num_rounds,
                config           = config,
                use_fedprox      = use_fedprox,
                mu               = mu,
                metrics_logger   = metrics_logger,
                progress_callback= progress_cb,
            )
        else:
            result = self._run_custom_loop(
                banks, all_data, num_rounds, config, use_fedprox, mu,
                metrics_logger, progress_cb
            )

        result["mode"] = mode
        return result

    # ──────────────────────────────────────────────────────────────────────
    # Centralized baseline
    # ──────────────────────────────────────────────────────────────────────

    def _run_centralized(self, banks, all_data, num_rounds, local_epochs, lr) -> dict:
        """Pool all bank data and train a single model (no FL, no DP)."""
        from src.federated.baseline import train_centralized_baseline
        import pandas as pd

        epochs = max(10, num_rounds * local_epochs)
        result = train_centralized_baseline(all_data, banks, epochs=epochs, lr=lr)
        return {
            "mode":            "centralized",
            "final_acc":       result["final_acc"],
            "final_auc":       result["final_auc"],
            "final_epsilon":   0.0,
            "model":           result["model"],
            "history":         result["history"],
            "acc_history":     result["history"]["val_acc"],
            "auc_history":     result["history"]["val_auc"],
            "epsilon_history": [0.0] * epochs,
            "bank_history":    {},
        }

    # ──────────────────────────────────────────────────────────────────────
    # Custom FL loop (no Flower dependency)
    # ──────────────────────────────────────────────────────────────────────

    def _run_custom_loop(
        self, banks, all_data, num_rounds, cfg, use_fedprox, mu,
        metrics_logger, progress_cb
    ) -> dict:
        """
        In-process FL loop implementing FedAvg / FedProx + optional DP.
        This is the fallback when Flower is not installed.
        """
        data_subset  = {b: all_data[b] for b in banks if b in all_data}
        global_model = CreditNet(input_dim=10)

        scalers = {}
        for b in banks:
            sc = StandardScaler()
            sc.fit(data_subset[b][FEATURE_NAMES].values)
            scalers[b] = sc

        acc_history     = []
        auc_history     = []
        epsilon_history = []
        bank_history    = {b: {"acc": [], "auc": [], "loss": []} for b in banks}

        for rnd in range(1, num_rounds + 1):
            client_weights, client_sizes = [], []

            for b in banks:
                local_model = copy.deepcopy(global_model)
                loss, n, _ = local_train(
                    model        = local_model,
                    df           = data_subset[b],
                    local_epochs = cfg["local_epochs"],
                    lr           = cfg["lr"],
                    use_dp       = cfg["use_dp"],
                    noise_mult   = cfg["noise_mult"],
                    max_norm     = cfg["max_norm"],
                    use_fedprox  = use_fedprox,
                    mu           = mu,
                    global_model = global_model if use_fedprox else None,
                    dp_backend   = cfg["dp_backend"],
                )
                acc, auc = evaluate_model(local_model, data_subset[b], scalers[b])
                bank_history[b]["acc"].append(acc)
                bank_history[b]["auc"].append(auc)
                bank_history[b]["loss"].append(loss)

                client_weights.append(get_weights(local_model))
                client_sizes.append(n)

                # Per-client epsilon
                eps_b = 0.0
                if cfg["use_dp"]:
                    sr    = 64 / n
                    steps = cfg["local_epochs"] * (n // 64) * rnd
                    eps_b = compute_epsilon(cfg["noise_mult"], sr, steps)

                if metrics_logger:
                    metrics_logger.log_client(rnd, b, acc, auc, loss, n, eps_b)

            # Aggregate
            set_weights(global_model, fed_avg(client_weights, client_sizes))

            # Global evaluation (macro-average across banks)
            g_accs, g_aucs = [], []
            for b in banks:
                a, au = evaluate_model(global_model, data_subset[b], scalers[b])
                g_accs.append(a)
                g_aucs.append(au)

            g_acc = float(np.mean(g_accs))
            g_auc = float(np.mean(g_aucs))

            eps = 0.0
            if cfg["use_dp"]:
                sr    = 64 / len(data_subset[banks[0]])
                steps = cfg["local_epochs"] * (len(data_subset[banks[0]]) // 64) * rnd
                eps   = compute_epsilon(cfg["noise_mult"], sr, steps)

            acc_history.append(g_acc)
            auc_history.append(g_auc)
            epsilon_history.append(eps)

            if metrics_logger:
                metrics_logger.log_global(rnd, g_acc, g_auc, eps)

            if progress_cb:
                progress_cb(rnd, num_rounds, {
                    "acc": g_acc, "auc": g_auc, "eps": eps,
                    "bank_history": bank_history,
                })

            logger.info(
                f"Round {rnd}/{num_rounds} — "
                f"acc={g_acc:.4f} auc={g_auc:.4f} eps={eps:.3f}"
            )

        return {
            "final_acc":     acc_history[-1] if acc_history else 0.0,
            "final_auc":     auc_history[-1] if auc_history else 0.0,
            "final_epsilon": epsilon_history[-1] if epsilon_history else 0.0,
            "model":         global_model,
            "history":       None,
            "acc_history":   acc_history,
            "auc_history":   auc_history,
            "epsilon_history": epsilon_history,
            "bank_history":  bank_history,
        }
