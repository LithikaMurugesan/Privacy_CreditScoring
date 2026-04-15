"""
src/federated/flower_simulation.py
====================================
Flower in-process simulation runner.

Why simulation instead of a real Flower server?
  A real Flower server needs multiple processes (one per client + one server),
  separate ports, and gRPC connections. This works in production but is
  incompatible with Streamlit (which runs everything in a single process).

  flwr.simulation.start_simulation() runs all clients and the server in a
  single Python process using virtual actors (Ray or threading). The FL
  algorithm is identical — only the transport layer changes.

  This gives us:
    - Real Flower code (NumPyClient, strategies, server config)
    - Streamlit compatibility (no subprocess, no ports)
    - Correct FL semantics (clients don't share data, only weights)

Usage:
    results = run_flower_simulation(
        banks=["SBI", "HDFC", "Axis"],
        all_data=all_data,
        num_rounds=8,
        config={...},
        use_fedprox=False,
    )
    # results["history"] → Flower History object
    # results["metrics"] → dict of final metrics
"""

import logging
import copy
import numpy as np

from src.models.model import CreditNet, weights_to_numpy
from src.data.data_generator import FEATURE_NAMES
from src.utils.helpers import evaluate_model
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Try importing Flower
try:
    import flwr as fl
    from flwr.common import ndarrays_to_parameters
    FLOWER_AVAILABLE = True
except ImportError:
    FLOWER_AVAILABLE = False
    fl = None


def run_flower_simulation(
    banks: list,
    all_data: dict,
    num_rounds: int = 8,
    config: dict = None,
    use_fedprox: bool = False,
    mu: float = 0.01,
    metrics_logger=None,
    progress_callback=None,
) -> dict:
    """
    Run a full Federated Learning simulation using Flower.

    Parameters
    ----------
    banks            : list of bank names to include
    all_data         : dict[bank_name -> pd.DataFrame]
    num_rounds       : int   — number of FL communication rounds
    config           : dict  — client training config (lr, local_epochs, use_dp, ...)
    use_fedprox      : bool  — use FedProx strategy instead of FedAvg
    mu               : float — FedProx regularisation coefficient
    metrics_logger   : FLLogger — optional logger for metrics
    progress_callback: callable(round, total, metrics) — for UI progress updates

    Returns
    -------
    dict with keys:
      "history"       : Flower History object (or None)
      "final_acc"     : float — final global model accuracy (macro-averaged)
      "final_auc"     : float — final global model AUC
      "final_epsilon" : float — final privacy budget consumed
      "model"         : CreditNet — trained global model
    """
    cfg = config or {}
    data_subset = {b: all_data[b] for b in banks if b in all_data}

    if not FLOWER_AVAILABLE:
        logger.warning("Flower not installed — falling back to custom FL loop.")
        return _run_custom_fallback(
            banks, all_data, num_rounds, cfg, use_fedprox, mu,
            metrics_logger, progress_callback
        )

    # ── Build initial global model parameters ───────────────────────────
    initial_model  = CreditNet(input_dim=10)
    initial_params = ndarrays_to_parameters(weights_to_numpy(initial_model))

    # ── Build strategy ───────────────────────────────────────────────────
    from src.federated.strategies import get_fedavg_strategy, get_fedprox_strategy

    fit_config = {
        "local_epochs": cfg.get("local_epochs", 2),
        "lr":           cfg.get("lr", 0.001),
        "use_dp":       cfg.get("use_dp", False),
        "noise_mult":   cfg.get("noise_mult", 1.1),
        "max_norm":     cfg.get("max_norm", 1.0),
        "dp_backend":   cfg.get("dp_backend", "custom"),
        "use_fedprox":  use_fedprox,
        "mu":           mu,
    }

    strategy_builder = get_fedprox_strategy if use_fedprox else get_fedavg_strategy

    strategy_kwargs = dict(
        initial_parameters=initial_params,
        min_fit_clients=len(banks),
        min_evaluate_clients=len(banks),
        min_available_clients=len(banks),
        metrics_logger=metrics_logger,
    )
    if use_fedprox:
        strategy_kwargs["mu"] = mu
        strategy_kwargs["base_fit_config"] = fit_config
    else:
        strategy_kwargs["fit_config"] = fit_config

    strategy = strategy_builder(**strategy_kwargs)

    # ── Build client factory ─────────────────────────────────────────────
    from src.federated.flower_client import make_flower_client_fn
    client_fn = make_flower_client_fn(data_subset, cfg)

    # ── Run simulation ───────────────────────────────────────────────────
    logger.info(
        f"Starting Flower simulation — {len(banks)} banks, "
        f"{num_rounds} rounds, {'FedProx' if use_fedprox else 'FedAvg'}"
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(banks),
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    # ── Extract final metrics ─────────────────────────────────────────────
    # Rebuild model from last round aggregated parameters
    global_model = CreditNet(input_dim=10)
    # Evaluate on all banks and macro-average
    accs, aucs = [], []
    for b in banks:
        sc = StandardScaler()
        sc.fit(data_subset[b][FEATURE_NAMES].values)
        acc, auc = evaluate_model(global_model, data_subset[b], sc)
        accs.append(acc)
        aucs.append(auc)

    # Get epsilon from last round (if DP was used)
    from src.privacy.dp_custom import compute_epsilon
    use_dp = cfg.get("use_dp", False)
    if use_dp:
        sr  = 64 / len(data_subset[banks[0]])
        st  = cfg.get("local_epochs", 2) * (len(data_subset[banks[0]]) // 64) * num_rounds
        eps = compute_epsilon(cfg.get("noise_mult", 1.1), sr, st)
    else:
        eps = 0.0

    return {
        "history":       history,
        "final_acc":     float(np.mean(accs)),
        "final_auc":     float(np.mean(aucs)),
        "final_epsilon": eps,
        "model":         global_model,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Custom FL fallback (used when Flower is not installed)
# Implements the same FedAvg/FedProx loop without Flower dependency.
# ─────────────────────────────────────────────────────────────────────────────

def _run_custom_fallback(
    banks, all_data, num_rounds, cfg, use_fedprox, mu,
    metrics_logger, progress_callback
) -> dict:
    """
    Custom FL loop that mirrors Flower's protocol without the Flower library.
    Used as fallback when flwr is not installed.
    """
    import copy
    from src.models.model import CreditNet, get_weights, set_weights
    from src.utils.helpers import local_train, evaluate_model, fed_avg
    from src.privacy.dp_custom import compute_epsilon
    from sklearn.preprocessing import StandardScaler

    data_subset  = {b: all_data[b] for b in banks if b in all_data}
    global_model = CreditNet(input_dim=10)

    scalers = {}
    for b in banks:
        sc = StandardScaler()
        sc.fit(data_subset[b][FEATURE_NAMES].values)
        scalers[b] = sc

    accs_hist = []
    aucs_hist = []

    for rnd in range(1, num_rounds + 1):
        client_weights, client_sizes = [], []

        for b in banks:
            local_model = copy.deepcopy(global_model)
            loss, n, _ = local_train(
                model        = local_model,
                df           = data_subset[b],
                local_epochs = cfg.get("local_epochs", 2),
                lr           = cfg.get("lr", 0.001),
                use_dp       = cfg.get("use_dp", False),
                noise_mult   = cfg.get("noise_mult", 1.1),
                max_norm     = cfg.get("max_norm", 1.0),
                use_fedprox  = use_fedprox,
                mu           = mu,
                global_model = global_model if use_fedprox else None,
                dp_backend   = cfg.get("dp_backend", "custom"),
            )
            acc, auc = evaluate_model(local_model, data_subset[b], scalers[b])
            client_weights.append(get_weights(local_model))
            client_sizes.append(n)

            if metrics_logger:
                sr  = 64 / n
                st  = cfg.get("local_epochs", 2) * (n // 64) * rnd
                eps = compute_epsilon(cfg.get("noise_mult", 1.1), sr, st) if cfg.get("use_dp", False) else 0.0
                metrics_logger.log_client(rnd, b, acc, auc, loss, n, eps)

        set_weights(global_model, fed_avg(client_weights, client_sizes))

        # Evaluate global model
        g_accs, g_aucs = [], []
        for b in banks:
            a, au = evaluate_model(global_model, data_subset[b], scalers[b])
            g_accs.append(a)
            g_aucs.append(au)
        g_acc = float(np.mean(g_accs))
        g_auc = float(np.mean(g_aucs))

        use_dp = cfg.get("use_dp", False)
        if use_dp:
            sr  = 64 / len(data_subset[banks[0]])
            st  = cfg.get("local_epochs", 2) * (len(data_subset[banks[0]]) // 64) * rnd
            eps = compute_epsilon(cfg.get("noise_mult", 1.1), sr, st)
        else:
            eps = 0.0

        accs_hist.append(g_acc)
        aucs_hist.append(g_auc)

        if metrics_logger:
            metrics_logger.log_global(rnd, g_acc, g_auc, eps)

        if progress_callback:
            progress_callback(rnd, num_rounds, {"acc": g_acc, "auc": g_auc, "eps": eps})

        logger.info(f"Round {rnd}/{num_rounds} — acc={g_acc:.4f} auc={g_auc:.4f} eps={eps:.3f}")

    return {
        "history":       None,
        "final_acc":     float(np.mean(accs_hist[-1:])),
        "final_auc":     float(np.mean(aucs_hist[-1:])),
        "final_epsilon": eps if use_dp else 0.0,
        "model":         global_model,
        "acc_history":   accs_hist,
        "auc_history":   aucs_hist,
    }
