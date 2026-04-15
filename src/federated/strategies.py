"""
src/federated/strategies.py
============================
Custom Flower server strategies: FedAvg and FedProx.

In Flower, a 'strategy' defines HOW the server aggregates client updates.
We extend Flower's built-in FedAvg strategy with:
  - Metric logging (accuracy, AUC from clients)
  - FedProx support (sends mu to clients via config)

How FedProx works in Flower:
  - The server sends mu (proximal coefficient) in the fit_config.
  - Each client uses mu to compute the FedProx proximal term during training.
  - The aggregation itself is still standard FedAvg (weighted average).
  - The benefit is in reduced client drift — not in the aggregation step.

References
----------
  - McMahan et al. (2017) "Communication-Efficient Learning of Deep Networks
    from Decentralized Data" [FedAvg]
  - Li et al. (2020) "Federated Optimization in Heterogeneous Networks" [FedProx]
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

# Try importing Flower
try:
    import flwr as fl
    from flwr.common import (
        FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays,
    )
    from flwr.server.client_proxy import ClientProxy
    FLOWER_AVAILABLE = True
except ImportError:
    FLOWER_AVAILABLE = False
    fl = None


def get_fedavg_strategy(
    initial_parameters=None,
    min_fit_clients: int = 2,
    min_evaluate_clients: int = 2,
    min_available_clients: int = 2,
    fit_config: dict = None,
    on_fit_config_fn=None,
    metrics_logger=None,
):
    """
    Build a Flower FedAvg strategy with optional metric logging.

    Parameters
    ----------
    initial_parameters    : Flower Parameters object (initial global model)
    min_fit_clients       : minimum clients that must participate in training
    min_evaluate_clients  : minimum clients for evaluation
    min_available_clients : minimum clients that must be connected
    fit_config            : static config dict to send to all clients each round
    on_fit_config_fn      : dynamic config function(round) -> dict (overrides fit_config)
    metrics_logger        : optional FLLogger instance for recording metrics

    Returns
    -------
    fl.server.strategy.FedAvg instance (or None if Flower not available)
    """
    if not FLOWER_AVAILABLE:
        logger.warning("Flower not available — returning None strategy.")
        return None

    def _fit_config_fn(server_round: int) -> dict:
        """Send training config to clients at the start of each round."""
        if on_fit_config_fn is not None:
            return on_fit_config_fn(server_round)
        return fit_config or {}

    def _evaluate_config_fn(server_round: int) -> dict:
        return {"round": server_round}

    def _fit_metrics_aggregation_fn(metrics: list) -> dict:
        """Aggregate per-client fit metrics using weighted average."""
        total = sum(n for n, _ in metrics)
        agg = {}
        for n, m in metrics:
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    agg[k] = agg.get(k, 0.0) + (n / total) * v
        # Log to FLLogger if provided
        if metrics_logger and "accuracy" in agg:
            logger.info(f"  Aggregated fit metrics: {agg}")
        return agg

    def _evaluate_metrics_aggregation_fn(metrics: list) -> dict:
        """Aggregate per-client evaluation metrics."""
        total = sum(n for n, _ in metrics)
        agg = {}
        for n, m in metrics:
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    agg[k] = agg.get(k, 0.0) + (n / total) * v
        return agg

    return fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        initial_parameters=initial_parameters,
        on_fit_config_fn=_fit_config_fn,
        on_evaluate_config_fn=_evaluate_config_fn,
        fit_metrics_aggregation_fn=_fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=_evaluate_metrics_aggregation_fn,
    )


def get_fedprox_strategy(
    mu: float = 0.01,
    initial_parameters=None,
    min_fit_clients: int = 2,
    min_evaluate_clients: int = 2,
    min_available_clients: int = 2,
    base_fit_config: dict = None,
    metrics_logger=None,
):
    """
    Build a FedProx strategy.

    FedProx uses the same FedAvg aggregation but sends mu to clients
    so they apply the proximal regularisation term during local training.

    Parameters
    ----------
    mu : float — proximal regularisation coefficient (0.001 to 0.5)
         Higher mu → local models stay closer to global (more stability,
         but slower convergence on heterogeneous data).

    All other params same as get_fedavg_strategy().
    """
    if not FLOWER_AVAILABLE:
        logger.warning("Flower not available — returning None strategy.")
        return None

    base_config = base_fit_config or {}

    def _fedprox_config_fn(server_round: int) -> dict:
        """Inject FedProx mu into client config each round."""
        cfg = dict(base_config)
        cfg["use_fedprox"] = True
        cfg["mu"]          = mu
        cfg["round"]       = server_round
        return cfg

    return get_fedavg_strategy(
        initial_parameters=initial_parameters,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        on_fit_config_fn=_fedprox_config_fn,
        metrics_logger=metrics_logger,
    )
