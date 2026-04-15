"""
src/federated/flower_client.py
==============================
Flower NumPyClient implementation for each bank.

In the Flower framework, each client:
  1. Receives global model weights from the server (set_parameters)
  2. Trains locally on its own data (fit)
  3. Returns updated weights + metrics to the server (fit return)
  4. Evaluates the global model locally (evaluate)

Non-IID handling:
  Each bank's client is initialised with ONLY that bank's dataset partition.
  No data is ever shared — only model weights travel between client and server.
  This is the core privacy guarantee of Federated Learning.

The CreditFlowerClient supports:
  - FedAvg (plain local training)
  - FedAvg + DP (custom or Opacus)
  - FedProx + DP (proximal regularisation for Non-IID stability)
"""

import copy
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.models.model import CreditNet, weights_to_numpy, numpy_to_weights
from src.utils.helpers import local_train, evaluate_model, fed_avg
from src.data.data_generator import FEATURE_NAMES, BANK_PROFILES

logger = logging.getLogger(__name__)

# Try importing Flower — fail gracefully
try:
    import flwr as fl
    FLOWER_AVAILABLE = True
except ImportError:
    FLOWER_AVAILABLE = False
    fl = None


class CreditFlowerClient:
    """
    Flower NumPyClient for one bank in the federated credit scoring system.

    Each bank gets its own client instance with its own local dataset.
    Raw data never leaves this client — only model weights are transmitted.

    Parameters
    ----------
    bank_name   : str          — bank identifier (e.g. "SBI", "HDFC")
    bank_data   : pd.DataFrame — local dataset (Non-IID partition)
    config      : dict         — training configuration
    """

    def __init__(self, bank_name: str, bank_data, config: dict):
        self.bank_name  = bank_name
        self.bank_data  = bank_data
        self.config     = config
        self.model      = CreditNet(input_dim=10)

        # Fit scaler on local data (never shared)
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.scaler.fit(bank_data[FEATURE_NAMES].values)

    def get_parameters(self) -> List[np.ndarray]:
        """Extract model weights as numpy arrays for transmission to server."""
        return weights_to_numpy(self.model)

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Load global weights received from server into local model."""
        numpy_to_weights(self.model, parameters)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: dict,
    ) -> Tuple[List[np.ndarray], int, dict]:
        """
        Receive global weights, train locally, return updated weights.

        This is the core FL training step. Flower calls this once per round.

        Parameters
        ----------
        parameters : global model weights from server
        config     : round configuration from server (epochs, dp settings, etc.)

        Returns
        -------
        (updated_weights, n_samples, metrics_dict)
        """
        self.set_parameters(parameters)

        # Merge global config with client config
        local_epochs = config.get("local_epochs", self.config.get("local_epochs", 2))
        lr           = config.get("lr", self.config.get("lr", 0.001))
        use_dp       = config.get("use_dp", self.config.get("use_dp", False))
        noise_mult   = config.get("noise_mult", self.config.get("noise_mult", 1.1))
        max_norm     = config.get("max_norm", self.config.get("max_norm", 1.0))
        use_fedprox  = config.get("use_fedprox", self.config.get("use_fedprox", False))
        mu           = config.get("mu", self.config.get("mu", 0.01))
        dp_backend   = config.get("dp_backend", self.config.get("dp_backend", "custom"))

        # Keep a copy of global model for FedProx proximal term
        global_model_ref = copy.deepcopy(self.model) if use_fedprox else None

        loss, n_samples, _ = local_train(
            model        = self.model,
            df           = self.bank_data,
            local_epochs = local_epochs,
            lr           = lr,
            use_dp       = use_dp,
            noise_mult   = noise_mult,
            max_norm     = max_norm,
            use_fedprox  = use_fedprox,
            mu           = mu,
            global_model = global_model_ref,
            dp_backend   = dp_backend,
        )

        # Compute local accuracy and AUC for reporting
        acc, auc = evaluate_model(self.model, self.bank_data, self.scaler)

        metrics = {
            "bank":     self.bank_name,
            "loss":     float(loss),
            "accuracy": float(acc),
            "auc":      float(auc),
        }

        logger.info(
            f"[{self.bank_name}] Round complete — "
            f"acc={acc:.4f} auc={auc:.4f} loss={loss:.4f}"
        )

        return self.get_parameters(), n_samples, metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: dict,
    ) -> Tuple[float, int, dict]:
        """
        Evaluate global model on local data.

        Flower calls this after each aggregation round to collect
        per-client evaluation metrics.

        Returns
        -------
        (loss, n_samples, metrics_dict)
        """
        self.set_parameters(parameters)
        acc, auc = evaluate_model(self.model, self.bank_data, self.scaler)

        # Approximate cross-entropy loss from accuracy (or compute directly)
        loss = 1.0 - acc

        return float(loss), len(self.bank_data), {
            "accuracy": float(acc),
            "auc":      float(auc),
            "bank":     self.bank_name,
        }


def make_flower_client_fn(all_data: dict, config: dict):
    """
    Return a Flower client factory function.

    Flower's start_simulation() calls this function with a client_id
    to get the appropriate client for each round.

    Parameters
    ----------
    all_data : dict[bank_name -> pd.DataFrame]
    config   : training configuration dict

    Returns
    -------
    Callable[str -> CreditFlowerClient]
    """
    banks = list(all_data.keys())

    if FLOWER_AVAILABLE:
        class _FlwrClient(fl.client.NumPyClient):
            def __init__(self, cid: str):
                bank_name = banks[int(cid) % len(banks)]
                self._client = CreditFlowerClient(bank_name, all_data[bank_name], config)

            def get_parameters(self, config):
                return self._client.get_parameters()

            def fit(self, parameters, config):
                return self._client.fit(parameters, config)

            def evaluate(self, parameters, config):
                return self._client.evaluate(parameters, config)

        def client_fn(cid: str) -> fl.client.NumPyClient:
            return _FlwrClient(cid)

    else:
        # Fallback when Flower is not installed
        def client_fn(cid: str):
            bank_name = banks[int(cid) % len(banks)]
            return CreditFlowerClient(bank_name, all_data[bank_name], config)

    return client_fn
