import logging
import numpy as np
from src.federated.fl_engine import FLEngine

logger = logging.getLogger(__name__)


def run_comparison(
    banks: list = None,
    all_data: dict = None,
    num_rounds: int = 8,
    local_epochs: int = 2,
    lr: float = 0.001,
    noise_mult: float = 1.1,
    max_norm: float = 1.0,
    mu: float = 0.01,
    fl_backend: str = "custom",
    dp_backend: str = "custom",
) -> list:
   
    if all_data is None:
        from src.data.data_generator import load_all_data, BANK_PROFILES
        all_data = load_all_data()
        if banks is None:
            banks = list(BANK_PROFILES.keys())
    elif banks is None:
        banks = list(all_data.keys())

    engine = FLEngine()
    results = []

    modes = [
        ("centralized", "Centralized (no privacy)", "None",           "All raw data"),
        ("fedavg",      "FedAvg",                   "None",           "Weights only"),
        ("fedavg_dp",   "FedAvg + DP",              "DP",             "Weights only"),
        ("fedprox_dp",  "FedProx + DP",             "DP + FedProx",   "Weights only"),
    ]

    for i, (mode_key, label, privacy_label, data_shared) in enumerate(modes, 1):

        try:
            result = engine.run(
                mode         = mode_key,
                banks        = banks,
                all_data     = all_data,
                num_rounds   = num_rounds,
                local_epochs = local_epochs,
                lr           = lr,
                noise_mult   = noise_mult,
                max_norm     = max_norm,
                mu           = mu,
                fl_backend   = fl_backend,
                dp_backend   = dp_backend,
            )

            eps = result.get("final_epsilon", 0.0) or 0.0
            row = {
                "Method":     label,
                "Accuracy":   round(result["final_acc"], 4),
                "AUC-ROC":    round(result["final_auc"], 4),
                "Privacy":    f"DP (epsilon~{eps:.2f})" if eps > 0 else privacy_label,
                "Data Shared": data_shared,
                "Epsilon":    eps if eps > 0 else None,
            }
            results.append(row)

        except Exception as e:
            logger.error(f"Error running mode {mode_key}: {e}")
            results.append({
                "Method":     label,
                "Accuracy":   0.0,
                "AUC-ROC":    0.0,
                "Privacy":    privacy_label,
                "Data Shared": data_shared,
                "Epsilon":    None,
                "Error":      str(e),
            })

    return results
