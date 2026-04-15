"""
src/federated/comparison.py
============================
Performance comparison module — runs all 4 training modes side-by-side.

Modes compared:
  1. Centralized (no privacy, upper bound)
  2. FedAvg        (federated, no DP)
  3. FedAvg + DP   (our main approach)
  4. FedProx + DP  (best for Non-IID data)

This module now uses FLEngine as the unified backend, so it correctly
supports both Flower and custom FL loop depending on what's installed.

Usage (from CLI):
    python run_experiment.py

Usage (from code):
    from src.federated.comparison import run_comparison
    rows = run_comparison(banks=["SBI","HDFC","Axis"], num_rounds=8)
"""

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
    verbose: bool = True,
) -> list:
    """
    Run all 4 training modes and return comparison results.

    Parameters
    ----------
    banks       : list of bank names (default: all banks)
    all_data    : dict[bank -> pd.DataFrame] (loaded if None)
    num_rounds  : FL communication rounds
    local_epochs: local epochs per round
    lr          : learning rate
    noise_mult  : DP noise multiplier (sigma)
    max_norm    : gradient clipping norm (C)
    mu          : FedProx proximal coefficient
    fl_backend  : "flower" | "custom"
    dp_backend  : "opacus" | "custom"
    verbose     : print progress to stdout

    Returns
    -------
    list of dicts — one per mode, suitable for pd.DataFrame
    """
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

    if verbose:
        print("\n" + "=" * 65)
        print("  PERFORMANCE COMPARISON — FL Credit Scoring")
        print("=" * 65)
        print(f"  Banks      : {banks}")
        print(f"  FL rounds  : {num_rounds}, Local epochs: {local_epochs}, LR: {lr}")
        print(f"  DP         : noise={noise_mult}, max_norm={max_norm}")
        print(f"  FedProx mu : {mu}")
        print(f"  FL backend : {fl_backend}, DP backend: {dp_backend}")
        print("-" * 65)

    for i, (mode_key, label, privacy_label, data_shared) in enumerate(modes, 1):
        if verbose:
            print(f"  [{i}/4] Running {label} ...")

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

            if verbose:
                eps_str = f"  epsilon={eps:.3f}" if eps > 0 else ""
                print(
                    f"  {label:25s} "
                    f"acc={result['final_acc']:.4f}  "
                    f"auc={result['final_auc']:.4f}"
                    f"{eps_str}"
                )

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

    if verbose:
        print("-" * 65)
        print("  Comparison complete.\n")

    return results
