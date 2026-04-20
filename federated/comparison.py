
import copy
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from data.data_generator import load_all_data, BANK_PROFILES, FEATURE_NAMES
from models.model import CreditNet, get_weights, set_weights
from federated.fl import local_train, evaluate_model, fed_avg
from federated.baseline import train_centralized_baseline
from privacy.dp import compute_epsilon


def _run_fl(
    all_data, banks, num_rounds=8, local_epochs=2, lr=0.001,
    use_dp=False, noise_mult=1.1, max_norm=1.0,
    use_fedprox=False, mu=0.01,
    verbose=True,
):
    label = (
        "FedProx" + (" + DP" if use_dp else "")
        if use_fedprox else
        "FedAvg"  + (" + DP" if use_dp else "")
    )

    global_model = CreditNet(input_dim=10)
    scalers = {}
    for b in banks:
        sc = StandardScaler()
        sc.fit(all_data[b][FEATURE_NAMES].values)
        scalers[b] = sc

    for rnd in range(1, num_rounds + 1):
        client_weights, client_sizes = [], []
        for b in banks:
            local_model = copy.deepcopy(global_model)
            _, n, _ = local_train(
                local_model, all_data[b],
                local_epochs=local_epochs, lr=lr,
                use_dp=use_dp, noise_mult=noise_mult, max_norm=max_norm,
                use_fedprox=use_fedprox, mu=mu,
                global_model=global_model,          # needed for proximal term
            )
            client_weights.append(get_weights(local_model))
            client_sizes.append(n)

        set_weights(global_model, fed_avg(client_weights, client_sizes))

    accs, aucs = [], []
    for b in banks:
        acc, auc = evaluate_model(global_model, all_data[b], scalers[b])
        accs.append(acc)
        aucs.append(auc)

    final_acc = float(np.mean(accs))
    final_auc = float(np.mean(aucs))

    if use_dp:
        sr  = 64 / BANK_PROFILES[banks[0]]["n"]
        st  = local_epochs * (BANK_PROFILES[banks[0]]["n"] // 64) * num_rounds
        eps = compute_epsilon(noise_mult, sr, st)
    else:
        eps = None

    if verbose:
        eps_str = f"  ε = {eps:.3f}" if eps is not None else ""
        print(f"  [{label:20s}]  acc={final_acc:.4f}  auc={final_auc:.4f}{eps_str}")

    return {
        "Method":       label,
        "Accuracy":     round(final_acc, 4),
        "AUC-ROC":      round(final_auc, 4),
        "Privacy":      f"DP (ε≈{eps:.2f})" if eps else "None",
        "Data Shared":  "Weights only",
        "Epsilon":      eps,
    }

def run_comparison(
    banks=None,
    num_rounds=8,
    local_epochs=2,
    lr=0.001,
    noise_mult=1.1,
    max_norm=1.0,
    mu=0.01,
    verbose=True,
) -> list:

    all_data = load_all_data()
    if banks is None:
        banks = list(BANK_PROFILES.keys())   # all 5 banks

    if verbose:
        print("\n" + "=" * 60)
        print("  PERFORMANCE COMPARISON — FL Credit Scoring")
        print("=" * 60)
        print(f"  Banks: {banks}")
        print(f"  FL rounds={num_rounds}  epochs={local_epochs}  lr={lr}")
        print(f"  DP: noise={noise_mult}  max_norm={max_norm}  FedProx: mu={mu}")
        print("-" * 60)

    results = []

    if verbose:
        print("  [1/4] Centralized training …")
    epochs_c = max(10, num_rounds * local_epochs)
    c = train_centralized_baseline(all_data, banks, epochs=epochs_c, lr=lr)
    results.append({
        "Method":      "Centralized (no privacy)",
        "Accuracy":    round(c["final_acc"], 4),
        "AUC-ROC":     round(c["final_auc"], 4),
        "Privacy":     "None",
        "Data Shared": "All raw data",
        "Epsilon":     None,
    })
    if verbose:
        print(f"  {'Centralized':20s}  acc={c['final_acc']:.4f}  auc={c['final_auc']:.4f}  (upper bound)")

    if verbose:
        print("  [2/4] FedAvg (no DP) …")
    results.append(_run_fl(
        all_data, banks, num_rounds, local_epochs, lr,
        use_dp=False, use_fedprox=False, verbose=verbose,
    ))

    if verbose:
        print("  [3/4] FedAvg + DP …")
    results.append(_run_fl(
        all_data, banks, num_rounds, local_epochs, lr,
        use_dp=True, noise_mult=noise_mult, max_norm=max_norm,
        use_fedprox=False, verbose=verbose,
    ))
    if verbose:
        print("  [4/4] FedProx + DP …")
    results.append(_run_fl(
        all_data, banks, num_rounds, local_epochs, lr,
        use_dp=True, noise_mult=noise_mult, max_norm=max_norm,
        use_fedprox=True, mu=mu, verbose=verbose,
    ))

    if verbose:
        print("-" * 60)
        print("  Done. Returning results list.\n")

    return results

if __name__ == "__main__":
    import pandas as pd
    rows = run_comparison(verbose=True)
    df = pd.DataFrame(rows)[["Method", "Accuracy", "AUC-ROC", "Privacy", "Data Shared"]]
    print(df.to_string(index=False))
