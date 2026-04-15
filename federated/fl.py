# ═════════════════════════════════════════════════════════════════════════════
# 4. LOCAL TRAINING  (one bank, one FL round)
#    Supports both FedAvg (standard) and FedProx (with proximal term).
# ═════════════════════════════════════════════════════════════════════════════
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from data.data_generator import FEATURE_NAMES
from privacy.dp import *


# ─────────────────────────────────────────────────────────────────────────────
# FedProx helper — computes the proximal regularisation term:
#   (mu / 2) * ||w_local - w_global||^2
# This term penalises how far the local model drifts from the global model,
# which is especially useful on heterogeneous (non-IID) data distributions.
# ─────────────────────────────────────────────────────────────────────────────
def fedprox_loss(local_model, global_model, mu: float) -> torch.Tensor:
    """Return the FedProx proximal term (scalar tensor)."""
    prox = torch.tensor(0.0)
    for (_, p_local), (_, p_global) in zip(
        local_model.named_parameters(), global_model.named_parameters()
    ):
        prox += torch.norm(p_local - p_global.detach()) ** 2
    return (mu / 2.0) * prox


# ─────────────────────────────────────────────────────────────────────────────
# Local training function
#   use_fedprox : if True, adds proximal term to BCE loss (FedProx behaviour)
#   mu          : FedProx regularisation strength (ignored when use_fedprox=False)
#   global_model: reference to the current global model (needed for FedProx)
# ─────────────────────────────────────────────────────────────────────────────
def local_train(
    model, df, local_epochs, lr, use_dp, noise_mult, max_norm,
    batch_size=64,
    use_fedprox: bool = False,
    mu: float = 0.01,
    global_model=None,
):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURE_NAMES].values).astype(np.float32)
    y = df["default"].values.astype(np.float32)

    dataset   = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()

    # Keep a frozen snapshot of the global model for FedProx distance computation
    if use_fedprox and global_model is not None:
        global_snapshot = copy.deepcopy(global_model)
        for p in global_snapshot.parameters():
            p.requires_grad = False
    else:
        global_snapshot = None

    model.train()
    total_loss = 0.0
    steps      = 0

    for _ in range(local_epochs):
        for Xb, yb in loader:
            optimizer.zero_grad()
            out  = model(Xb)
            loss = criterion(out, yb)

            # ── FedProx: add proximal term ──────────────────────────────────
            if use_fedprox and global_snapshot is not None:
                loss = loss + fedprox_loss(model, global_snapshot, mu)

            loss.backward()

            if use_dp:
                clip_gradients(model, max_norm)
                add_dp_noise(model, noise_mult, max_norm, len(Xb))

            optimizer.step()
            total_loss += loss.item()
            steps += 1

    return total_loss / max(steps, 1), len(df), scaler


def evaluate_model(model, df, scaler):
    X = scaler.transform(df[FEATURE_NAMES].values).astype(np.float32)
    y = df["default"].values

    model.eval()
    with torch.no_grad():
        probs = model(torch.from_numpy(X)).numpy()
    preds = (probs > 0.5).astype(int)

    acc = accuracy_score(y, preds)
    try:
        auc = roc_auc_score(y, probs)
    except Exception:
        auc = 0.5
    return acc, auc


# ═════════════════════════════════════════════════════════════════════════════
# 5. FEDERATED AVERAGING  (FedAvg)  — unchanged, works with both FedAvg & FedProx
# ═════════════════════════════════════════════════════════════════════════════

def fed_avg(client_weights, client_sizes):
    total = sum(client_sizes)
    avg   = [torch.zeros_like(w) for w in client_weights[0]]
    for weights, size in zip(client_weights, client_sizes):
        frac = size / total
        for a, w in zip(avg, weights):
            a.add_(w * frac)
    return avg