"""
src/utils/helpers.py
====================
Consolidated training and evaluation utilities.

This is the canonical module for:
  - local_train()   — one bank, one FL round (supports FedAvg + FedProx + DP)
  - evaluate_model() — accuracy + AUC-ROC on a dataset

Previously, equivalent functions existed in BOTH utils/helper.py and
federated/fl.py — that duplication is now eliminated here.

All federated training code imports from this module.
"""

import copy
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

from src.data.data_generator import FEATURE_NAMES

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# FedProx proximal regularisation term
# ─────────────────────────────────────────────────────────────────────────────

def fedprox_loss(
    local_model: nn.Module,
    global_model: nn.Module,
    mu: float,
) -> torch.Tensor:
    """
    Compute the FedProx proximal regularisation term.

    FedProx (Li et al., 2020) adds a quadratic penalty to the local loss:
        L_total = L_CE + (mu/2) * ||w_local - w_global||^2

    This 'tether' prevents local models from drifting too far from the global
    model during training, which is the main cause of performance degradation
    in Non-IID federated settings (also known as 'client drift').

    Parameters
    ----------
    local_model  : model being trained (weights change each step)
    global_model : frozen snapshot of the global model (reference point)
    mu           : float — regularisation strength (0.01 to 0.1 typically)

    Returns
    -------
    torch.Tensor — scalar proximal term to add to the loss
    """
    prox = torch.tensor(0.0)
    for (_, p_local), (_, p_global) in zip(
        local_model.named_parameters(),
        global_model.named_parameters(),
    ):
        prox = prox + torch.norm(p_local - p_global.detach()) ** 2
    return (mu / 2.0) * prox


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL TRAIN — one bank, one FL round
# ─────────────────────────────────────────────────────────────────────────────

def local_train(
    model: nn.Module,
    df,
    local_epochs: int,
    lr: float,
    use_dp: bool,
    noise_mult: float,
    max_norm: float,
    batch_size: int = 64,
    use_fedprox: bool = False,
    mu: float = 0.01,
    global_model: nn.Module = None,
    dp_backend: str = "custom",
):
    """
    Train model on a single bank's dataset for one FL round.

    Supports:
      - Standard FedAvg training (use_fedprox=False, use_dp=False)
      - FedAvg + DP             (use_dp=True, dp_backend="custom"/"opacus")
      - FedProx                 (use_fedprox=True)
      - FedProx + DP            (both flags True)

    Parameters
    ----------
    model        : local copy of global model
    df           : pd.DataFrame — bank's dataset
    local_epochs : int   — training epochs per FL round
    lr           : float — Adam learning rate
    use_dp       : bool  — apply Differential Privacy
    noise_mult   : float — DP noise multiplier (σ)
    max_norm     : float — gradient clipping norm (C)
    batch_size   : int   — mini-batch size
    use_fedprox  : bool  — add FedProx proximal term to loss
    mu           : float — FedProx regularisation strength
    global_model : nn.Module — global model snapshot (needed for FedProx)
    dp_backend   : str   — "custom" or "opacus" (ignored if use_dp=False)

    Returns
    -------
    (avg_loss, n_samples, scaler)
      avg_loss   : float        — mean training loss over all steps
      n_samples  : int          — number of training samples
      scaler     : StandardScaler — fitted scaler (needed for evaluation)
    """
    # ── Data preparation ─────────────────────────────────────────────────
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURE_NAMES].values).astype(np.float32)
    y = df["default"].values.astype(np.float32)

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()

    # ── Opacus setup (if requested) ──────────────────────────────────────
    if use_dp and dp_backend == "opacus":
        from src.privacy.dp_manager import DPManager
        dm = DPManager(
            backend="opacus",
            noise_multiplier=noise_mult,
            max_grad_norm=max_norm,
        )
        try:
            model, optimizer, loader = dm.setup(model, optimizer, loader)
        except Exception as e:
            logger.warning(f"Opacus setup failed ({e}), falling back to custom DP.")
            dp_backend = "custom"

    # ── FedProx: freeze global model snapshot ───────────────────────────
    if use_fedprox and global_model is not None:
        global_snapshot = copy.deepcopy(global_model)
        for p in global_snapshot.parameters():
            p.requires_grad = False
    else:
        global_snapshot = None

    # ── Training loop ────────────────────────────────────────────────────
    model.train()
    total_loss = 0.0
    steps      = 0

    for _ in range(local_epochs):
        for Xb, yb in loader:
            optimizer.zero_grad()

            out  = model(Xb)
            loss = criterion(out, yb)

            # FedProx: add proximal regularisation term
            if use_fedprox and global_snapshot is not None:
                loss = loss + fedprox_loss(model, global_snapshot, mu)

            loss.backward()

            # Custom DP: clip + noise (applied after backward, before step)
            if use_dp and dp_backend == "custom":
                from src.privacy.dp_custom import clip_gradients, add_dp_noise
                clip_gradients(model, max_norm)
                add_dp_noise(model, noise_mult, max_norm, len(Xb))

            optimizer.step()
            total_loss += loss.item()
            steps += 1

    avg_loss = total_loss / max(steps, 1)
    return avg_loss, len(df), scaler


# ─────────────────────────────────────────────────────────────────────────────
# MODEL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model: nn.Module,
    df,
    scaler: StandardScaler,
) -> tuple:
    """
    Evaluate model on a dataset.

    Parameters
    ----------
    model  : PyTorch model (eval mode applied internally)
    df     : pd.DataFrame — dataset with FEATURE_NAMES + "default" columns
    scaler : StandardScaler fitted on the same bank's training data

    Returns
    -------
    (accuracy, auc_roc) — both floats in [0, 1]
    """
    X = scaler.transform(df[FEATURE_NAMES].values).astype(np.float32)
    y = df["default"].values

    model.eval()
    with torch.no_grad():
        probs = model(torch.from_numpy(X)).numpy()

    preds = (probs > 0.5).astype(int)
    acc   = accuracy_score(y, preds)

    try:
        auc = roc_auc_score(y, probs)
    except Exception:
        auc = 0.5   # fallback if only one class present

    return float(acc), float(auc)


# ─────────────────────────────────────────────────────────────────────────────
# FEDERATED AVERAGING (FedAvg)
# ─────────────────────────────────────────────────────────────────────────────

def fed_avg(client_weights: list, client_sizes: list) -> list:
    """
    Aggregate client model weights using FedAvg (weighted by dataset size).

    FedAvg (McMahan et al., 2017) computes:
        w_global = sum_k (n_k / N) * w_k
    where n_k is the number of samples at client k and N = sum(n_k).

    Parameters
    ----------
    client_weights : list of lists of torch.Tensors
                     — one weight list per client
    client_sizes   : list of int — n_samples per client

    Returns
    -------
    list of torch.Tensors — aggregated global weights
    """
    total = sum(client_sizes)
    avg   = [torch.zeros_like(w) for w in client_weights[0]]
    for weights, size in zip(client_weights, client_sizes):
        frac = size / total
        for a, w in zip(avg, weights):
            a.add_(w * frac)
    return avg
