
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import copy

from data.data_generator import FEATURE_NAMES
from models.model import CreditNet, get_weights, set_weights

def train_centralized_baseline(
    all_data: dict,
    selected_banks: list,
    epochs: int = 15,
    lr: float = 0.001,
    batch_size: int = 64,
    progress_cb=None,          # optional callback(epoch, total, loss, acc, auc)
) -> dict:
    """
    Pool all selected banks' data → train one model centrally (no DP, no FL).
    Returns metrics dict with per-epoch history + final scaler + model.
    This simulates the 'ideal upper bound' if all data were shared freely.
    """

   
    frames = [all_data[b] for b in selected_banks if b in all_data]
    df_all = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=42)

    scaler = StandardScaler()
    X = scaler.fit_transform(df_all[FEATURE_NAMES].values).astype(np.float32)
    y = df_all["default"].values.astype(np.float32)

   
    split    = int(0.8 * len(X))
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=batch_size, shuffle=True
    )

    model     = CreditNet(input_dim=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = {"epoch": [], "loss": [], "acc": [], "auc": [], "val_acc": [], "val_auc": []}

    for ep in range(1, epochs + 1):
        model.train()
        total_loss, steps = 0.0, 0

        for Xb, yb in tr_loader:
            optimizer.zero_grad()
            out  = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1

        scheduler.step()

        model.eval()
        with torch.no_grad():
            tr_probs = model(torch.from_numpy(X_tr)).numpy()
            vl_probs = model(torch.from_numpy(X_val)).numpy()

        tr_acc = accuracy_score(y_tr, (tr_probs > 0.5).astype(int))
        vl_acc = accuracy_score(y_val, (vl_probs > 0.5).astype(int))
        try:
            tr_auc = roc_auc_score(y_tr, tr_probs)
            vl_auc = roc_auc_score(y_val, vl_probs)
        except Exception:
            tr_auc = vl_auc = 0.5

        avg_loss = total_loss / max(steps, 1)
        history["epoch"].append(ep)
        history["loss"].append(round(avg_loss, 6))
        history["acc"].append(round(tr_acc, 6))
        history["auc"].append(round(tr_auc, 6))
        history["val_acc"].append(round(vl_acc, 6))
        history["val_auc"].append(round(vl_auc, 6))

        if progress_cb:
            progress_cb(ep, epochs, avg_loss, vl_acc, vl_auc)

    return {
        "model":       model,
        "scaler":      scaler,
        "history":     history,
        "final_acc":   history["val_acc"][-1],
        "final_auc":   history["val_auc"][-1],
        "n_samples":   len(df_all),
        "banks_used":  selected_banks,
    }


def train_local_only_baselines(
    all_data: dict,
    selected_banks: list,
    epochs: int = 10,
    lr: float = 0.001,
) -> dict:
    
    results = {}

    for bank in selected_banks:
        df   = all_data[bank]
        sc   = StandardScaler()
        X    = sc.fit_transform(df[FEATURE_NAMES].values).astype(np.float32)
        y    = df["default"].values.astype(np.float32)

        split = int(0.8 * len(X))
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]

        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
            batch_size=64, shuffle=True
        )

        model     = CreditNet(input_dim=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.BCELoss()

        for _ in range(epochs):
            model.train()
            for Xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            vl_probs = model(torch.from_numpy(X_val)).numpy()

        val_acc = accuracy_score(y_val, (vl_probs > 0.5).astype(int))
        try:
            val_auc = roc_auc_score(y_val, vl_probs)
        except Exception:
            val_auc = 0.5

        results[bank] = {
            "model":   model,
            "scaler":  sc,
            "val_acc": round(val_acc, 6),
            "val_auc": round(val_auc, 6),
            "n":       len(df),
        }

    return results
