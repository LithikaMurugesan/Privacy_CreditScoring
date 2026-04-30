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

        assert mode in ("centralized", "fedavg", "fedavg_dp", "fedprox_dp")

        logger.info(f"FL run → mode={mode}, backend={fl_backend}, dp={dp_backend}")

        if mode == "centralized":
            return self._run_centralized(banks, all_data, num_rounds, local_epochs, lr)

        use_dp = mode in ("fedavg_dp", "fedprox_dp")
        use_fedprox = mode == "fedprox_dp"

        config = {
            "local_epochs": local_epochs,
            "lr": lr,
            "use_dp": use_dp,
            "noise_mult": noise_mult,
            "max_norm": max_norm,
            "dp_backend": dp_backend if use_dp else "none",
        }

        return self._run_custom_loop(
            banks, all_data, num_rounds,
            config, use_fedprox, mu,
            metrics_logger, progress_cb
        )

    def _run_centralized(self, banks, all_data, num_rounds, local_epochs, lr):
        from src.federated.baseline import train_centralized_baseline

        epochs = max(10, num_rounds * local_epochs)

        result = train_centralized_baseline(all_data, banks, epochs=epochs, lr=lr)

        history = result.get("history", {})

        return {
            "mode": "centralized",
            "final_acc": result.get("final_acc", 0.0),
            "final_auc": result.get("final_auc", 0.0),
            "final_epsilon": 0.0,
            "model": result["model"],
            "acc_history": history.get("val_acc", [result.get("final_acc", 0.0)]),
            "auc_history": history.get("val_auc", [result.get("final_auc", 0.0)]),
            "epsilon_history": [0.0] * epochs,

            "bank_history": {},
        }

    def _run_custom_loop(
        self,
        banks,
        all_data,
        num_rounds,
        cfg,
        use_fedprox,
        mu,
        metrics_logger,
        progress_cb
    ):

        data_subset = {b: all_data[b] for b in banks if b in all_data}

        global_model = CreditNet(input_dim=10)

        scalers = {
            b: StandardScaler().fit(data_subset[b][FEATURE_NAMES].values)
            for b in banks
        }

        acc_history = []
        auc_history = []
        epsilon_history = []

        bank_history = {
            b: {"acc": [], "auc": [], "loss": []}
            for b in banks
        }

        for rnd in range(1, num_rounds + 1):

            client_weights = []
            client_sizes = []
            for b in banks:

                local_model = copy.deepcopy(global_model)

                loss, n, _ = local_train(
                    model=local_model,
                    df=data_subset[b],
                    local_epochs=cfg["local_epochs"],
                    lr=cfg["lr"],
                    use_dp=cfg["use_dp"],
                    noise_mult=cfg["noise_mult"],
                    max_norm=cfg["max_norm"],
                    use_fedprox=use_fedprox,
                    mu=mu,
                    global_model=global_model if use_fedprox else None,
                    dp_backend=cfg["dp_backend"],
                )

                acc, auc = evaluate_model(local_model, data_subset[b], scalers[b])

                bank_history[b]["acc"].append(acc)
                bank_history[b]["auc"].append(auc)
                bank_history[b]["loss"].append(loss)

                client_weights.append(get_weights(local_model))
                client_sizes.append(n)

                eps_b = 0.0
                if cfg["use_dp"]:
                    eps_b = compute_epsilon(
                        cfg["noise_mult"],
                        64 / max(n, 1),
                        cfg["local_epochs"] * rnd
                    )

                if metrics_logger:
                    metrics_logger.log_client(rnd, b, acc, auc, loss, n, eps_b)

            set_weights(global_model, fed_avg(client_weights, client_sizes))

            g_accs, g_aucs = [], []

            for b in banks:
                a, au = evaluate_model(global_model, data_subset[b], scalers[b])
                g_accs.append(a)
                g_aucs.append(au)

            g_acc = float(np.mean(g_accs)) if g_accs else 0.0
            g_auc = float(np.mean(g_aucs)) if g_aucs else 0.0

            eps = 0.0
            if cfg["use_dp"]:
                eps = compute_epsilon(cfg["noise_mult"], 0.5, rnd)

            acc_history.append(g_acc)
            auc_history.append(g_auc)
            epsilon_history.append(eps)

            if metrics_logger:
                metrics_logger.log_global(rnd, g_acc, g_auc, eps)

            if progress_cb:
                progress_cb(rnd, num_rounds, {
                    "acc": g_acc,
                    "auc": g_auc,
                    "eps": eps,
                    "bank_history": bank_history
                })

            logger.info(f"Round {rnd}: acc={g_acc:.4f}, auc={g_auc:.4f}")

        return {
            "final_acc": acc_history[-1] if acc_history else 0.0,
            "final_auc": auc_history[-1] if auc_history else 0.0,
            "final_epsilon": epsilon_history[-1] if epsilon_history else 0.0,

            "model": global_model,
            "acc_history": acc_history if acc_history else [0.0],
            "auc_history": auc_history if auc_history else [0.0],
            "epsilon_history": epsilon_history if epsilon_history else [0.0],

            "bank_history": bank_history,
        }
