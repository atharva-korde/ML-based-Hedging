import numpy as np
import matplotlib.pyplot as plt
import torch

from hedging import HedgingEnv
from policy import PolicyNNet


def evaluate(env: HedgingEnv, model: PolicyNNet) -> dict:
    # Return summary statistics for the trained policy.
    model.eval()
    with torch.no_grad():
        pnl = env.rollout(model).cpu().numpy()

    return {
        "E[PnL]": float(np.mean(pnl)),
        "Std[PnL]": float(np.std(pnl)),
        "Var[PnL]": float(np.var(pnl)),
        "Min PnL": float(np.min(pnl)),
        "Max PnL": float(np.max(pnl)),
    }


def plot_results(losses: list[float], pnl_ml: np.ndarray, pnl_bs: np.ndarray):
    # A comparison of the two strategies side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # A graph of the loss function
    axes[0].plot(losses, color="#226e36", linewidth=1.5)
    axes[0].set_title("Training Loss  (Var(PnL) − E[PnL])", fontsize=13)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)

    # PnL distributions compared by overlapping histograms
    axes[1].hist(pnl_bs, bins=80, alpha=0.55, label="BS-Delta Hedge", color="#e41e1e")
    axes[1].hist(pnl_ml, bins=80, alpha=0.55, label="ML Hedge", color="#2563eb")
    axes[1].axvline(
        np.mean(pnl_ml),
        color="#2563eb",
        linestyle="--",
        linewidth=1.5,
        label=f"ML mean={np.mean(pnl_ml):.4f}",
    )
    axes[1].axvline(
        np.mean(pnl_bs),
        color="#e41e1e",
        linestyle="--",
        linewidth=1.5,
        label=f"BS mean={np.mean(pnl_bs):.4f}",
    )
    axes[1].set_title("PnL Distribution  (Short call, ML vs BS)", fontsize=13)
    axes[1].set_xlabel("Final PnL")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
