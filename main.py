import torch
import numpy as np

from var_config import *
from montecarlosimulator import simulate_paths_gbm
from hedging import HedgingEnv
from benchmark_hedging import BSDeltaHedger
from policy import PolicyNNet
from training import train
from evals import evaluate, plot_results


def main():
    print("Simulating training paths...")
    train_paths = simulate_paths_gbm(S0, r, sigma, T, steps, paths, seed=42)
    test_paths  = simulate_paths_gbm(S0, r, sigma, T, steps, tests,  seed=73)

    print("Building environments...")
    train_env = HedgingEnv(train_paths, K, r, sigma, T, transaction_cost)
    test_env  = HedgingEnv(test_paths,  K, r, sigma, T, transaction_cost)

    print("Initiating model...")
    model = PolicyNNet(hidden=64)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training model
    print(f"\nTraining for {n_epochs} epochs …\n")
    losses = train(train_env, model, n_epochs=300, lr=5e-4, c=const)

    # Evaluations
    model.eval()
    with torch.no_grad():
        pnl_ml = test_env.rollout(model).numpy()

    stats_ml = evaluate(test_env, model)
    print("\n── ML Hedger (test set) ──")
    for k, v in stats_ml.items():
        print(f"  {k:<15}: {v:+.4f}")

    # Evaluate with benchmark
    bs_hedger = BSDeltaHedger(test_env)
    pnl_bs    = bs_hedger.rollout()

    print("\n── BS Delta Hedge (test set) ──")
    print(f"  {'E[PnL]':<15}: {np.mean(pnl_bs):+.4f}")
    print(f"  {'Std[PnL]':<15}: {np.std(pnl_bs):+.4f}")
    print(f"  {'Var[PnL]':<15}: {np.var(pnl_bs):+.4f}")
    print(f"  {'Min[PnL]':<15}: {np.min(pnl_bs):+.4f}")
    print(f"  {'Max[PnL]':<15}: {np.max(pnl_bs):+.4f}")

    # Visualizations
    plot_results(losses, pnl_ml, pnl_bs)


if __name__ == "__main__":
    main()
