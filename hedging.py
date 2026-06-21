import numpy as np
import torch

from torch import Tensor
from numpy.typing import NDArray
from black_scholes import bs_call_price
from policy import PolicyNNet


class HedgingEnv:

    # A class enabling ML-based Delta-hedging
    # Functions will return tensors as outputs (for autograd)

    def __init__(
        self,
        S: np.NDArray[np.float64],
        K: float,
        r: float,
        sigma: float,
        T: int,
        cost: float,
    ) -> None:

        self.S = torch.tensor(
            S, dtype=torch.float32
        )  # The array of simulated price paths needs to be a tensor
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.cost = cost

        # Get stock price at time t=0 from the array of paths
        S0_np = S[:, 0]
        # Compute the inital cash-in-hand, given by the price of the option
        self.premium = torch.tensor(
            bs_call_price(S0_np, K, r, sigma, T), dtype=torch.float32
        )

        self.steps = (
            S.shape[1] - 1
        )  # Remember that the -1 is because the first column is the inital price S0
        self.paths = S.shape[0]
        self.dt = T / self.steps
        self.times = np.linspace(0, T, self.steps + 1)

        # Compute the time to maturity at each step
        self.tau = torch.tensor(T - self.times[:-1], dtype=torch.float32)

    def _normalise_inputs(self, S: Tensor, delta: Tensor, tau: Tensor) -> Tensor:
        """ Normalises inputs to mean=0, variance=1 and stacks them """
        S_norm = S / self.S[:, 0]
        tau_norm = tau / self.T
        return torch.stack([S_norm, delta, tau_norm], dim=1)

    def rollout(self, model: PolicyNNet) -> torch.Tensor:

        model_device = next(model.parameters()).device
        S = self.S.to(model_device)

        # Pre-compute r times dt
        r_dt = torch.tensor(self.r * self.dt, device=model_device)

        pnl = self.premium.to(model_device).clone()  # Start with collected premium
        delta = torch.zeros(self.paths, device=model_device)  # Initial hedge

        for t in range(self.steps):
            tau_t = self.tau[t].expand(self.paths).to(model_device)
            x = self._normalise_inputs(S[:, t], delta, tau_t)
            new_delta = model(x).squeeze(1)  # (n_paths,)

            # Cost of rebalancing: Buy (new_delta - delta) shares at S_t and account
            # for transaction costs of abs(new_delta - delta) * stock price
            trade = (new_delta - delta) * S[:, t]
            pnl -= trade
            pnl -= self.cost * torch.abs(new_delta - delta) * S[:, t]

            # Carry cash at risk-free rate for one step
            pnl *= torch.exp(r_dt)

            delta = new_delta  # Update delta

        # Liquidate stock position at maturity
        pnl += delta * S[:, -1]

        # Pay the call option payoff
        payoff = torch.clamp(S[:, -1] - self.K, min=0.0)
        pnl -= payoff

        return pnl
