import numpy as np

from blackscholes import bs_delta
from hedging import HedgingEnv
from numpy.typing import NDArray


class BSDeltaHedger:
    # A class enabling deterministic BS-Delta hedging

    def __init__(self, env: HedgingEnv) -> None:
        self.env = env

    def rollout(self) -> np.NDArray[np.float64]:

        env = self.env
        S = env.S.numpy()
        pnl = env.premium.numpy().copy()

        delta = np.zeros(env.paths)  # Initial hedge

        # Use the same algorithm as in the ML-hedger here: it is deterministic
        for t in range(env.steps):
            tau_t = env.T - env.times[t]
            new_delta = bs_delta(
                S[:, t], env.K, env.r, env.sigma, tau_t
            )  # Change to BS-Delta
            trade = (new_delta - delta) * S[:, t]
            pnl -= trade
            pnl -= env.cost * np.abs(new_delta - delta) * S[:, t]
            pnl *= np.exp(env.r * env.dt)
            delta = new_delta

        pnl += delta * S[:, -1]  # Liquidate stock position at maturity
        pnl -= np.maximum(S[:, -1] - env.K, 0)  # Pay option value to buyer
        return pnl
