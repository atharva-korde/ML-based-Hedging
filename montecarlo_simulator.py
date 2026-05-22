import numpy as np
from numpy.typing import NDArray


# We assume that the stock price follows Geometric Brownian Motion
def simulate_paths_gbm(
    S0: float, r: float, sigma: float, T: int, steps: int, paths: int, seed: int = 42
) -> np.NDArray[np.float64]:

    rng = np.random.default_rng(seed)
    dt = T / steps

    S = np.zeros((paths, steps + 1))
    S[:, 0] = S0

    for t in range(1, steps + 1):

        z = rng.standard_normal(paths)
        S[:, t] = S[:, t - 1] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        )

    return S  # Array of size paths x (steps + 1), the rows are the paths
