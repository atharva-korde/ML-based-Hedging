import numpy as np
from scipy.stats import norm


def bs_call_price(S0: float, K: float, tau: float, r: float, sigma: float) -> float:
    """ Returns the price of a European call option at maturity
        We use 'tau' for the time to maturity """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    return S0 * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)

def bs_delta(S0: float, K: float, tau: float, r: float, sigma: float) -> float:
    """ Returns the Delta (first derivative of option price wrt stock price) """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))

    return norm.cdf(d1)
