# ML-Based Hedging

## Introduction

Assume that we have sold one European Call option for a stock with price $S=S(t)$, that is we are in 'short call' position. In order to prevent losses at maturity $T$, we wish to buy some shares of the stock and hedge our position over time depending on the stock price graph. Let $V=V(t)$ be the option price, whose value can be computed at any time $t \in [0,T]$ by the Black-Scholes formula. If our portfolio is $P = -V + bS$ (one short call position and $b$ shares of stock) then the first derivative wrt stock price is $\frac{dP}{dS} = -\frac{dV}{dS} + b$. Therefore, $b$ should equal the Delta of the option price at any time $t$ in order to maintain zero change in the portfolio no matter how the stock price moves. Clearly, this is only a suboptimal (and impractical) strategy as aggressively rebalancing the portfolio over each discrete time interval will lead to mounting transaction costs.

As an alternative comparison, we attempt a ML-based approach to hedge our position. Suppose that, at time $t$, the portfolio is $P = -V + b_tS(t)$. Instead of hedging to the Black-Scholes Delta, we use a neural net to 'learn a function' $\Phi: (S(t), b_t, \tau) \rightarrow b_{t+1}$, where $\tau=T-t$ denotes the time to maturity, and $b_{t+1}$ is the position at the next discrete time step. This 'learning' is implemented as follows: We construct random price paths for the stock price. For each path, we receive the price of the option at $t=0$ initially. Upto maturity, we lose some money due to transaction costs of rebalancing stock. At maturity, we need to pay the value of the option to the buyer and we gain value equal to $b_TS(T)$, as we have $b_T$ shares of stock at maturity. Therefore, we compute our overall profit/loss over a price path. Then, the model learns a function in order to minimize $\operatorname{Var}(PnL) - c \cdot \mathbb{E}(PnL)$, where $c$ is some constant the variance and expectation of profit/loss is taken over all price paths. Note that this is a sensible function to minimize, as we would like to minimize the variance and maximize the expectation simultaneously. The 'learning' happens using the standard gradient descent methods, which updates the model's parameters after every epoch.

For comparison, we apply both, Black-Scholes theory and the model to a test set.


## Results

The results show that the Machine-Learning approach performs slightly better than the theoretical approach. Here is a figure demonstrating the decrease in the loss function which we aimed to minimize, and the comparison between the two approaches.

![Visualization](Figue1.png)

