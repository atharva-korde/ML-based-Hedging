import torch
from torch import Tensor


# This is a utility function that we choose to minimize.
# Can choose other functions, for example, just the negative of the expected PnL without caring about variance
def hedging_loss(pnl: Tensor, c: float = 1.0) -> Tensor:

    var_pnl = torch.var(pnl, unbiased=True)
    exp_pnl = torch.mean(pnl)
    return var_pnl - c * exp_pnl
    
