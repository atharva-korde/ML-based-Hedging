import torch
import torch.nn as nn
from torch import Tensor


class PolicyNNet(nn.Module):

    def __init__(self, hidden: int = 64) -> None:

        super().__init__()

        # Create a neural network
        # Three input parameters: Stock price, Current hedge position, Time to maturity
        # One output: New hedge position
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # Constrains the output to (0, 1)
        )

    def forward(self, x: Tensor) -> Tensor:

        return self.net(x)
