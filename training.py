import torch
import torch.optim as optim
import torch.nn as nn

from policy import PolicyNNet
from hedging import HedgingEnv
from loss_function import hedging_loss


def train(
    env: HedgingEnv,
    model: PolicyNNet,
    n_epochs: int = 200,
    lr: float = 1e-3,
    c: float = 1.0,
    batch_size: int | None = None,  # None = full batch
    verbose: bool = True,
) -> list[float]:
    # Trains the hedging network and returns a list of loss values per epoch. 
    # Training is done over the entire batch size, that is, the model's weights remain fixed for all paths during one epoch.
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    losses = [] # Array to store the values of the loss function after each epoch
    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()  # Set zero gradients

        pnl = env.rollout(model)  # Apply to benchmark hedging / ML-based hedging
        loss = hedging_loss(pnl)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if verbose and epoch % 20 == 0:
            print(
                f"Epoch {epoch:4d}/{n_epochs}  |  "
                f"Loss: {loss.item():+.4f}  |  "
                f"E[PnL]: {pnl.mean().item():+.4f}  |  "
                f"Var[PnL]: {pnl.var().item():.4f}"
            )

    return losses
