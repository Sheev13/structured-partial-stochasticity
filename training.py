import torch
from torch import nn
from tqdm import tqdm
from typing import Optional, List, Tuple
from collections import defaultdict
from data.custom_dataset import choose_k_from_n
from models import MeanFieldBNN


def train(
    model: MeanFieldBNN,
    x: torch.Tensor,
    y: torch.Tensor,
    variational: bool = False,
    epochs: int = 100,
    learning_rate: float = 1e-2,
    num_samples: int = 1,
    batch_size: Optional[int] = None,
    x_test: Optional[torch.Tensor] = None,
    y_test: Optional[torch.Tensor] = None,
    final_learning_rate: Optional[float] = None,
) -> torch.Tensor:

    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if final_learning_rate is not None:
        end_factor = final_learning_rate / learning_rate
    else:
        end_factor = 1.0
    lr_sched = torch.optim.lr_scheduler.LinearLR(
        optimiser, start_factor=1.0, end_factor=end_factor, total_iters=epochs
    )

    tracker = defaultdict(list)
    pbar = tqdm(range(epochs))

    if batch_size is None:
        x_batch = x
        y_batch = y

    for _ in pbar:

        if batch_size is not None:
            assert batch_size <= x.shape[0]
            batch_idx = choose_k_from_n(x.shape[0], batch_size)[0]
            x_batch, y_batch = x[batch_idx], y[batch_idx]

        optimiser.zero_grad()
        if variational:
            loss, metrics = model.elbo_loss(
                x_batch,
                y_batch,
                num_samples=num_samples,
            )
        else:
            loss, metrics = model.map_loss(x_batch, y_batch)
        loss.backward()
        optimiser.step()
        lr_sched.step()

        if x_test is not None:
            assert y_test is not None
            with torch.no_grad():
                rmse, mlpp = model.evaluate(x_test, y_test, variational, num_samples)
            metrics["test rmse"] = rmse.detach()
            metrics["test mlpp"] = mlpp.detach()

        if model.train_c:
            metrics["c"] = model.c.detach()

        for key, value in metrics.items():
            tracker[key].append(float(value))

        pbar.set_postfix(metrics)

    return tracker
