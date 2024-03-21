import torch
from typing import Tuple


def generate_cubic_dataset(
    noise_std: float = 3.0,
    dataset_size: int = 100,
) -> Tuple[torch.Tensor]:

    x_neg, x_pos = torch.zeros(dataset_size // 2), torch.zeros(dataset_size // 2)
    x_neg, x_pos = x_neg.uniform_(-4, -2), x_pos.uniform_(2, 4)
    x = torch.cat((x_neg, x_pos))

    y = x**3 + torch.tensor(noise_std) * torch.normal(
        torch.zeros(dataset_size), torch.ones(dataset_size)
    )

    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()

    return x.unsqueeze(-1), y.unsqueeze(-1)
