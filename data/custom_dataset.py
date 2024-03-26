import torch
import pandas as pd
from typing import Tuple


def choose_k_from_n(n: int, k: int):
    assert k <= n
    idx = torch.randperm(n)
    return idx[:k], idx[k:]


def uci_to_normalised_ttsplit(
    X: pd.DataFrame, y: pd.DataFrame, train_proportion: float = 0.9
) -> Tuple[torch.Tensor]:
    assert train_proportion > 0.0 and train_proportion < 1.0

    X = torch.tensor(X.values)
    y = torch.tensor(y.values)
    
    if len(y.shape) < 2:
        y = y.unsqueeze(-1)

    X = (X - X.mean(0)) / X.std(0)
    y = (y - y.mean(0)) / y.std(0)

    N = X.shape[0]
    N_train = int(N * train_proportion)

    train_idx, test_idx = choose_k_from_n(N, N_train)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    return X_train, y_train, X_test, y_test
