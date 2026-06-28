"""MLP para classificação binária (passo 3)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, OPTIMIZER
from data import to_tensors


class MLP(nn.Module):
    """Camada oculta ReLU + saída sigmoid."""

    def __init__(self, n_hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def criar_modelo(n_hidden: int) -> MLP:
  return MLP(n_hidden)


def _criar_dataloader(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    X_t, y_t = to_tensors(X, y)
    dataset = TensorDataset(X_t, y_t)  # type: ignore[arg-type]
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def treinar(
    modelo: MLP,
    X: np.ndarray,
    y: np.ndarray,
    *,
    learning_rate: float = LEARNING_RATE,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """Treina com BCELoss e Adam ou SGD."""
    criterion = nn.BCELoss()
    if OPTIMIZER == "adam":
        optimizer = torch.optim.Adam(modelo.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(modelo.parameters(), lr=learning_rate)

    loader = _criar_dataloader(X, y, batch_size, shuffle=True)
    loss_history: list[float] = []

    modelo.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            y_pred = modelo(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        loss_history.append(epoch_loss / n_batches)

    return {"modelo": modelo, "loss_history": loss_history}


def avaliar_loss(modelo: MLP, X: np.ndarray, y: np.ndarray) -> float:
    """Calcula BCELoss no conjunto informado."""
    criterion = nn.BCELoss()
    X_t, y_t = to_tensors(X, y)
    modelo.eval()
    with torch.no_grad():
        y_pred = modelo(X_t)
        return float(criterion(y_pred, y_t).item())


def prever_proba(modelo: MLP, X: np.ndarray) -> np.ndarray:
    """Probabilidades da classe positiva."""
    X_t, _ = to_tensors(X)
    modelo.eval()
    with torch.no_grad():
        return modelo(X_t).numpy().ravel()


def prever_classe(
    modelo: MLP, X: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    """Classes preditas (0 ou 1)."""
    proba = prever_proba(modelo, X)
    return (proba >= threshold).astype(np.int64)
