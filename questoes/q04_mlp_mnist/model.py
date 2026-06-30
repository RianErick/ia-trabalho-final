"""MLP para classificação multiclasse no MNIST."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import HIDDEN_SIZE, INPUT_SIZE, LEARNING_RATE, NUM_CLASSES, OPTIMIZER


class MLP(nn.Module):
    """Uma camada oculta ReLU e saída com 10 logits."""

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def criar_modelo() -> MLP:
    return MLP()


def _criar_otimizador(
    modelo: nn.Module,
    learning_rate: float = LEARNING_RATE,
) -> torch.optim.Optimizer:
    if OPTIMIZER == "adam":
        return torch.optim.Adam(modelo.parameters(), lr=learning_rate)
    return torch.optim.SGD(modelo.parameters(), lr=learning_rate)


def treinar(
    modelo: nn.Module,
    train_loader: DataLoader,
    *,
    epochs: int,
    learning_rate: float = LEARNING_RATE,
    device: torch.device | None = None,
) -> dict:
    """Treina a MLP usando CrossEntropyLoss."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = _criar_otimizador(modelo, learning_rate)
    loss_history: list[float] = []

    for epoch in range(1, epochs + 1):
        modelo.train()
        epoch_loss = 0.0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = modelo(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            batch_size = X_batch.size(0)
            epoch_loss += loss.item() * batch_size
            total += batch_size

        mean_loss = epoch_loss / total
        loss_history.append(mean_loss)
        print(f"Época {epoch:02d}/{epochs} - loss treino: {mean_loss:.4f}")

    return {"modelo": modelo, "loss_history": loss_history}


def avaliar(
    modelo: nn.Module,
    data_loader: DataLoader,
    *,
    device: torch.device | None = None,
) -> dict:
    """Calcula loss média e acurácia em um DataLoader."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo.to(device)

    criterion = nn.CrossEntropyLoss()
    modelo.eval()

    total_loss = 0.0
    total = 0
    corretos = 0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = modelo(X_batch)
            loss = criterion(logits, y_batch)
            pred = torch.argmax(logits, dim=1)

            batch_size = X_batch.size(0)
            total_loss += loss.item() * batch_size
            total += batch_size
            corretos += int((pred == y_batch).sum().item())

    return {
        "loss": total_loss / total,
        "acuracia": corretos / total,
        "corretos": corretos,
        "total": total,
    }


def prever(
    modelo: nn.Module,
    imagens: torch.Tensor,
    *,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Retorna classes previstas e probabilidades para um lote de imagens."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo.to(device)
    modelo.eval()

    with torch.no_grad():
        logits = modelo(imagens.to(device))
        probabilidades = torch.softmax(logits, dim=1)
        predicoes = torch.argmax(probabilidades, dim=1)

    return predicoes.cpu(), probabilidades.cpu()
