"""Seleção do melhor número de neurônios (passos 4–6)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import (
  BATCH_SIZE,
  EPOCHS,
  HIDDEN_NEURONS_OPTIONS,
  LEARNING_RATE,
)
from model import avaliar_loss, criar_modelo, treinar


@dataclass
class SelecaoNeuronios:
    melhor_n: int
    melhor_modelo: object
    melhor_val_loss: float
    historico: dict[int, dict]


def selecionar_melhor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> SelecaoNeuronios:
    """Treina modelos com n ∈ {5, 10, 20, 50} e escolhe o de menor loss em validação."""
    historico: dict[int, dict] = {}
    melhor_n = HIDDEN_NEURONS_OPTIONS[0]
    melhor_loss = float("inf")
    melhor_modelo = None

    for n in HIDDEN_NEURONS_OPTIONS:
        print(f"Treinando modelo com n={n} neurônios...")
        m = criar_modelo(n)
        resultado = treinar(
            m,
            X_train,
            y_train,
            learning_rate=LEARNING_RATE,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
        )
        val_loss = avaliar_loss(resultado["modelo"], X_val, y_val)
        historico[n] = {
            "modelo": resultado["modelo"],
            "val_loss": val_loss,
            "loss_history": resultado["loss_history"],
        }
        print(f"  val_loss = {val_loss:.4f}")
        if val_loss < melhor_loss:
            melhor_loss = val_loss
            melhor_n = n
            melhor_modelo = resultado["modelo"]

    return SelecaoNeuronios(
        melhor_n=melhor_n,
        melhor_modelo=melhor_modelo,
        melhor_val_loss=melhor_loss,
        historico=historico,
    )
