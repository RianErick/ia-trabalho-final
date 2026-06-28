"""Orquestração da Questão 3 — MLP com seleção de neurônios."""

from __future__ import annotations

import numpy as np

from config import OUTPUT_DIR
from data import dividir_conjuntos, gerar_dados
from model import prever_classe
from train import selecionar_melhor
from visualize import plotar_fronteira_decisao, plotar_loss_por_neuronios


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  return float(np.mean(y_true.ravel() == y_pred.ravel()))


def resolver() -> dict:
  X, y = gerar_dados()
  X_train, X_val, X_test, y_train, y_val, y_test = dividir_conjuntos(X, y)

  print(f"Conjuntos: treino={len(X_train)}, val={len(X_val)}, teste={len(X_test)}")

  selecao = selecionar_melhor(X_train, y_train, X_val, y_val)

  y_pred = prever_classe(selecao.melhor_modelo, X_test)
  acc = accuracy(y_test, y_pred)

  fig_loss = plotar_loss_por_neuronios(selecao.historico, OUTPUT_DIR)
  fig_fronteira = plotar_fronteira_decisao(
    X,
    y,
    selecao.melhor_modelo,
    selecao.melhor_n,
    acc,
    OUTPUT_DIR,
  )

  resultado = {
    "melhor_n": selecao.melhor_n,
    "val_loss": selecao.melhor_val_loss,
    "acuracia_teste": acc,
    "figuras": [str(fig_loss), str(fig_fronteira)],
  }

  print("\n=== Resultados ===")
  print(f"Melhor n (neurônios ocultos): {resultado['melhor_n']}")
  print(f"Loss de validação: {resultado['val_loss']:.4f}")
  print(f"Acurácia no teste: {resultado['acuracia_teste']:.2%}")
  print("Figuras salvas:")
  for fig in resultado["figuras"]:
    print(f"  - {fig}")

  return resultado


if __name__ == "__main__":
  resolver()
