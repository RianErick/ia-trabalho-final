"""Visualizações (passos 5 e 7)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from model import prever_proba


def _salvar_figura(path: Path) -> Path:
  path.parent.mkdir(parents=True, exist_ok=True)
  plt.tight_layout()
  plt.savefig(path, dpi=150, bbox_inches="tight")
  plt.close()
  return path


def plotar_loss_por_neuronios(
  historico: dict[int, dict],
  output_dir: Path,
) -> Path:
  """Plota evolução da loss ao longo das épocas para cada n."""
  fig, axes = plt.subplots(2, 2, figsize=(10, 8))
  fig.suptitle("Evolução da Loss de Treino por Número de Neurônios")

  for ax, n in zip(axes.ravel(), sorted(historico.keys())):
    loss_history = historico[n]["loss_history"]
    epochs = range(1, len(loss_history) + 1)
    ax.plot(epochs, loss_history, linewidth=1.5)
    ax.set_title(f"n = {n} neurônios")
    ax.set_xlabel("Época")
    ax.set_ylabel("BCELoss")
    ax.grid(True, alpha=0.3)

  return _salvar_figura(output_dir / "loss_por_neuronios.png")


def plotar_fronteira_decisao(
  X: np.ndarray,
  y: np.ndarray,
  modelo: object,
  melhor_n: int,
  acuracia_teste: float,
  output_dir: Path,
) -> Path:
  """Fronteira de decisão do modelo escolhido."""
  margin = 0.5
  x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
  y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin

  xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200),
  )
  grid = np.c_[xx.ravel(), yy.ravel()]
  Z = prever_proba(modelo, grid).reshape(xx.shape)

  fig, ax = plt.subplots(figsize=(8, 6))
  ax.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap="RdYlBu")
  ax.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=1.5)

  scatter = ax.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    cmap="RdYlBu",
    edgecolors="black",
    linewidths=0.5,
    s=40,
  )
  ax.set_title(
    f"Fronteira de Decisão (n={melhor_n}, acurácia teste={acuracia_teste:.2%})"
  )
  ax.set_xlabel("Feature 1")
  ax.set_ylabel("Feature 2")
  fig.colorbar(scatter, ax=ax, label="Classe")

  return _salvar_figura(output_dir / "fronteira_decisao.png")
