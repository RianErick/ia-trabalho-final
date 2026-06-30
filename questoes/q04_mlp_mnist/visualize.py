"""Visualizações da Questão 4."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from config import N_PREDICTIONS_TO_SHOW, OUTPUT_DIR
from model import prever


def _salvar_figura(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plotar_loss(loss_history: list[float], output_dir: Path = OUTPUT_DIR) -> Path:
    """Plota a evolução da loss de treino ao longo das épocas."""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(loss_history) + 1)
    ax.plot(epochs, loss_history, marker="o", linewidth=1.8)
    ax.set_title("Evolução da Loss de Treino — MLP MNIST")
    ax.set_xlabel("Época")
    ax.set_ylabel("CrossEntropyLoss")
    ax.grid(True, alpha=0.3)

    return _salvar_figura(output_dir / "loss_treino.png")


def _desnormalizar(imagem: torch.Tensor) -> torch.Tensor:
    """Reverte a normalização padrão usada no MNIST."""
    return imagem * 0.3081 + 0.1307


def plotar_previsoes(
    modelo: torch.nn.Module,
    test_loader: DataLoader,
    *,
    output_dir: Path = OUTPUT_DIR,
    n_images: int = N_PREDICTIONS_TO_SHOW,
    device: torch.device | None = None,
) -> Path:
    """Salva uma grade com imagens, rótulos reais e predições do modelo."""
    imagens, rotulos = next(iter(test_loader))
    imagens = imagens[:n_images]
    rotulos = rotulos[:n_images]
    predicoes, probabilidades = prever(modelo, imagens, device=device)

    n_cols = 4
    n_rows = (n_images + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 2.6 * n_rows))
    axes_list = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for idx, ax in enumerate(axes_list):
        ax.axis("off")
        if idx >= len(imagens):
            continue

        imagem = _desnormalizar(imagens[idx].squeeze()).clamp(0, 1)
        real = int(rotulos[idx].item())
        pred = int(predicoes[idx].item())
        conf = float(probabilidades[idx, pred].item())
        cor = "green" if pred == real else "red"

        ax.imshow(imagem, cmap="gray")
        ax.set_title(f"real={real} | pred={pred}\nconf={conf:.1%}", color=cor)

    return _salvar_figura(output_dir / "previsoes.png")
