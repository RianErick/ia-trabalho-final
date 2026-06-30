"""Orquestração da Questão 4 — MLP para classificação MNIST."""

from __future__ import annotations

import torch

from config import EPOCHS, LEARNING_RATE, OUTPUT_DIR
from data import carregar_mnist, fixar_semente
from model import avaliar, criar_modelo, treinar
from visualize import plotar_loss, plotar_previsoes


def resolver() -> dict:
    fixar_semente()
    device = torch.device("cpu")

    train_loader, test_loader, train_dataset, test_dataset = carregar_mnist()
    print(
        "Conjuntos: "
        f"treino={len(train_dataset)}, teste={len(test_dataset)}, device={device}"
    )

    modelo = criar_modelo()
    resultado_treino = treinar(
        modelo,
        train_loader,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        device=device,
    )

    metricas_teste = avaliar(modelo, test_loader, device=device)
    fig_loss = plotar_loss(resultado_treino["loss_history"], OUTPUT_DIR)
    fig_previsoes = plotar_previsoes(
        modelo,
        test_loader,
        output_dir=OUTPUT_DIR,
        device=device,
    )

    resultado = {
        "loss_teste": metricas_teste["loss"],
        "acuracia_teste": metricas_teste["acuracia"],
        "corretos": metricas_teste["corretos"],
        "total": metricas_teste["total"],
        "figuras": [str(fig_loss), str(fig_previsoes)],
    }

    print("\n=== Resultados ===")
    print(f"Loss no teste: {resultado['loss_teste']:.4f}")
    print(
        "Acurácia no teste: "
        f"{resultado['acuracia_teste']:.2%} "
        f"({resultado['corretos']}/{resultado['total']})"
    )
    print("Figuras salvas:")
    for fig in resultado["figuras"]:
        print(f"  - {fig}")

    return resultado


if __name__ == "__main__":
    resolver()
