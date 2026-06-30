"""Carregamento do MNIST para a Questão 4."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import BATCH_SIZE, DATA_DIR, RANDOM_STATE

try:
    from torchvision import datasets, transforms
except ModuleNotFoundError as exc:
    datasets = None
    transforms = None
    _TORCHVISION_ERROR = exc
else:
    _TORCHVISION_ERROR = None


def fixar_semente(seed: int = RANDOM_STATE) -> None:
    """Mantém a execução mais reprodutível entre rodadas."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def carregar_mnist(
    batch_size: int = BATCH_SIZE,
) -> tuple[DataLoader, DataLoader, object, object]:
    """Baixa/carrega o MNIST e retorna dataloaders de treino e teste."""
    if datasets is None or transforms is None:
        raise RuntimeError(
            "A Questão 4 precisa do pacote torchvision para carregar o MNIST. "
            "Instale com: python3 -m pip install --user --break-system-packages torchvision"
        ) from _TORCHVISION_ERROR

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform,
    )

    generator = torch.Generator().manual_seed(RANDOM_STATE)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader, train_dataset, test_dataset
