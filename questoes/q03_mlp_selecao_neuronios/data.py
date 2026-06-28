"""Geração e divisão dos dados (passos 1 e 2)."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.datasets import make_moons

from config import RANDOM_STATE, TRAIN_RATIO, VAL_RATIO, N_SAMPLES


def gerar_dados() -> tuple[np.ndarray, np.ndarray]:
    """Gera dados com make_moons (500 amostras)."""
    X, y = make_moons(
        n_samples=N_SAMPLES, noise=0.2, random_state=RANDOM_STATE
    )
    return X.astype(np.float32), y.astype(np.float32)


def dividir_conjuntos(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Divide em treino (70%), validação (15%) e teste (15%)."""
    rng = np.random.default_rng(RANDOM_STATE)
    indices = rng.permutation(len(X))

    train_end = int(len(X) * TRAIN_RATIO)
    val_end = train_end + int(len(X) * VAL_RATIO)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return (
        X[train_idx],
        X[val_idx],
        X[test_idx],
        y[train_idx],
        y[val_idx],
        y[test_idx],
    )


def to_tensors(
    X: np.ndarray, y: np.ndarray | None = None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Converte arrays NumPy em tensores PyTorch."""
    X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
    if y is None:
        return X_t, None
    return X_t, torch.from_numpy(np.asarray(y, dtype=np.float32)).reshape(-1, 1)
