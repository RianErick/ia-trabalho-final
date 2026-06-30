"""Hiperparâmetros da Questão 4 — MLP para MNIST."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data" / "mnist"
OUTPUT_DIR = ROOT_DIR / "outputs" / "q04_mlp_mnist"

INPUT_SIZE = 28 * 28
HIDDEN_SIZE = 64
NUM_CLASSES = 10

EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001
OPTIMIZER = "adam"
RANDOM_STATE = 42

N_PREDICTIONS_TO_SHOW = 12
