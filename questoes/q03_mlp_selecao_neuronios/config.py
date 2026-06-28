"""Hiperparâmetros da Questão 3 — MLP com seleção de neurônios."""

from pathlib import Path

N_SAMPLES = 500
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
RANDOM_STATE = 42

HIDDEN_NEURONS_OPTIONS = [5, 10, 20, 50]

LEARNING_RATE = 0.01
EPOCHS = 200
BATCH_SIZE = 32
OPTIMIZER = "adam"

OUTPUT_DIR = (
    Path(__file__).resolve().parents[2] / "outputs" / "q03_mlp_selecao_neuronios"
)
