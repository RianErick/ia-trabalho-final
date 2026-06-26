# Q3 — MLP com Seleção de Neurônios

## Enunciado

MLP para classificação binária com validação para escolher o número de neurônios ocultos.

## Passos

1. `make_moons`: **500** amostras
2. Treino (**70%**) / validação (**15%**) / teste (**15%**)
3. MLP: oculta **ReLU** (n neurônios) + saída **sigmoid** (1 neurônio), **BCELoss**, Adam ou SGD
4. Treinar com n ∈ **{5, 10, 20, 50}**
5. Plotar evolução da **loss** por época
6. Escolher n com **menor loss em validação**
7. Avaliar no teste + fronteira de decisão

## Arquivos

| Arquivo | Responsabilidade |
|---------|------------------|
| `config.py` | Hiperparâmetros |
| `data.py` | ✅ Geração e split (pronto) |
| `model.py` | ⚠️ MLP, treino, predição |
| `train.py` | ✅ Loop de seleção (pronto) |
| `visualize.py` | ⚠️ Loss e fronteira |
| `solve.py` | Orquestração |

## Executar

```bash
ia run q03_mlp_selecao_neuronios
```
