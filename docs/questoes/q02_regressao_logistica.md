# Q2 — Regressão Logística (Classificação Binária)

## Enunciado

Implemente regressão logística para classificação binária com dados sintéticos.

## Passos

1. `make_classification`: **500** amostras, **2** features, **2** classes
2. Treino (**70%**) / teste (**30%**)
3. Rede com 1 saída + **sigmoid**
4. Treinar com **gradiente descendente batch** (não-estocástico)
5. Acurácia no teste + **fronteira de decisão**

## Arquivos

| Arquivo | Responsabilidade |
|---------|------------------|
| `config.py` | Hiperparâmetros |
| `data.py` | ✅ Geração e split (pronto) |
| `model.py` | ⚠️ Implementar treino e predição |
| `visualize.py` | ⚠️ Fronteira de decisão |
| `solve.py` | Orquestração |

## Executar

```bash
ia run q02_regressao_logistica
```
