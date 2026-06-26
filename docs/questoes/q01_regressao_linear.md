# Q1 — Regressão Linear

## Enunciado

Implemente um modelo de regressão linear com dados sintéticos:

```
y = 3x + 5 + ε
```

- `x` ~ Uniforme(-10, 10)
- `ε` ~ Gaussiano(μ=0, σ=2)

## Passos

1. Gere pelo menos **100 pontos**
2. Divida em treino (**80%**) e teste (**20%**)
3. Implemente:
   - Mínimos quadrados (pseudo-inversa)
   - Rede neural (1 camada, SGD + MSE-Loss)
4. Apresente as soluções de cada método
5. Avalie desempenho e visualize os resultados

## Arquivos

| Arquivo | Responsabilidade |
|---------|------------------|
| `config.py` | Hiperparâmetros |
| `data.py` | ✅ Geração e split (pronto) |
| `models.py` | ⚠️ Implementar MQ e rede neural |
| `evaluate.py` | ✅ Métricas (pronto) |
| `visualize.py` | ⚠️ Implementar gráficos |
| `solve.py` | Orquestração |

## Executar

```bash
ia run q01_regressao_linear
```
