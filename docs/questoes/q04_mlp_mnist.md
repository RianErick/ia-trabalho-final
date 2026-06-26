# Q4 — Classificação Multiclasse com MLP (MNIST)

## Enunciado

Rede neural para classificação de dígitos MNIST.

## Passos

1. Carregar MNIST (Torchvision ou Keras)
2. MLP: oculta **64** neurônios (**ReLU**) + saída **10** neurônios (**softmax**)
3. **CrossEntropyLoss** + Adam ou SGD
4. Treinar por **10 épocas**, avaliar acurácia no teste
5. Exibir previsões (imagem + classe predita)

## Arquivos

| Arquivo | Responsabilidade |
|---------|------------------|
| `config.py` | Hiperparâmetros |
| `data.py` | ⚠️ Carregar MNIST |
| `model.py` | ⚠️ MLP, treino, predição |
| `visualize.py` | ⚠️ Grid de previsões |
| `solve.py` | Orquestração |

## Executar

```bash
ia run q04_mlp_mnist
```
