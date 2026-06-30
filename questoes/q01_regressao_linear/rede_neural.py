import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def rede_neural():
     #seed para reproduzir os mesmos resultados
    np.random.seed(42)

    n = 100

    #valores de x variando de -10 até 10(100 vezes)
    x = np.random.uniform(-10, 10, n)

    #ruído gaussiano com média 0 e desvio padrão 2
    ruido = np.random.normal(loc=0, scale=2, size=n)

    #equação da reta
    y = 3 * x + 5 + ruido

    #divisão em treino (80%) e teste (20%)
    X_treino, X_teste, y_treino, y_teste = train_test_split(
    x.reshape(-1,1),
    y.reshape(-1,1),
    test_size=0.2,
    random_state=42
)
    #conversão para tensores
    X_treino = torch.FloatTensor(X_treino)
    y_treino = torch.FloatTensor(y_treino)

    X_teste = torch.FloatTensor(X_teste)
    y_teste = torch.FloatTensor(y_teste)

    #criação da rede neural( com um peso w e um bias b)
    modelo = nn.Linear(1,1)

    #definindo a funcao de perda
    perda = nn.MSELoss()

    #definindo o otimizador
    optimizer = optim.SGD(modelo.parameters(), lr=0.01)

    epocas = 500

    historico_loss = []

    
    #treinamento do modelo
    for epoca in range(epocas):

        y_pred = modelo(X_treino)

        loss = perda(y_pred, y_treino)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        historico_loss.append(loss.item())

        if (epoca+1) % 100 == 0:
            print(f"Época {epoca+1}: Loss = {loss.item():.4f}")

    #grafico do mse loss
    plt.figure(figsize=(8,5))

    plt.plot(historico_loss)

    plt.title("Evolução da MSE Loss")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.show()    
    
    #parâmetros aprendidos
    peso = modelo.weight.item()
    bias = modelo.bias.item()

    print("Coeficiente Angular:", peso)
    print("Intercepto:", bias)
    
    #avaliação
    modelo.eval()

    with torch.no_grad():
        y_pred = modelo(X_teste)

    mse = torch.mean((y_teste - y_pred)**2)

    print("MSE:", mse.item())

    #convertendo para numpy
    x = X_teste.numpy().flatten()
    y_real = y_teste.numpy().flatten()
    y_pred = y_pred.flatten()

    #grafico da reta
    indices = np.argsort(x)

    plt.figure(figsize=(8,5))

    plt.scatter(x, y_real, color='blue', label='Dados de teste')
    plt.plot(x[indices], y_pred[indices],
            color='red',
            linewidth=2,
            label='Reta ajustada')

    plt.title("Regressão Linear com Rede Neural")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)

    plt.show()

rede_neural()