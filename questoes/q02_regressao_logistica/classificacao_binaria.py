import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def classificacao_binaria():
    #gerando o conjunto de dado
    X, y = make_classification(
        n_samples=500,      #número de amostras
        n_features=2,       #número total de atributos
        n_informative=2,    #atributos que realmente carregam informação
        n_redundant=0,      #atributos redundantes
        n_repeated=0,       #atributos repetidos
        n_classes=2,        #classificação binária
        random_state=42
    )


    #divisão de treino e teste
    X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.30,      #30% para teste
    random_state=42,      #garante a mesma divisão em cada execução
    stratify=y            #mantém a proporção das classes
)
    
    #convertando os dados para tensores
    X_train = torch.tensor(X_treino, dtype=torch.float32)
    y_train = torch.tensor(y_treino, dtype=torch.float32).view(-1, 1)

    X_test = torch.tensor(X_teste, dtype=torch.float32)
    y_test = torch.tensor(y_teste, dtype=torch.float32).view(-1, 1)

    modelo = RegressaoLogistica()
    
    #função de perda
    perda = nn.BCELoss()
   
    optimizer = torch.optim.SGD(
    modelo.parameters(),
    lr=0.01
)
    
    num_epocas = 500

    for epoca in range(num_epocas):

        # Forward
        y_pred = modelo(X_train)

        # Calcula a perda
        loss = perda(y_pred, y_train)

        # Zera os gradientes antigos
        optimizer.zero_grad()

        # Calcula os novos gradientes
        loss.backward()

        # Atualiza os pesos
        optimizer.step()

        if (epoca + 1) % 50 == 0:
            print(f"Época {epoca+1}: Loss = {loss.item():.4f}")
    
    modelo.eval()
    #avaliando a acurácia
    with torch.no_grad():

        y_prob = modelo(X_test)

        y_pred = (y_prob >= 0.5).float()

        acuracia = (y_pred == y_test).float().mean()

    print(f"Acurácia: {acuracia.item()*100:.2f}%")

    #visualizando a fronteira de decisão
    
    #criação de uma grade de pontos
    x_min, x_max = X_treino[:,0].min()-1, X_treino[:,0].max()+1
    y_min, y_max = X_treino[:,1].min()-1, X_treino[:,1].max()+1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02)
    )
    
    #classificando todos os pontos
    grid = np.c_[xx.ravel(), yy.ravel()]

    grid_tensor = torch.tensor(grid, dtype=torch.float32)

    with torch.no_grad():
        Z = modelo(grid_tensor)

    Z = Z.numpy().reshape(xx.shape)

    #plotando a fronteira
    plt.figure(figsize=(8,6))

    plt.contourf(
        xx,
        yy,
        Z >= 0.5,
        alpha=0.3,
        cmap="coolwarm"
    )

    plt.scatter(
        X_teste[:,0],
        X_teste[:,1],
        c=y_teste.squeeze(),
        cmap="coolwarm",
        edgecolors="k"
    )

    plt.xlabel("Variável 1")
    plt.ylabel("Variável 2")
    plt.title("Fronteira de decisão - Regressão Logística")

    plt.show()

    print("Pesos:", modelo.linear.weight.data)
    print("Bias:", modelo.linear.bias.data)

#.

class RegressaoLogistica(nn.Module):

    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.linear(x)
        x = self.sigmoid(x)

        return x

classificacao_binaria()