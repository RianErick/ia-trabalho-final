import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#y = dependente(saida), x = independente(entrada)
#pelo o que eu entendi, na solução dos mínimos quadrados utilizando a pseudo inversa, o modelo é o vetor de parâmetros theta
#geramos os dados e aplicamos a pseudo inversa("treinando o modelo")
#LinearRegression() resolve o problema de mínimos quadrados, mas não calcula theta = np.linalg.pinv(X_train) @ y_treino de forma explícita
#Ambos teriam a mesma solução matemática ou bem parecida

def pseudo_inversa():
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
        x.reshape(-1, 1),  #transforma em matriz (100,1)
        y,
        test_size=0.2,
        random_state=42
    )

    #adiciona coluna de 1 pois a reta possui intercepto
    X_train = np.column_stack((np.ones(len(X_treino)), X_treino))
    X_test = np.column_stack((np.ones(len(X_teste)), X_teste))

    #calcula os coeficientes usando a pseudo inversa/"treina o modelo com a pseudo inversa"
    #theta 0 é o intercepto(intercepto = 5)
    #theta 1 é o coeficiente angular/inclinação(coeficiente angular = 3)
    theta = np.linalg.pinv(X_train) @ y_treino

    #previsão
    y_pred = X_test @ theta

    #avaliação
    #erro quadradico médio
    #coeficiente de determinação
    mse = mean_squared_error(y_teste, y_pred)
    r2 = r2_score(y_teste, y_pred)

    print(f"solução:")
    print(f"Intercepto: {theta[0]:.3f}")
    print(f"Coeficiente Angular: {theta[1]:.3f}")
    print(f"avaliação:")
    print(f"MSE: {mse:.3f}")
    print(f"R²: {r2:.3f}")

    #dados de teste
    z = X_teste.ravel()

    #ordena os pontos para desenhar a reta
    indices = np.argsort(z)

    plt.figure(figsize=(8,5))
    plt.scatter(z, y_teste, color='blue', label='Dados de teste')
    plt.plot(z[indices], y_pred[indices], color='red', linewidth=2, label='Reta ajustada')

    plt.title("Regressão Linear por Mínimos Quadrados (Pseudoinversa)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)

    plt.show()


pseudo_inversa()