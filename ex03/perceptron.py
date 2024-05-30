import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Carregar e preparar os dados
digits = load_digits()
x = digits.data / 16.0
y = digits.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Definição do Perceptron
class perceptronSigmoid:
    def __init__(self, input_size, output_size, learning_rate=0.1):
        self.weights = np.random.randn(output_size, input_size) * 0.01
        self.bias = np.zeros(output_size)
        self.lr = learning_rate

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


    def sigmoid_derivada(self, z):
        return z * (1 - z)


    def predict(self, x):
        return self.sigmoid(np.dot(x, self.weights.T) + self.bias)

    def train(self, x, y, epochs=1000):
        y_encoded = np.eye(10)[y]
        for _ in range(epochs):
            for xi, target in zip(x, y_encoded):
                pred = self.predict(xi)
                error = target - pred
                delta = self.lr * error * self.sigmoid_derivative(pred)
                self.weights += np.outer(delta, xi)
                self.bias += delta

    def accuracy(self, x, y):
        preds = np.argmax(self.predict(x), axis=1)
        return np.mean(preds == y)

# Inicializar e treinar o Perceptron
input_size = x_train.shape[1]
output_size = 10
perceptron = perceptronSigmoid(input_size, output_size, learning_rate=0.1)
perceptron.train(x_train, y_train, epochs=1000)

# Avaliar a acurácia do modelo no conjunto de teste
accuracy = perceptron.accuracy(x_test, y_test)
print(f"Acurácia do Perceptron: {accuracy * 100:.2f}%")

# Visualizar os pesos de um dos dígitos
plt.imshow(perceptron.weights[0].reshape(8, 8), cmap='viridis')
plt.colorbar()
plt.title("Visualização dos Pesos do Perceptron para o Dígito 0")
plt.show()
