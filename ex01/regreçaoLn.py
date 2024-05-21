import pandas as pd
from sklearn.model_selection import train_test_split # x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.4, random_state=3)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

dados = pd.read_csv('ConsumoCo2.csv')


x = dados[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']]  # Características
y = dados['CO2EMISSIONS'] 

# test_size=y: Define que x% dos dados serão usados para o conjunto de teste, enquanto os outros y% serão usados para treino.
# random_state=x: Define a semente para a geração de números aleatórios para garantir que a divisão dos dados seja reprodutível.
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.4, random_state=3)

# Criar o modelo de regressão linear
modelo = LinearRegression()

# Treinar o modelo
modelo.fit(x_treino, y_treino)

# Prever os valores de emissões de CO2 para os dados de teste
y_pred = modelo.predict(x_teste)

# Calcular o MAE
mae = mean_absolute_error(y_teste, y_pred)
print("Erro Medio Absoluto:", mae)
 