

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Carregar o dataset
df = pd.read_csv("ConsumoCo2.csv")

# Selecionar as características relevantes
dados = ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB', 'FUELCONSUMPTION_COMB_MPG']
x = df[dados]
y = df['CO2EMISSIONS']

# Dividir os dados em conjuntos de treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Instanciar o modelo de Árvore de Decisão
dt_model = DecisionTreeRegressor(random_state=42)

# Treinar o modelo
dt_model.fit(x_train, y_train)

# Fazer previsões
y_pred_dt = dt_model.predict(x_test)

mae = mean_squared_error(y_test, y_pred_dt)
r2  = r2_score(y_test, y_pred_dt)


print("Mean Squared Error:", mae)
print("R² Score:", r2)

