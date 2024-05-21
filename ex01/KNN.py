import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error

# Carregar os dados do arquivo CSV
dados = pd.read_csv('ConsumoCo2.csv')
dados_tratados = pd.get_dummies(dados, columns=['MAKE', 'MODEL', 'VEHICLECLASS', 'TRANSMISSION', 'FUELTYPE'])

x = dados_tratados.drop(columns=['CO2EMISSIONS'])  
y = dados_tratados['CO2EMISSIONS'] 

# Dividir os dados em conjuntos de treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.5, random_state=45)

# Padronizar os dados (importante para KNN)
scaler = StandardScaler()
x_treino = scaler.fit_transform(x_treino)
x_teste = scaler.transform(x_teste)

knn = KNeighborsRegressor(n_neighbors=5)

# Treinar o modelo
knn.fit(x_treino, y_treino)

# Prever os valores de CO2EMISSIONS para os dados de teste
y_pred = knn.predict(x_teste)


# Calcular o MAE
mae = mean_absolute_error(y_teste, y_pred)
print("Erro Medio Absoluto:", mae)
