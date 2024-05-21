
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Carregar o dataset
df = pd.read_csv("ConsumoCo2.csv")

dados = ["ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB', 'FUELCONSUMPTION_COMB_MPG"]
x = df[dados]
y = df["CO2EMISSIONS"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



# Instanciar o modelo Naive Bayes
nb_model = GaussianNB()

# Treinar o modelo
nb_model.fit(x_train, y_train)

# Fazer previsões
y_pred_nb = nb_model.predict(x_test)

# Avaliar o desempenho do modelo
mae = mean_squared_error(y_test, y_pred_nb)
r2 = r2_score(y_test, y_pred_nb)


print("Mean Squared Error:", mae)
print("R² Score:", r2)
