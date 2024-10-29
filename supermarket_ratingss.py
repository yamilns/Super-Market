import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta, norm, skew, kurtosis

# Función para calcular y ajustar la distribución beta
def ajuste_beta(data):
    alpha, beta_param, _, _ = beta.fit(data)
    media_norm = alpha / (alpha + beta_param)
    varianza_norm = (alpha * beta_param) / ((alpha + beta_param) ** 2 * (alpha + beta_param + 1))
    return alpha, beta_param, media_norm, varianza_norm

# Cargar los datos del archivo CSV
data = pd.read_csv("C:\\Users\\yamil\\Downloads\\SuperMarketData.csv")

# Visualización de la distribución de los ratings (opcional)
data["Rating"].plot(kind="hist", bins=20, edgecolor="black", alpha=0.7)
plt.xlabel("Rating")
plt.title("Distribución de Ratings")
plt.show()

print(data.head())
print('Cantidad de registros en el DataFrame:', len(data))

# Calcular el máximo y mínimo
ratings = np.array(data["Rating"])
max_rating, min_rating = np.max(ratings), np.min(ratings)

# Normalizar los ratings para que estén entre 0 y 1
normalized_ratings = (ratings - min_rating) / (max_rating - min_rating)

# Ajustar la distribución beta y obtener parámetros
alpha, beta_param, mean_norm, var_norm = ajuste_beta(ratings)
print("Parámetros alpha y beta:", alpha, beta_param)

mean_rating = (max_rating - min_rating) * mean_norm + min_rating
variance_rating = (max_rating - min_rating) ** 2 * var_norm
std_dev_rating = np.sqrt(variance_rating)

print("Media y varianza en la escala de ratings:", mean_rating, variance_rating)

sesgo = skew(ratings)
curtosis = kurtosis(ratings)
print("Sesgo de la distribución:", sesgo)
print("Curtosis de la distribución:", curtosis)

# Calcular la probabilidad de que el promedio de los ratings sea mayor a 8.5
probabilidad = 1 - norm.cdf(8.5, mean_rating, std_dev_rating)
print("Probabilidad de que el promedio de los ratings sea mayor a 8.5:", probabilidad)

# Calcular el intervalo de confianza del 95% para la media
confianza_95 = norm.interval(0.95, loc=mean_rating, scale=std_dev_rating)
print("Intervalo de confianza del 95% para la media de ratings:", confianza_95)

# Visualización de la distribución ajustada
x = np.linspace(min_rating, max_rating, 100)
beta_dist = beta.pdf((x - min_rating) / (max_rating - min_rating), alpha, beta_param)
plt.hist(ratings, bins=20, density=True, alpha=0.6, color="skyblue", edgecolor="black", label="Datos")
plt.plot(x, beta_dist * (max_rating - min_rating), color="darkred", label="Ajuste Beta", lw=2)
plt.xlabel("Rating")
plt.title("Distribución de Ratings con Ajuste Beta")
plt.legend()
plt.show()

