import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta, norm

# Función para calcular y ajustar la distribución beta
def ajuste_beta(data):
    alpha, beta_param, _, _ = beta.fit(data)
    media_norm = alpha / (alpha + beta_param)
    varianza_norm = (alpha * beta_param) / ((alpha + beta_param) ** 2 * (alpha + beta_param + 1))
    return alpha, beta_param, media_norm, varianza_norm

# Cargar los datos del archivo CSV
data = pd.read_csv("C:\\Users\\yamil\\Downloads\\SuperMarketData.csv")

# Inspeccionar los primeros registros y la longitud del DataFrame
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

# Escalar la media y varianza normalizadas a la escala original de ratings
mean_rating = (max_rating - min_rating) * mean_norm + min_rating
variance_rating = (max_rating - min_rating) ** 2 * var_norm
std_dev_rating = np.sqrt(variance_rating)

print("Media y varianza en la escala de ratings:", mean_rating, variance_rating)

# Calcular la probabilidad de que el promedio de los ratings sea mayor a 8.5
probabilidad = 1 - norm.cdf(8.5, mean_rating, std_dev_rating)
print("Probabilidad de que el promedio de los ratings sea mayor a 8.5:", probabilidad)

# Calcular el intervalo de confianza del 95% para la media
confianza_95 = norm.interval(0.95, loc=mean_rating, scale=std_dev_rating)
print("Intervalo de confianza del 95% para la media de ratings:", confianza_95)

# Visualización de la distribución de los ratings
plt.hist(ratings, bins=20, edgecolor="black", alpha=0.7, color="skyblue")
plt.xlabel("Rating")
plt.ylabel("Frecuencia")
plt.title("Distribución de Ratings")
plt.show()
