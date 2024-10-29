import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta, norm

# Cargar los datos del supermercado desde el archivo CSV
df = pd.read_csv("C:\\Users\\yamil\\Downloads\\SuperMarketData.csv")

# Visualización de la distribución de las ventas para entender la distribución de los datos
df["Sales"].plot(kind="hist", bins=20, edgecolor="black", alpha=0.7)
plt.xlabel("Sales")
plt.title("Distribución de Sales")
plt.show()

# Mostrar las primeras filas del dataframe y su tamaño
print(df.head())
print('Longitud del dataframe:', len(df))

# Transformación de ventas y normalización
sales = np.array(df["Sales"]) * 19.88  # Escalar ventas
max_sales = max(sales)
min_sales = min(sales)
sales_norm = 1 / (max_sales - min_sales) * (sales - min_sales)  # Normalizar a [0, 1]

# Ajuste de la distribución Beta para modelar la distribución de las ventas
a, b, _, _ = beta.fit(sales)
print("alpha y beta:", a, b)

# Cálculo de media y varianza normalizadas usando parámetros de la distribución beta
mu_norm = a / (a + b)
var_norm = (a * b) / ((a + b) ** 2 * (a + b + 1))
desv_norm = np.sqrt(var_norm)

# Convertir los valores normalizados de nuevo a la escala original de ventas
mu = (max_sales - min_sales) * mu_norm + min_sales
var = (max_sales - min_sales) ** 2 * var_norm
sigma = np.sqrt(var)

print("mu normalizada, varianza normalizada:", mu_norm, var_norm)

# ------ Cálculo de nómina y otros gastos ------
fact = 1.15  # Factor de aumento de salarios

# Cálculo de salarios para distintos roles en el supermercado
sal_cajeros = 258.25
num_cajeros = 30
dias_t = 24
tot_sal_caj = sal_cajeros * num_cajeros * dias_t * fact

sal_conserjes = 5000
num_conserjes = 20
tot_sal_conserjes = sal_conserjes * num_conserjes * fact

gerente = 100000

sub_gerentes = 45000
num_subgerentes = 4
tot_sal_sub = sub_gerentes * num_subgerentes

sal_almacenista = 262.13
num_almacenista = 40
tot_sal_alm = sal_almacenista * num_almacenista * dias_t * fact

g_pasillo = 264.65
num_pasillos = 40
tot_pasillo = g_pasillo * num_pasillos * fact

# Cálculo de la nómina total sumando todos los roles
nomina_tot = tot_sal_caj + tot_sal_conserjes + tot_sal_sub + gerente + tot_sal_sub + tot_sal_alm + tot_pasillo
print("Nómina total:", nomina_tot)

# Cálculo del gasto en electricidad (aproximado)
gasto_luz = 120 * 2000 * 24 * 3.134 * 30
print("Gasto luz:", gasto_luz)

# Cálculo de los gastos totales (nómina + electricidad) y los ingresos requeridos
gastos_tot = gasto_luz + nomina_tot
ingreso = gastos_tot + 1500000  # Margen adicional de 1,500,000
print("Gastos totales:", gastos_tot)
print("Ingresos requeridos:", ingreso)

# ------ Cálculo del porcentaje de aprobación necesario ------
omega = norm.ppf(0.1)  # Cuantil para una distribución normal al 10%
a_ = mu ** 2
b_ = -2 * mu * ingreso - omega ** 2 * sigma ** 2
c_ = ingreso ** 2

# Soluciones para el porcentaje de aprobación
N1 = (-b_ + np.sqrt(b_ ** 2 - 4 * a_ * c_)) / (2 * a_)
N2 = (-b_ - np.sqrt(b_ ** 2 - 4 * a_ * c_)) / (2 * a_)

# Selección de N con base en la condición
if (ingreso / N1 - mu > 0):
    N = N1
else:
    N = N2

# Calcular el porcentaje de aprobación final respecto a una población de 160,000
porc_pob = N / 160000
print("Porcentaje de aprobación:", porc_pob)
