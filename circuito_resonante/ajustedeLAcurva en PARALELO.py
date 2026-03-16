# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 20:09:44 2024
este programa hace ajuste de una curva de resonancia
@author: Maria Laura
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#leemos el archivo

raw_data = pd.read_csv("datos_teleferico_n3300.csv", delimiter= ",")
data = raw_data.copy()


x = data["f generada (Hz)"] #frecuencia
y = data["V (V)"] #voltaje

# Convertir las listas a arrays de NumPy 
x = np.array(x) 
y = np.array(y) 
# Crear una máscara booleana combinada para filtrar valores NaN e Inf en ambos arrays 
mask = np.isfinite(x) & np.isfinite(y) 
# Aplicar la máscara a ambos arrays 
x_clean = x[mask] 
y_clean = y[mask]

# Define la función de lorentz a ajustar
def Lorentz(x, A, x_0, gamma, offset):
    return A / ( (x - x_0)**2 + gamma**2) + offset

#p0 = (np.max(y), 2 * np.pi * 600000, 0)  # Valores iniciales apropiados


# Define los límites para A, x_0, gamma y offset 
lower_bounds = [0, 0, 0, -np.inf] 
upper_bounds = [np.inf, np.inf, np.inf, np.inf] 
# Proporciona valores iniciales manualmente 
p0 = [10000, 509975.65, 0.001, np.mean(y)]

# Realiza el ajuste de curva
popt, pcov = curve_fit(Lorentz, x_clean, y_clean, p0, bounds=(lower_bounds, upper_bounds))

# Parámetros del ajuste
A, x_0, gamma, offset = popt

# Los errores estándar son la raíz cuadrada de los elementos diagonales de la matriz de covarianza
perr = np.sqrt(np.diag(pcov))
#Errores de cada parametro de ajuste
A_err, x_0_err, gamma_err, off_err = perr

# Calcula los valores ajustados
y_fit = Lorentz(x, A, x_0, gamma, offset)

# Grafica los datos originales y la curva ajustada
plt.scatter(x, y, label='Datos originales', color='blue')
plt.plot(x, y_fit, label='Curva ajustada', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ajuste de curva sinusoidal')
plt.legend()
plt.show()

print("-----Parámetros del ajuste:")
print(f"A: {A:.8f}")
print(f"x_0: {x_0:.7f}")
print(f"gamma: {gamma:.8f}")
print(f"offset: {offset:.8f}")


print("-----Errores estándar:")
print(f"A: {A_err:.8f}")
print(f"x_0: {x_0_err:.7f}")
print(f"gamma: {gamma_err:.8f}")
print(f"offset: {off_err:.8f}")
