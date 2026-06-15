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

raw_data = pd.read_csv(r"C:\Users\mlrg1\Documents\TESIS_CON_GREAVES\Experiemento con circuito resonante\CircuitoFINAL\3Mar26-1.csv", delimiter= ",")
data = raw_data.copy()


x = data["Frecuencia"] #frecuencia
y = data["Voltaje"] #voltaje

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



# Define los límites para A, x_0, gamma y offset 
lower_bounds = [0, 0, 0, -np.inf] 
upper_bounds = [np.inf, np.inf, np.inf, np.inf] 
# Proporciona valores iniciales manualmente 
#p0 = [213000, 535000, 500000, np.mean(y)]

# =============================================================================
# ESTIMACIÓN AUTOMÁTICA DE PARÁMETROS INICIALES (p0)
# =============================================================================

# 1. Estimación del Offset (Fondo): Usamos el percentil 10 para evitar ruidos mínimos
guess_offset = np.percentile(y_clean, 10)

# 2. Estimación de x_0 (Centro del pico): Frecuencia donde el voltaje es máximo
idx_max = np.argmax(y_clean)
guess_x0 = x_clean[idx_max]

# Altura neta del pico sobre el fondo
y_max_neta = y_clean[idx_max] - guess_offset

# 3. Estimación de gamma (Ancho medio): Buscamos el FWHM
# Encontramos los puntos en X que están por encima de la mitad de la altura neta
mitad_altura = guess_offset + (y_max_neta / 2.0)
indices_dentro_pico = np.where(y_clean >= mitad_altura)[0]

if len(indices_dentro_pico) > 1:
    # FWHM aproximado es la diferencia entre la frecuencia máxima y mínima en esa zona
    fwhm = x_clean[indices_dentro_pico[-1]] - x_clean[indices_dentro_pico[0]]
    guess_gamma = fwhm / 2.0
else:
    # Salvaguarda por si el barrido es muy rústico o tiene pocos puntos
    guess_gamma = (np.max(x_clean) - np.min(x_clean)) / 10.0

# 4. Estimación de A: Basado en la relación matemática A = H_neta * gamma^2
guess_A = y_max_neta * (guess_gamma ** 2)

# Empaquetamos en el orden de la función: Lorentz(x, A, x_0, gamma, offset)
p0 = [guess_A, guess_x0, guess_gamma, guess_offset]



# Realiza el ajuste de curva
popt, pcov = curve_fit(Lorentz, x_clean, y_clean, p0, bounds=(lower_bounds, upper_bounds),maxfev=20000)

# Parámetros del ajuste
A, x_0, gamma, offset = popt

# Los errores estándar son la raíz cuadrada de los elementos diagonales de la matriz de covarianza
perr = np.sqrt(np.diag(pcov))
#Errores de cada parametro de ajuste
A_err, x_0_err, gamma_err, off_err = perr

# Calcula los valores ajustados
y_fit = Lorentz(x, A, x_0, gamma, offset)

# Grafica los datos originales y la curva ajustada
plt.scatter(x, y, label='Datos originales', color='#440154')
plt.plot(x, y_fit, label='Curva ajustada', color='#22a884')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Voltaje (V)')
plt.title('Ajuste de curva de resonancia')
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