# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 22:52:59 2024

@author: Maria Laura
Este codigo va a graficar la funcion dada en el intervalo dadoy ademas de volver sendos csv con los datos
la funcón dada es el comportamiento del circuito LC en paralelo, amplitud de voltaje como función de la frecuencia
"""

import matplotlib.pyplot as plt
import numpy as np
import csv

# Define los límites y resolución de la gráfica
f_min = 509975  # frecuencia mínima para graficar
f_max = 509976  # frecuencia máxima para graficar
Resolucion = 0.004 # mínimo paso de frecuencia que se da con la fuente en Hz
N = int((f_max - f_min) / Resolucion)

# Define las constantes C, R, S, E, G
c_values = [299792471.69, 299792473.97, 299792475.47]  # velocidad de la luz en metros por segundo
names = ["carmen de urea", "USB", "teleferico"]
#R = 0.01680  # resistencia en ohms
L = 0.0011  # inductancia en henrio
d = 0.001  # distancia entre las dos placas del condensador en metros
A = 0.01  # área de las placas del condensador en metros cuadrados
I = 0.00153  # amplitud de corriente aplicados al circuito
vs = 0.306 #amplitud del voltaje de la fuente en Voltios


# Define la función V(f), que es la amplitud del voltaje en función de la frecuencia
#para el caso en que L y C se conectan en paralelo
def V(f, c, L, I, d, A):
    denominador = np.power(1 - np.pi * f **2 * L * A / (c**2 * 10**-7 * d ), 2)
    denominador = np.sqrt(denominador)
    numerador = I * 2 * np.pi * f *L 
    if denominador == 0:
        return vs
    else:
        return numerador/denominador

# Crea un rango de valores de frecuencia
f = np.linspace(f_min, f_max, N)

# Crea la figura y el eje
fig, ax = plt.subplots()

# Realiza las iteraciones y guarda los resultados en archivos CSV
for i in range(3):
    c = c_values[i]
    y = V(f, c, L, I, d, A)

    # Guarda los datos en un archivo CSV
    filename = f'datos_{names[i]}.csv'
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['f (Hz)', 'V (V)'])
        for f_val, y_val in zip(f, y):
            writer.writerow([f'{f_val:6f}', f'{y_val:.5f}'])
    
    # Grafica los resultados
    ax.plot(f, y, label=f'{names[i]} {c} m/s')

ax.set_title('Respuesta del circuito')
ax.set_xlabel('f (Hz)')
ax.set_ylabel('V (V)')
ax.legend()

# Muestra el gráfico
plt.show()

print("Cantidad de puntos graficados:", N)
