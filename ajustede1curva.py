# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 23:00:20 2024

@author: Maria Laura

Este programa hace la cuerva de ajuste senusoidal para un solo archivo csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#leemos el archivo

raw_data = pd.read_csv("lacopia.csv", delimiter= ",")
data = raw_data.copy()


x = data.x
y = data.y



# Define la función sinusoidal a ajustar
def sinusoidal(x, A, omega, phi, offset):
    return A * np.sin(omega * x + phi) + offset

#p0 = (np.max(y), 2 * np.pi * 600000, 0)  # Valores iniciales apropiados

def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])
    return guess


# Realiza el ajuste de curva
# 1/ periodo aproximado 105307.4979
popt, _ = curve_fit(sinusoidal, x, y, p0=fit_sin(x, y))

# Parámetros del ajuste
A, omega, phi, offset = popt

# Calcula los valores ajustados
y_fit = sinusoidal(x, A, omega, phi, offset)

# Grafica los datos originales y la curva ajustada
plt.scatter(x, y, label='Datos originales', color='blue')
plt.plot(x, y_fit, label='Curva ajustada', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ajuste de curva sinusoidal')
plt.legend()
plt.show()

print(f"Parámetros del ajuste:")
print(f"A: {A:.8f}")
print(f"omega: {omega:.2f}")
print(f"phi: {phi:.8f}")
print(f"offset: {offset:.8f}")

