# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 23:00:20 2024

@author: Maria Laura

Este programa que hace el ajuste de curva sinusoidal para todos los archivos 
y devulve un csv con los parametros de la funcion de ajuste de cada uno
"""
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit




#Creamos el data frame donde guardaremos todos los datos

Table = pd.DataFrame(columns= ("File3", "A3", "Omega3", "Phi3", "File4", "A4", "Omega4", "Phi4"))

# Definimos la función sinusoidal que se usara para ajustar
def sinusoidal(x, A, omega, phi, offset):
    return A * np.sin(omega * x + phi) + offset

#p0 = (np.max(y), 2 * np.pi * 600000, 0)  
# Calculo de los valores iniciales apropiadospara la curva

def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # asume espacio uniforme
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluye la frecuencia cero que se realciona con el offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])
    return guess


# Itera sobre los números 0 y 6
for i in range(10, 15):
    # Itera sobre los números 1 y 3 (estos son los canales usados)
    Lista = []
    for j in range(1, 3):
        # Crea el nombre del archivo
        filename = f"F00{i}CH{j}.csv"
        raw_data = pd.read_csv(filename, delimiter = ",")
        data = raw_data.copy()
        #extrae las columnas con los datos utiles
        x = data.x
        y = data.y
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
        plt.title(f'Ajuste de curva sinusoidal para {filename}') 
        plt.legend() 
        plt.show()

        #guardamos los parametros de ajuste en la lista
        Lista.extend([filename, A, omega, phi])
    #agregamos la lista con los parametros como una fila mas en el data frame
    df = pd.DataFrame([Lista], columns= Table.columns)    
    Table = Table.append(df, ignore_index=True)

#finalmente devolvemos el archivo con los datos de ajuste de todas las mediciones hechas

Table.to_csv("datos_de_ajuste.csv",index=False)

